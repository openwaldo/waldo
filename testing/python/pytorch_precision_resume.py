# Copyright (c) 2026 OpenWALDO Project contributors
# SPDX-License-Identifier: Apache-2.0
"""BF16 numerical and fresh-process checkpoint-resume qualification.

Uses native Trainer math/save/restore. Checkpoint tests deliberately SIGKILL only
an experiment process group after all ranks acknowledge a committed checkpoint.
No CLI transaction, ingest/export audit, or mid-accumulation resume is claimed.
"""
import argparse
import copy
import datetime
import hashlib
import json
import os
import re
from pathlib import Path
import signal
import subprocess
import sys
import time

import pytorch_fixed_batch as base

base.SOURCE_PATHS = base.SOURCE_PATHS + ["testing/python/pytorch_precision_resume.py"]
# Proposed bounded eight-update BF16-versus-FP32 policy, not an upstream standard.
# Frozen before GPU qualification; failures must not silently relax these values.
BF16_TOLERANCES = {
    "logits": [1e-2, 2e-2], "loss": [5e-3, 2e-3],
    "gradients": [2e-3, 5e-2], "weights": [2e-3, 2e-2],
    "exp_avg": [5e-4, 5e-2], "exp_avg_sq": [5e-5, 5e-2],
}
POLICY = {"schema": 3, "bf16_atol_rtol": BF16_TOLERANCES,
          "bf16_reference": "same-weight independent FP32 functional model; explicit AdamW on measured gradients",
          "bf16_topology": "separate unchanged tensor tolerances against the singleton trajectory",
          "resume": "bitwise equal tensors, RNG, counters and sample order at the same topology",
          "interruption": "rank-0 SIGKILL followed by verified worker shutdown",
          "checkpoint_step": 4, "total_steps": 8, "resume_dropout": 0.1,
          "resume_precisions": ["float32", "bfloat16"]}


def runtime():
    global torch
    base.load_runtime()
    torch = base.torch


def sources():
    return {p: base.digest(base.ROOT / p) for p in base.SOURCE_PATHS}


def seal(directory):
    directory = Path(directory)
    files = sorted(p for p in directory.rglob("*") if p.is_file() and p.name != "SHA256SUMS")
    (directory / "SHA256SUMS").write_text("".join(f"{base.digest(p)}  {p.relative_to(directory)}\n" for p in files))


def checkpoint_receipt(directory):
    directory = Path(directory)
    receipt = json.loads((directory / "checkpoint-ready.json").read_text())
    expected = {f"checkpoints/step-00000004/{n}" for n in ("model.safetensors", "runtime.pt", "state.json")}
    if set(receipt["files"]) != expected or receipt["item"]["step"] != 4:
        raise ValueError("unexpected checkpoint receipt")
    for name, info in receipt["files"].items():
        p = directory / name
        if p.is_symlink() or p.stat().st_size != info["bytes"] or base.digest(p) != info["sha256"]:
            raise ValueError(f"checkpoint integrity failure: {name}")
    return receipt


def worker(args):
    base.require_cuda_authorization(args.device)
    fixture = base.verify_fixture(args.fixture)
    if fixture["steps"] != 8:
        raise ValueError("this policy requires the staged eight-update fixture")
    rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    base.write_json(output / f"participant-{rank}.json", {"pid": os.getpid(), "pgid": os.getpgid(0), "rank": rank})
    runtime()
    resume_mode = args.phase != "bf16"
    if args.device == "cpu" and (args.strategy != "data-parallel" or (resume_mode and world != 1)):
        raise ValueError("CPU resume tests use the unmodified serial CPU Trainer")
    if args.device == "cuda":
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    torch.distributed.init_process_group(backend="nccl" if args.device == "cuda" else "gloo",
                                        timeout=datetime.timedelta(seconds=120))
    try:
        module = base.load_worker()
        module.IS_PRIMARY = rank == 0
        begin = base.begin_frame(copy.deepcopy(fixture), args.fixture, args.accumulation, args.strategy)
        begin["parameters"]["compute_precision"] = args.precision
        if resume_mode:
            begin["architecture"]["dropout"] = POLICY["resume_dropout"]
            begin["architecture_sha256"] = hashlib.sha256(json.dumps(begin["architecture"], sort_keys=True).encode()).hexdigest()
            begin["parameters"]["checkpoint_every"] = 4
        if args.phase == "resume":
            receipt = checkpoint_receipt(args.prefix)
            begin["resume"] = {"step": 4, "tokens": receipt["item"]["tokens"],
                               "paths": [str(Path(args.prefix).resolve() / n) for n in sorted(receipt["files"])],
                               "checkpoint": receipt["item"], "checkpoints": [receipt["item"]]}
        output = Path(args.output)
        output.mkdir(parents=True, exist_ok=True)
        trainer = module.Trainer(begin, str(output), "artifacts", "torchtitan" if args.device == "cuda" else "cpu")
        if args.device == "cpu" and not resume_mode:
            trainer.distributed, trainer.rank, trainer.world_size = True, rank, world
            trainer.batch_size = trainer.global_micro_batch_size // world
            trainer.model = torch.nn.parallel.DistributedDataParallel(
                trainer.model, broadcast_buffers=False, gradient_as_bucket_view=True, static_graph=True)
            trainer.compiled_model = trainer.model
        if any(p.dtype != torch.float32 for _, p in base.named_parameters(trainer)):
            raise AssertionError("master weights must remain FP32")

        def rng():
            return {"cpu": torch.get_rng_state().cpu(),
                    "cuda": torch.cuda.get_rng_state(trainer.device).cpu() if args.device == "cuda" else None}

        restored_rng = rng() if args.phase == "resume" else None
        snapshots, observed_rows, local_logits, logit_dtypes = [], [], {}, set()
        rows = fixture["cases"][args.case]
        lookup = {tuple(r["tokens"]): i for i, r in enumerate(rows)}
        original_batch = trainer.train_batch

        def record_batch(final=False):
            # Replayed data must position the cursor without advancing math/RNG.
            ids = [lookup[tuple(item[0])] for item in trainer.batch]
            if not trainer.replay_micro_batches:
                observed_rows.extend(ids)
            trainer._qualification_batch_ids = ids
            return original_batch(final=final)

        trainer.train_batch = record_batch

        def forward_hook(model, inputs, value):
            logit_dtypes.add(str(value.dtype))
            for row_id, logits in zip(trainer._qualification_batch_ids, value):
                local_logits[row_id] = base.tensor_cpu(logits).float()

        trainer.model.register_forward_hook(forward_hook)

        def before_step(optimizer, positional, keywords):
            gradients = {n: base.tensor_cpu(p.grad) for n, p in base.named_parameters(trainer)}
            if any(t.dtype != torch.float32 for t in gradients.values()):
                raise AssertionError("master gradients must remain FP32")
            snapshots.append({"gradients": gradients, "logits": dict(local_logits),
                              "rows": list(observed_rows), "valid_tokens": trainer.accumulated_tokens,
                              "loss": trainer.accumulated_loss_sum / trainer.accumulated_tokens,
                              "step": trainer.step_number + 1})
            observed_rows.clear()
            local_logits.clear()

        def after_step(optimizer, positional, keywords):
            snap = snapshots[-1]
            snap["weights"] = {n: base.tensor_cpu(p) for n, p in base.named_parameters(trainer)}
            for key in ("exp_avg", "exp_avg_sq"):
                snap[key] = {n: base.tensor_cpu(optimizer.state[p][key]) for n, p in base.named_parameters(trainer)}
                if any(t.dtype != torch.float32 for t in snap[key].values()):
                    raise AssertionError("optimizer moments must remain FP32")
            snap["optimizer_steps"] = {n: int(optimizer.state[p]["step"].item()) for n, p in base.named_parameters(trainer)}

        trainer.optimizer.register_step_pre_hook(before_step)
        trainer.optimizer.register_step_post_hook(after_step)

        def save_rank():
            torch.save({"schema": 1, "case": args.case, "world_size": world, "rank": rank,
                        "accumulation": args.accumulation, "device": args.device, "strategy": args.strategy,
                        "precision": args.precision, "phase": args.phase, "snapshots": snapshots,
                        "consumed_tokens": trainer.consumed_tokens, "consumption": trainer.consumed_by_corpus,
                        "rng": rng(), "restored_rng": restored_rng, "logit_dtypes": sorted(logit_dtypes),
                        "fixture_sha256": base.digest(Path(args.fixture) / "MANIFEST.json"),
                        "sources": sources(), "policy": POLICY, "torch": str(torch.__version__),
                        "cuda_runtime": torch.version.cuda, "python": sys.version}, output / f"rank-{rank}.pt")

        for row in rows:
            trainer.add_sequence(row["tokens"], row["mask"], ["fixture"] * 8)
            if args.phase == "prefix" and trainer.step_number == 4:
                save_rank()
                torch.distributed.barrier()
                if rank == 0:
                    item = trainer.checkpoints[-1]
                    files = {}
                    for name in ("model.safetensors", "runtime.pt", "state.json"):
                        p = output / "checkpoints/step-00000004" / name
                        files[str(p.relative_to(output))] = {"bytes": p.stat().st_size, "sha256": base.digest(p)}
                    # Native save_checkpoint has committed/fsynced the payloads.
                    module.write_json(str(output / "checkpoint-ready.json"), {"item": item, "files": files, "killed_rank": 0, "killed_pid": os.getpid()})
                    os.kill(os.getpid(), signal.SIGKILL)
                while True:
                    time.sleep(1)  # torchrun shuts down peers after rank 0 dies.
        if trainer.step_number != 8 or trainer.accumulation_number or trainer.batch:
            raise AssertionError("incomplete optimizer trajectory")
        save_rank()
    finally:
        torch.distributed.destroy_process_group()


def terminate_group(child, sig=signal.SIGTERM):
    try:
        os.killpg(child.pid, sig)
    except ProcessLookupError:
        pass
    try:
        child.wait(timeout=10)
    except subprocess.TimeoutExpired:
        os.killpg(child.pid, signal.SIGKILL)
        child.wait()


def cleanup_participants(directory):
    """torchrun workers have their own sessions; never signal unrelated PIDs."""
    def live(pid, directory):
        p = subprocess.run(["ps", "-p", str(pid), "-o", "stat=", "-o", "command="], text=True, capture_output=True)
        if p.returncode or not p.stdout.strip() or p.stdout.lstrip().startswith("Z"):
            return False
        if str(Path(directory).resolve()) not in p.stdout or "pytorch_precision_resume.py" not in p.stdout:
            raise RuntimeError(f"PID {pid} no longer identifies this test; refusing to signal it")
        return True
    for path in Path(directory).glob("participant-*.json"):
        item = json.loads(path.read_text())
        pid, group = item["pid"], item["pgid"]
        if not live(pid, directory):
            continue
        if pid != group or os.getpgid(pid) != group or group == os.getpgrp():
            raise RuntimeError("unexpected participant process group")
        os.killpg(group, signal.SIGTERM)
        deadline = time.monotonic() + 5
        while live(pid, directory) and time.monotonic() < deadline:
            time.sleep(0.1)
        if live(pid, directory):
            os.killpg(group, signal.SIGKILL)
            deadline = time.monotonic() + 5
            while live(pid, directory) and time.monotonic() < deadline:
                time.sleep(0.1)
        if live(pid, directory):
            raise RuntimeError("test participant did not terminate")


def launch_worker(args, directory, precision, case, world, accumulation, phase, timeout, prefix=None):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    command = [sys.executable, "-m", "torch.distributed.run", "--standalone", "--local-addr=127.0.0.1",
               "--nnodes=1", f"--nproc-per-node={world}", str(Path(__file__).resolve()), "worker",
               "--fixture", str(Path(args.fixture).resolve()), "--output", str(directory.resolve()),
               "--device", args.device, "--strategy", args.strategy, "--precision", precision,
               "--case", case, "--accumulation", str(accumulation), "--phase", phase]
    if prefix is not None:
        command += ["--prefix", str(Path(prefix).resolve())]
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", CUBLAS_WORKSPACE_CONFIG=":4096:8")
    if args.device == "cpu":
        env["CUDA_VISIBLE_DEVICES"] = ""
    receipt = {"command": command, "phase": phase, "timeout_seconds": timeout,
               "started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(), "status": "running"}
    base.write_json(directory / "attempt.json", receipt)
    started = time.monotonic()
    child = None
    try:
        with (directory / "worker.log").open("w") as log:
            child = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            if phase == "prefix":
                deadline = started + timeout
                while not (directory / "checkpoint-ready.json").exists():
                    if child.poll() is not None:
                        raise RuntimeError(f"prefix exited {child.returncode} before checkpoint commit")
                    if time.monotonic() >= deadline:
                        raise TimeoutError("checkpoint-ready deadline exceeded")
                    time.sleep(0.1)
                checkpoint_receipt(directory)
                if any(not (directory / f"rank-{rank}.pt").is_file() for rank in range(world)):
                    raise AssertionError("not all ranks reached the checkpoint boundary")
                child.wait(timeout=max(1, deadline - time.monotonic()))
                log.flush()
                if child.returncode != 1 or not re.search(r"exitcode\s*:\s*-9", (directory / "worker.log").read_text()):
                    raise AssertionError("prefix lacks torchrun evidence of rank SIGKILL")
                cleanup_participants(directory)
                receipt.update(status="expected-interruption", signal="SIGKILL", checkpoint_step=4, no_live_participants=True)
            else:
                if child.wait(timeout=timeout) != 0:
                    raise RuntimeError(f"worker exit {child.returncode}; see {directory}/worker.log")
                receipt["status"] = "pass"
    except BaseException as error:
        if child is not None:
            terminate_group(child)
        receipt.update(status="fail", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        try:
            cleanup_participants(directory)
        except BaseException as error:
            receipt.update(status="fail", cleanup_error=f"{type(error).__name__}: {error}")
            raise
        finally:
            receipt["elapsed_seconds"] = time.monotonic() - started
            receipt["exit_code"] = child.returncode if child is not None else None
            base.write_json(directory / "attempt.json", receipt)
    return receipt


def load_rank(args, directory, rank, world, precision, phase):
    data = torch.load(Path(directory) / f"rank-{rank}.pt", map_location="cpu", weights_only=True)
    expected = {"rank": rank, "world_size": world, "device": args.device, "strategy": args.strategy,
                "precision": precision, "phase": phase, "sources": sources(), "policy": POLICY,
                "fixture_sha256": base.digest(Path(args.fixture) / "MANIFEST.json")}
    if any(data.get(k) != v for k, v in expected.items()):
        raise AssertionError("rank provenance mismatch")
    if data["logit_dtypes"] != ["torch." + precision]:
        raise AssertionError("requested compute precision did not execute")
    return data


def check_bf16(args, fixture, case, world, accumulation, reference, singleton, write=True):
    path = Path(args.output) / f"{case}-w{world}-a{accumulation}"
    attempt = json.loads((path / "attempt.json").read_text())
    if attempt["status"] != "pass" or attempt["exit_code"] != 0:
        raise AssertionError("BF16 process did not exit successfully")
    initial = base.oracle.load_checkpoint_safetensors(Path(args.fixture) / "initial.safetensors")
    errors, trajectory_errors, trajectory_failures = {}, {}, []
    rank_zero = None
    opt, lr = fixture["optimizer"], fixture["learning_rate"]
    for rank in range(world):
        data = load_rank(args, path, rank, world, "bfloat16", "bf16")
        if data["case"] != case or data["accumulation"] != accumulation or [s["step"] for s in data["snapshots"]] != list(range(1, 9)):
            raise AssertionError("BF16 case/step coverage mismatch")
        previous = initial
        first = {n: torch.zeros_like(p) for n, p in initial.items()}
        second = {n: torch.zeros_like(p) for n, p in initial.items()}
        total_tokens = 0
        for step, snap in enumerate(data["snapshots"]):
            rows = fixture["cases"][case][step * 8:(step + 1) * 8]
            ids = list(range(step * 8 + rank, (step + 1) * 8, world))
            if snap["rows"] != ids or sorted(snap["logits"]) != ids:
                raise AssertionError("BF16 sample ownership mismatch")
            params = {n: p.clone().requires_grad_(True) for n, p in previous.items()}
            tokens = torch.tensor([r["tokens"] for r in rows], dtype=torch.long)
            mask = torch.tensor([r["mask"] for r in rows], dtype=torch.float32)
            logits = base.oracle.reference_logits(tokens[:, :-1], params, fixture["architecture"])
            loss = base.oracle.masked_reference_loss(logits, tokens[:, 1:], mask)
            loss.backward()
            if snap["valid_tokens"] != int(mask.sum()):
                raise AssertionError("BF16 valid-token accounting mismatch")
            total_tokens += int(mask.sum())
            prefix = f"rank{rank}/step{step}"
            base.compare_tensor(prefix + "/loss", torch.tensor(snap["loss"]), loss.detach(), BF16_TOLERANCES["loss"], errors)
            for row_id in ids:
                base.compare_tensor(f"{prefix}/logits/{row_id}", snap["logits"][row_id], logits[row_id - step * 8].detach(), BF16_TOLERANCES["logits"], errors)
            if set(snap["optimizer_steps"]) != set(initial) or any(v != step + 1 for v in snap["optimizer_steps"].values()):
                raise AssertionError("BF16 optimizer step counter mismatch")
            for key in ("gradients", "weights", "exp_avg", "exp_avg_sq"):
                if set(snap[key]) != set(initial) or any(p.dtype != torch.float32 for p in snap[key].values()):
                    raise AssertionError("BF16 master tensor coverage/dtype mismatch")
            for name in sorted(initial):
                gradient = snap["gradients"][name]
                base.compare_tensor(f"{prefix}/gradients/{name}", gradient, params[name].grad,
                                    BF16_TOLERANCES["gradients"], errors)
                # A small BF16 gradient error can reverse Adam's update near zero.
                # Check optimizer arithmetic on its actual input gradient, not on
                # a different FP32 trajectory. Precision drift is reported below.
                mean = opt["beta1"] * first[name] + (1 - opt["beta1"]) * gradient
                square = opt["beta2"] * second[name] + (1 - opt["beta2"]) * gradient.square()
                expected = previous[name] * (1 - lr * opt["weight_decay"]) - lr * (mean / (1 - opt["beta1"] ** (step + 1))) / ((square / (1 - opt["beta2"] ** (step + 1))).sqrt() + opt["epsilon"])
                for key, wanted in (("weights", expected), ("exp_avg", mean), ("exp_avg_sq", square)):
                    base.compare_tensor(f"{prefix}/{key}/{name}", snap[key][name], wanted,
                                        base.TOLERANCES[key], errors)
            if rank_zero is not None:
                for key in ("gradients", "weights", "exp_avg", "exp_avg_sq", "loss", "optimizer_steps", "valid_tokens"):
                    exact(f"{prefix}/rank_consistency/{key}", snap[key], rank_zero[step][key], [0])
            if singleton is not None:
                baseline = singleton[step]
                comparisons = [("loss", "loss", torch.tensor(snap["loss"]), torch.tensor(baseline["loss"]))]
                comparisons += [("logits", f"logits/{i}", snap["logits"][i], baseline["logits"][i]) for i in ids]
                for key in ("gradients", "weights", "exp_avg", "exp_avg_sq"):
                    comparisons += [(key, f"{key}/{n}", snap[key][n], baseline[key][n]) for n in sorted(initial)]
                for key, name, actual, wanted in comparisons:
                    try:
                        base.compare_tensor(f"{prefix}/{name}", actual, wanted, BF16_TOLERANCES[key], trajectory_errors)
                    except AssertionError as error:
                        trajectory_failures.append(str(error))
            previous, first, second = snap["weights"], snap["exp_avg"], snap["exp_avg_sq"]
        if data["consumed_tokens"] != total_tokens:
            raise AssertionError("BF16 consumed tokens mismatch")
        if rank == 0:
            rank_zero = data["snapshots"]
    report = {"arithmetic_status": "pass", "topology_status": "fail" if trajectory_failures else "pass",
              "checks": len(errors), "maximum_absolute_errors": errors,
              "topology_errors": trajectory_errors, "topology_failures": trajectory_failures}
    if write:
        base.write_json(path / "comparisons.json", report)
    return rank_zero, len(errors), report["topology_status"]


def exact(name, left, right, count):
    if isinstance(left, torch.Tensor):
        if not isinstance(right, torch.Tensor) or left.shape != right.shape or left.dtype != right.dtype:
            raise AssertionError(f"{name}: tensor layout differs")
        if left.is_floating_point() and (not torch.isfinite(left).all() or not torch.isfinite(right).all()):
            raise AssertionError(f"{name}: nonfinite tensor")
        if not torch.equal(left, right):
            raise AssertionError(f"{name}: continuation is not bitwise equal; max error {base.oracle.maximum_error(left, right)}")
        count[0] += 1
    elif isinstance(left, dict):
        if not isinstance(right, dict) or set(left) != set(right):
            raise AssertionError(f"{name}: keys differ")
        for key in left:
            exact(f"{name}/{key}", left[key], right[key], count)
    elif isinstance(left, (list, tuple)):
        if type(left) != type(right) or len(left) != len(right):
            raise AssertionError(f"{name}: sequence differs")
        for i, (a, b) in enumerate(zip(left, right)):
            exact(f"{name}/{i}", a, b, count)
    elif left != right:
        raise AssertionError(f"{name}: {left!r} != {right!r}")


def check_resume(args, fixture, precision, world, write=True):
    directory = Path(args.output) / f"{precision}-w{world}"
    receipt = checkpoint_receipt(directory / "prefix")
    attempt = json.loads((directory / "prefix/attempt.json").read_text())
    if attempt["status"] != "expected-interruption" or attempt["exit_code"] != 1 or not attempt.get("no_live_participants"):
        raise AssertionError("resume lacks a real committed-checkpoint interruption")
    for phase in ("control", "resume"):
        result = json.loads((directory / phase / "attempt.json").read_text())
        if result["status"] != "pass" or result["exit_code"] != 0:
            raise AssertionError("control/resume process did not exit successfully")
    saved_runtime = torch.load(directory / "prefix/checkpoints/step-00000004/runtime.pt", map_location="cpu", weights_only=True)
    initial = base.oracle.load_checkpoint_safetensors(Path(args.fixture) / "initial.safetensors")
    count = [0]
    for rank in range(world):
        control = load_rank(args, directory / "control", rank, world, precision, "control")
        prefix = load_rank(args, directory / "prefix", rank, world, precision, "prefix")
        resumed = load_rank(args, directory / "resume", rank, world, precision, "resume")
        if [s["step"] for s in control["snapshots"]] != list(range(1, 9)):
            raise AssertionError("control step coverage mismatch")
        if [s["step"] for s in prefix["snapshots"]] != list(range(1, 5)) or [s["step"] for s in resumed["snapshots"]] != list(range(5, 9)):
            raise AssertionError("resume step coverage mismatch")
        for data in (control, prefix, resumed):
            if data["case"] != "uneven" or data["accumulation"] != 4 // world:
                raise AssertionError("resume case/accumulation identity mismatch")
        for step, snap in enumerate(control["snapshots"]):
            for field in ("weights", "gradients", "exp_avg", "exp_avg_sq"):
                if set(snap[field]) != set(initial) or any(t.dtype != torch.float32 for t in snap[field].values()):
                    raise AssertionError("resume tensor coverage or master dtype mismatch")
            if set(snap["optimizer_steps"]) != set(initial) or any(v != step + 1 for v in snap["optimizer_steps"].values()):
                raise AssertionError("resume optimizer step counter mismatch")
            expected_rows = list(range(step * 8 + rank, (step + 1) * 8, world))
            if snap["rows"] != expected_rows or sorted(snap["logits"]) != expected_rows:
                raise AssertionError("control sample accounting mismatch")
            valid = sum(sum(r["mask"]) for r in fixture["cases"]["uneven"][step * 8:(step + 1) * 8])
            if snap["valid_tokens"] != valid:
                raise AssertionError("control token accounting mismatch")
        exact(f"rank{rank}/prefix", prefix["snapshots"], control["snapshots"][:4], count)
        exact(f"rank{rank}/continuation", resumed["snapshots"], control["snapshots"][4:], count)
        exact(f"rank{rank}/restored_rng", resumed["restored_rng"], saved_runtime["random_states"][rank], count)
        local_tokens = sum(sum(r["mask"]) for i, r in enumerate(fixture["cases"]["uneven"]) if i % world == rank)
        if control["consumption"] != {"fixture": local_tokens}:
            raise AssertionError("per-rank corpus accounting mismatch")
        for field in ("rng", "consumed_tokens", "consumption"):
            exact(f"rank{rank}/{field}", resumed[field], control[field], count)
        if resumed["consumed_tokens"] != 288 or prefix["consumed_tokens"] != receipt["item"]["tokens"]:
            raise AssertionError("total consumed tokens mismatch")
    for phase in ("control", "resume"):
        state = json.loads((directory / phase / "checkpoints/step-00000008/state.json").read_text())
        if state["step"] != 8 or state["consumed_tokens"] != 288:
            raise AssertionError("final native checkpoint metadata mismatch")
    a = base.oracle.load_checkpoint_safetensors(directory / "control/checkpoints/step-00000008/model.safetensors")
    b = base.oracle.load_checkpoint_safetensors(directory / "resume/checkpoints/step-00000008/model.safetensors")
    exact("native_final_weights", a, b, count)
    if write:
        base.write_json(directory / "comparisons.json", {"status": "pass", "bitwise_tensor_checks": count[0], "checkpoint_step": 4})
    return count[0]


def run(args):
    base.require_cuda_authorization(args.device)
    fixture = base.verify_fixture(args.fixture)
    if fixture["steps"] != 8:
        raise ValueError("requires the eight-update fixture")
    base.matrix(args.world_sizes)
    if args.world_sizes != sorted(args.world_sizes) or args.world_sizes[0] != 1:
        raise ValueError("ascending world sizes starting with 1 required")
    if args.device == "cpu" and args.mode == "resume" and args.world_sizes != [1]:
        raise ValueError("CPU resume supports world size 1 only")
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    summary = {"schema": 1, "status": "running", "mode": args.mode, "device": args.device,
               "strategy": args.strategy, "world_sizes": args.world_sizes, "policy": POLICY,
               "sources": sources(), "fixture_sha256": base.digest(Path(args.fixture) / "MANIFEST.json"),
               "started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
               "cases": [], "rank_weighted_seconds": 0.0, "cuda_qualified": False}
    base.write_json(output / "summary.json", summary)
    try:
        runtime()
        if args.device == "cuda":
            if not os.environ.get("CUDA_VISIBLE_DEVICES") or torch.cuda.device_count() != max(args.world_sizes):
                raise ValueError("visible GPUs must exactly match explicitly allocated matrix size")
        summary.update(torch=str(torch.__version__), cuda_runtime=torch.version.cuda, python=sys.version)
        references = {c: base.reference(fixture, args.fixture, c) for c in fixture["cases"]} if args.mode == "bf16" else {}
        if references:
            torch.save(references, output / "fp32-oracle.pt")
        cases = [("bfloat16", c, w, a) for c, w, a in base.matrix(args.world_sizes)] if args.mode == "bf16" else [
            (precision, "uneven", w, 4 // w) for precision in POLICY["resume_precisions"] for w in args.world_sizes]
        singleton = {}
        for precision, case, world, accumulation in cases:
            name = f"{case}-w{world}-a{accumulation}" if args.mode == "bf16" else f"{precision}-w{world}"
            item = {"name": name, "precision": precision, "world_size": world, "accumulation": accumulation,
                    "case": case, "status": "running", "attempts": []}
            summary["cases"].append(item)
            base.write_json(output / "summary.json", summary)
            phases = ["bf16"] if args.mode == "bf16" else ["control", "prefix", "resume"]
            for phase in phases:
                directory = output / name if args.mode == "bf16" else output / name / phase
                timeout = args.timeout_seconds
                if args.device == "cuda":
                    timeout = min(timeout, int((args.gpu_hour_budget * 3600 - summary["rank_weighted_seconds"]) / world))
                if timeout <= 0:
                    raise RuntimeError("aggregate GPU budget exhausted")
                try:
                    attempt = launch_worker(args, directory, precision, case, world, accumulation, phase, timeout,
                                            output / name / "prefix" if phase == "resume" else None)
                    item["attempts"].append(attempt)
                finally:
                    if (directory / "attempt.json").exists():
                        attempt = json.loads((directory / "attempt.json").read_text())
                        summary["rank_weighted_seconds"] += attempt["elapsed_seconds"] * world
            if args.mode == "bf16":
                snaps, checks, topology = check_bf16(args, fixture, case, world, accumulation, references[case], singleton.get(case))
                item["topology_status"] = topology
                if world == 1 and accumulation == 1:
                    singleton[case] = snaps
            else:
                # A completed numerical mismatch must not hide independent cells.
                # Launch, interruption, and cleanup errors still stop immediately.
                try:
                    checks = check_resume(args, fixture, precision, world)
                except AssertionError as error:
                    item.update(status="fail", comparison_error=str(error))
                    base.write_json(output / name / "comparisons.json", {
                        "status": "fail", "comparison_error": str(error)})
                    continue
            item.update(status="pass", tensor_checks=checks)
        topology_pass = all(c.get("topology_status", "pass") == "pass" for c in summary["cases"])
        all_pass = topology_pass and all(c["status"] == "pass" for c in summary["cases"])
        summary.update(status="pass" if all_pass else "fail",
                       cuda_qualified=all_pass and args.device == "cuda" and args.world_sizes == [1, 2, 4])
        if args.mode == "bf16":
            summary["arithmetic_status"] = "pass"
        elif not all_pass:
            summary["error"] = "Completed resume comparisons failed the unchanged bitwise policy"
        if not topology_pass:
            summary["error"] = "BF16 arithmetic passed; topology trajectory exceeds declared tolerances"

    except BaseException as error:
        summary.update(status="fail", error=f"{type(error).__name__}: {error}")
        if summary["cases"] and summary["cases"][-1]["status"] == "running":
            summary["cases"][-1].update(status="fail", error=summary["error"])
        raise
    finally:
        summary["finished_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        base.write_json(output / "summary.json", summary)
        seal(output)
    print(json.dumps({"status": summary["status"], "mode": args.mode, "cuda_qualified": summary["cuda_qualified"]}))


def verify(args):
    base.verify_inventory(args.output)
    fixture = base.verify_fixture(args.fixture)
    summary = json.loads((Path(args.output) / "summary.json").read_text())
    if summary["status"] not in ("pass", "fail") or summary["policy"] != POLICY or summary["sources"] != sources():
        raise ValueError("result is not a completed run of this frozen policy/source")
    if summary["fixture_sha256"] != base.digest(Path(args.fixture) / "MANIFEST.json"):
        raise ValueError("fixture identity mismatch")
    args.device, args.strategy, args.mode = summary["device"], summary["strategy"], summary["mode"]
    worlds = summary["world_sizes"]
    base.matrix(worlds)
    if args.mode not in ("bf16", "resume") or args.device not in ("cpu", "cuda") or args.strategy not in ("data-parallel", "fully-sharded-data-parallel"):
        raise ValueError("invalid qualification identity")
    if args.device == "cpu" and args.mode == "resume" and worlds != [1]:
        raise ValueError("invalid CPU resume coverage")
    if summary["cuda_qualified"] != (summary["status"] == "pass" and args.device == "cuda" and worlds == [1, 2, 4]):
        raise ValueError("false CUDA qualification claim")
    runtime()  # CPU verification only, including CUDA result bundles.
    expected = [f"{c}-w{w}-a{a}" for c, w, a in base.matrix(worlds)] if args.mode == "bf16" else [
        f"{p}-w{w}" for p in POLICY["resume_precisions"] for w in worlds]
    allowed_statuses = ("pass",) if args.mode == "bf16" else ("pass", "fail")
    if [c["name"] for c in summary["cases"]] != expected or any(c["status"] not in allowed_statuses for c in summary["cases"]):
        raise ValueError("matrix coverage mismatch")
    if args.mode == "bf16":
        refs = {c: base.reference(fixture, args.fixture, c) for c in fixture["cases"]}
        singleton = {}
        topology_results = []
        for case, world, accumulation in base.matrix(worlds):
            snaps, _, topology = check_bf16(args, fixture, case, world, accumulation, refs[case], singleton.get(case), write=False)
            topology_results.append(topology)
            if world == 1 and accumulation == 1:
                singleton[case] = snaps
        expected_status = "pass" if all(v == "pass" for v in topology_results) else "fail"
        if summary["status"] != expected_status or [c["topology_status"] for c in summary["cases"]] != topology_results:
            raise AssertionError("BF16 topology verdict mismatch")
    else:
        verdicts = []
        for precision in POLICY["resume_precisions"]:
            for world in worlds:
                item = summary["cases"][len(verdicts)]
                try:
                    check_resume(args, fixture, precision, world, write=False)
                except AssertionError as error:
                    if item["status"] != "fail" or item.get("comparison_error") != str(error):
                        raise AssertionError("resume negative verdict mismatch") from error
                    verdicts.append("fail")
                else:
                    if item["status"] != "pass":
                        raise AssertionError("false negative resume verdict")
                    verdicts.append("pass")
        expected_status = "pass" if all(v == "pass" for v in verdicts) else "fail"
        if summary["status"] != expected_status:
            raise AssertionError("resume aggregate verdict mismatch")
    print(json.dumps({"verified": True, "mode": args.mode, "device": args.device, "qualification_status": summary["status"], "cuda_qualified": summary["cuda_qualified"]}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("run", "worker", "verify"):
        p = commands.add_parser(name)
        p.add_argument("--fixture", required=True)
        p.add_argument("--output", required=True)
        if name != "verify":
            p.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
            p.add_argument("--strategy", choices=["data-parallel", "fully-sharded-data-parallel"], default="data-parallel")
        if name == "run":
            p.add_argument("--mode", choices=["bf16", "resume"], required=True)
            p.add_argument("--world-sizes", type=int, nargs="+", default=[1, 2, 4])
            p.add_argument("--timeout-seconds", type=int, default=600)
            p.add_argument("--gpu-hour-budget", type=float, default=2)
        if name == "worker":
            p.add_argument("--precision", choices=["float32", "bfloat16"], required=True)
            p.add_argument("--case", choices=["dense", "uneven"], required=True)
            p.add_argument("--accumulation", type=int, required=True)
            p.add_argument("--phase", choices=["bf16", "control", "prefix", "resume"], required=True)
            p.add_argument("--prefix")
    args = parser.parse_args()
    if args.command == "run":
        if args.timeout_seconds <= 0 or not 0 < args.gpu_hour_budget <= 8:
            parser.error("positive timeout and GPU budget <=8 hours required")
        run(args)
        if json.loads((Path(args.output) / "summary.json").read_text())["status"] != "pass":
            sys.exit(1)
    elif args.command == "worker":
        worker(args)
    else:
        verify(args)


if __name__ == "__main__":
    main()
