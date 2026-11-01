# Copyright (c) 2026 OpenWALDO Project contributors
# SPDX-License-Identifier: Apache-2.0

"""Fixed-global-batch FP32 qualification of the production training worker.

CPU/Gloo uses a test-only placement adapter; CUDA uses Trainer's real TorchTitan
constructor. No training math is replaced. The functional oracle and explicit
AdamW equations are independent of Trainer. This is not CLI/ingestion/resume
acceptance, BF16 qualification, or a throughput benchmark.
"""

import argparse
import ast
import datetime
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import types

ROOT = Path(__file__).resolve().parents[2]
SOURCE_PATHS = [
    "internal/pytorchruntime/model.py",
    "internal/training/workers/pytorch.py",
    "testing/python/pytorch_conformance.py",
    "testing/python/pytorch_fixed_batch.py",
]
# Existing CUDA FP32 oracle thresholds; frozen before any qualification run.
# They are deliberately not presented as BF16 or long-training thresholds.
TOLERANCES = {
    "logits": [8e-5, 2e-4], "loss": [2e-5, 2e-4],
    "gradients": [4e-4, 2e-4], "weights": [2e-5, 2e-4],
    "exp_avg": [2e-5, 2e-4], "exp_avg_sq": [2e-5, 2e-4],
}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def matrix(world_sizes):
    if not world_sizes or len(set(world_sizes)) != len(world_sizes):
        raise ValueError("world sizes must be nonempty and unique")
    if any(w not in (1, 2, 4) for w in world_sizes):
        raise ValueError("only world sizes 1, 2, 4 are qualified")
    # Both world-size scaling alone and scaling with constant local microbatch.
    return [(case, w, a) for case in ("dense", "uneven")
            for w in world_sizes for a in sorted({1, 4 // w})]


def require_cuda_authorization(device):
    if device == "cuda" and os.environ.get("WALDO_FIXED_BATCH_ALLOW_CUDA") != "1":
        raise ValueError("CUDA execution requires WALDO_FIXED_BATCH_ALLOW_CUDA=1 after GPU allocation")


def verify_fixture(directory):
    directory = Path(directory)
    manifest = json.loads((directory / "MANIFEST.json").read_text())
    if manifest.get("schema") != 1 or set(manifest.get("files", {})) != {"fixture.json", "initial.safetensors"}:
        raise ValueError("unexpected fixture manifest")
    for name, item in manifest["files"].items():
        path = directory / name
        if path.is_symlink() or path.stat().st_size != item["bytes"] or digest(path) != item["sha256"]:
            raise ValueError(f"fixture integrity failure: {name}")
    fixture = json.loads((directory / "fixture.json").read_text())
    if fixture["global_batch"] != 8 or fixture["sequence_length"] != 8 or not 1 <= fixture["steps"] <= 32:
        raise ValueError("fixture exceeds bounded qualification dimensions")
    if fixture["tolerances"] != TOLERANCES:
        raise ValueError("fixture tolerances differ from the checked-in contract")
    if set(fixture["cases"]) != {"dense", "uneven"}:
        raise ValueError("missing required fixture case")
    for rows in fixture["cases"].values():
        if len(rows) != fixture["steps"] * fixture["global_batch"]:
            raise ValueError("fixture row count mismatch")
        for row in rows:
            if len(row["tokens"]) != 9 or len(row["mask"]) != 8:
                raise ValueError("invalid sequence dimensions")
            if any(type(t) is not int or not 0 <= t < 259 for t in row["tokens"]):
                raise ValueError("invalid token ID")
            if any(m not in (0, 1) for m in row["mask"]) or not any(row["mask"]):
                raise ValueError("invalid or wholly unsupervised row")
    return fixture


def load_runtime():
    # Lazy imports allow integrity/authorization checks and stdlib unit tests
    # to run without PyTorch, CUDA, network access, or a model download.
    global torch, oracle
    import torch
    import pytorch_conformance as oracle
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.set_float32_matmul_precision("highest")


def load_worker():
    path = ROOT / "internal/training/workers/pytorch.py"
    tree = ast.parse(path.read_text(), filename=str(path))
    # Match the known CLI driver boundary, and fail closed if upstream changes it.
    driver = tree.body[-1]
    if not (isinstance(driver, ast.Try) and len(driver.body) == 1
            and isinstance(driver.body[0], ast.Expr)
            and isinstance(driver.body[0].value, ast.Call)
            and isinstance(driver.body[0].value.func, ast.Name)
            and driver.body[0].value.func.id == "run"):
        raise ValueError("worker CLI boundary changed; review the test loader")
    module = types.ModuleType("waldo_fixed_batch_worker")
    model_path = ROOT / "internal/pytorchruntime/model.py"
    exec(compile(model_path.read_text(), str(model_path), "exec"), module.__dict__)
    tree.body.pop()  # Omit only stdin CLI dispatch, not Trainer or its helpers.
    exec(compile(tree, str(path), "exec"), module.__dict__)
    return module


def prepare(args):
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    architecture = dict(vocabulary_size=259, hidden_size=8, intermediate_size=24,
                        layers=2, attention_heads=2, key_value_heads=1, dropout=0.0,
                        qk_normalization=False, initialization="normal", tie_embeddings=True,
                        parameter_dtype="float32", context_tokens=8,
                        tokenizer={"name": "byte", "revision": "builtin-byte-schema-1"})
    cases = {}
    for case in ("dense", "uneven"):
        rows = []
        for ordinal in range(args.steps * 8):
            tokens = [3 + ordinal % 256, 3 + ordinal // 256]
            tokens += [3 + (ordinal * 17 + position * 7) % 256 for position in range(7)]
            valid = 8 if case == "dense" else 1 + ordinal % 8
            rows.append({"tokens": tokens, "mask": [int(i < valid) for i in range(8)]})
        cases[case] = rows
    fixture = dict(schema=1, steps=args.steps, global_batch=8, sequence_length=8,
                   architecture=architecture, cases=cases, tolerances=TOLERANCES,
                   optimizer=dict(name="adamw", beta1=0.9, beta2=0.95, epsilon=1e-8, weight_decay=0.1),
                   learning_rate=0.0017, seed=7,
                   provenance="Synthetic integer fixtures; no external corpus or pretrained weights.")
    write_json(out / "fixture.json", fixture)
    load_runtime()
    model = oracle.load_waldo_model(ROOT / "internal/pytorchruntime/model.py").DecoderLM(architecture).cpu()
    oracle.deterministic_parameters(model)
    oracle.save_checkpoint_safetensors(out / "initial.safetensors", model.state_dict())
    write_json(out / "MANIFEST.json", {
        "schema": 1, "files": {name: {"bytes": (out / name).stat().st_size, "sha256": digest(out / name)}
                                for name in ("fixture.json", "initial.safetensors")},
        "generator_sources": {p: digest(ROOT / p) for p in SOURCE_PATHS},
        "torch": torch.__version__, "python": sys.version,
    })
    verify_fixture(out)
    print(f"Prepared and verified {out}")


def tensor_cpu(value):
    if hasattr(value, "full_tensor"):
        value = value.full_tensor()  # Collective; every rank calls in name order.
    return value.detach().cpu().clone()


def named_parameters(trainer):
    model = trainer.model.module if isinstance(trainer.model, torch.nn.parallel.DistributedDataParallel) else trainer.model
    return sorted(model.named_parameters())


def begin_frame(fixture, directory, accumulation, strategy):
    architecture = fixture["architecture"]
    return dict(run_id="fixed-batch-fp32", model_id="fixed-batch-fp32",
                architecture_sha256=hashlib.sha256(json.dumps(architecture, sort_keys=True).encode()).hexdigest(),
                architecture=architecture,
                tokenizer=dict(name="byte", revision="builtin-byte-schema-1", vocabulary_size=259,
                               pad_id=0, bos_id=1, eos_id=2),
                initialization={"path": str(Path(directory).resolve() / "initial.safetensors")},
                parameters=dict(batch_size=8, gradient_accumulation_steps=accumulation,
                                sequence_length=8, steps=fixture["steps"], seed=fixture["seed"],
                                learning_rate=fixture["learning_rate"], optimizer=fixture["optimizer"],
                                schedule=dict(name="warmup-stable-warmdown", warmup_steps=0,
                                              warmdown_steps=0, minimum_rate_ratio=1.0),
                                compute_precision="float32", compile=False, activation_checkpointing=False,
                                checkpoint_every=0, evaluate_every=0),
                parallelism={"strategy": strategy})


def worker(args):
    require_cuda_authorization(args.device)
    fixture = verify_fixture(args.fixture)
    load_runtime()
    rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    if world not in (1, 2, 4) or 8 % (world * args.accumulation):
        raise ValueError("invalid batch topology")
    if args.device == "cpu" and args.strategy != "data-parallel":
        raise ValueError("CPU placement adapter only supports DDP")
    if args.device == "cuda":
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    torch.distributed.init_process_group(
        backend="nccl" if args.device == "cuda" else "gloo",
        timeout=datetime.timedelta(seconds=120),
    )
    try:
        module = load_worker()
        module.IS_PRIMARY = rank == 0
        begin = begin_frame(fixture, args.fixture, args.accumulation, args.strategy)
        trainer = module.Trainer(begin, args.output, "artifacts", "torchtitan" if args.device == "cuda" else "cpu")
        if args.device == "cpu":
            # Production distributed placement is CUDA-only. Adapt placement
            # after CPU construction; preserve the real sampler/train_batch.
            trainer.distributed = True
            trainer.rank, trainer.world_size = rank, world
            trainer.batch_size = trainer.global_micro_batch_size // world
            trainer.model = torch.nn.parallel.DistributedDataParallel(
                trainer.model, broadcast_buffers=False, gradient_as_bucket_view=True, static_graph=True)
            trainer.compiled_model = trainer.model
        snapshots, observed_rows, local_logits = [], [], {}
        rows = fixture["cases"][args.case]
        lookup = {tuple(row["tokens"]): i for i, row in enumerate(rows)}
        original_train_batch = trainer.train_batch

        def record_batch(final=False):
            ids = [lookup[tuple(item[0])] for item in trainer.batch]
            observed_rows.extend(ids)
            trainer._qualification_batch_ids = ids
            return original_train_batch(final=final)

        trainer.train_batch = record_batch

        def record_forward(model, inputs, output):
            for row_id, logits in zip(trainer._qualification_batch_ids, output):
                local_logits[row_id] = tensor_cpu(logits)

        trainer.model.register_forward_hook(record_forward)

        def before_step(optimizer, positional, keywords):
            snapshots.append({"gradients": {n: tensor_cpu(p.grad) for n, p in named_parameters(trainer)},
                              "logits": dict(local_logits), "rows": list(observed_rows),
                              "loss": trainer.accumulated_loss_sum / trainer.accumulated_tokens,
                              "valid_tokens": trainer.accumulated_tokens})
            observed_rows.clear()
            local_logits.clear()

        def after_step(optimizer, positional, keywords):
            snapshot = snapshots[-1]
            snapshot["weights"] = {n: tensor_cpu(p) for n, p in named_parameters(trainer)}
            for key in ("exp_avg", "exp_avg_sq"):
                snapshot[key] = {n: tensor_cpu(optimizer.state[p][key]) for n, p in named_parameters(trainer)}
            snapshot["optimizer_steps"] = {n: int(optimizer.state[p]["step"].item()) for n, p in named_parameters(trainer)}

        trainer.optimizer.register_step_pre_hook(before_step)
        trainer.optimizer.register_step_post_hook(after_step)
        for row in rows:
            trainer.add_sequence(row["tokens"], row["mask"], ["fixture"] * 8)
        if trainer.step_number != fixture["steps"] or trainer.accumulation_number or trainer.batch:
            raise AssertionError("incomplete optimizer trajectory")
        output = Path(args.output)
        output.mkdir(parents=True, exist_ok=True)
        torch.save({"schema": 1, "case": args.case, "world_size": world, "rank": rank,
                    "accumulation": args.accumulation, "device": args.device, "strategy": args.strategy,
                    "snapshots": snapshots, "consumed_tokens": trainer.consumed_tokens,
                    "fixture_sha256": digest(Path(args.fixture) / "MANIFEST.json"),
                    "sources": {p: digest(ROOT / p) for p in SOURCE_PATHS},
                    "torch": str(torch.__version__), "cuda_runtime": torch.version.cuda,
                    "gpu_name": torch.cuda.get_device_name() if args.device == "cuda" else None,
                    "python": sys.version}, output / f"rank-{rank}.pt")
    finally:
        torch.distributed.destroy_process_group()


def reference(fixture, directory, case):
    initial = oracle.load_checkpoint_safetensors(Path(directory) / "initial.safetensors")
    # Oracle uses one whole global batch and explicit functional model/AdamW.
    parameters = {n: p.clone().requires_grad_(True) for n, p in initial.items()}
    moments = {n: torch.zeros_like(p) for n, p in parameters.items()}
    squares = {n: torch.zeros_like(p) for n, p in parameters.items()}
    result = []
    opt, lr = fixture["optimizer"], fixture["learning_rate"]
    for step in range(fixture["steps"]):
        rows = fixture["cases"][case][step * 8:(step + 1) * 8]
        tokens = torch.tensor([r["tokens"] for r in rows], dtype=torch.long)
        mask = torch.tensor([r["mask"] for r in rows], dtype=torch.float32)
        logits = oracle.reference_logits(tokens[:, :-1], parameters, fixture["architecture"])
        loss = oracle.masked_reference_loss(logits, tokens[:, 1:], mask)
        loss.backward()
        snap = {"loss": float(loss.detach()), "valid_tokens": int(mask.sum()),
                "logits": {step * 8 + i: tensor_cpu(value) for i, value in enumerate(logits)},
                "gradients": {n: tensor_cpu(p.grad) for n, p in parameters.items()}}
        with torch.no_grad():
            for n, p in parameters.items():
                g = p.grad
                moments[n] = opt["beta1"] * moments[n] + (1 - opt["beta1"]) * g
                squares[n] = opt["beta2"] * squares[n] + (1 - opt["beta2"]) * g.square()
                mean = moments[n] / (1 - opt["beta1"] ** (step + 1))
                variance = squares[n] / (1 - opt["beta2"] ** (step + 1))
                p.mul_(1 - lr * opt["weight_decay"])
                p.add_(-lr * mean / (variance.sqrt() + opt["epsilon"]))
                p.grad = None
        snap.update(weights={n: tensor_cpu(p) for n, p in parameters.items()},
                    exp_avg={n: tensor_cpu(p) for n, p in moments.items()},
                    exp_avg_sq={n: tensor_cpu(p) for n, p in squares.items()})
        result.append(snap)
    return result


def compare_tensor(name, actual, expected, tolerance, errors):
    if actual.shape != expected.shape or actual.dtype != expected.dtype:
        raise AssertionError(f"{name}: shape/dtype mismatch")
    if not torch.isfinite(actual).all() or not torch.isfinite(expected).all():
        raise AssertionError(f"{name}: nonfinite tensor")
    error = oracle.maximum_error(actual, expected)
    errors[name] = error
    if not torch.allclose(actual, expected, atol=tolerance[0], rtol=tolerance[1]):
        raise AssertionError(f"{name}: max absolute error {error}; atol/rtol={tolerance}")


def check_cell(fixture, args, case, world, accumulation, expected, baseline=None, write_report=True):
    cell = Path(args.output) / f"{case}-w{world}-a{accumulation}"
    errors, rank_zero = {}, None
    for rank in range(world):
        data = torch.load(cell / f"rank-{rank}.pt", map_location="cpu", weights_only=True)
        identity = dict(case=case, world_size=world, rank=rank, accumulation=accumulation,
                        device=args.device, strategy=args.strategy)
        if any(data.get(k) != v for k, v in identity.items()):
            raise AssertionError("rank result identity mismatch")
        if data["fixture_sha256"] != digest(Path(args.fixture) / "MANIFEST.json"):
            raise AssertionError("rank fixture identity mismatch")
        if data["sources"] != {p: digest(ROOT / p) for p in SOURCE_PATHS}:
            raise AssertionError("source changed during execution")
        if len(data["snapshots"]) != fixture["steps"]:
            raise AssertionError("missing optimizer steps")
        if data["consumed_tokens"] != sum(s["valid_tokens"] for s in expected):
            raise AssertionError("global token accounting mismatch")
        for step, (actual, wanted) in enumerate(zip(data["snapshots"], expected)):
            rows = list(range(step * 8 + rank, (step + 1) * 8, world))
            if actual["rows"] != rows or sorted(actual["logits"]) != rows:
                raise AssertionError(f"rank {rank} step {step}: missing, reordered, or duplicated samples")
            if actual["valid_tokens"] != wanted["valid_tokens"]:
                raise AssertionError("optimizer valid-token count mismatch")
            if set(actual["optimizer_steps"]) != set(wanted["weights"]) or any(
                    value != step + 1 for value in actual["optimizer_steps"].values()):
                raise AssertionError("optimizer step-state mismatch")
            comparisons = [("oracle", wanted)]
            if baseline is not None:
                comparisons.append(("singleton", baseline[step]))
            for label, reference_step in comparisons:
                prefix = f"{label}/rank{rank}/step{step}"
                compare_tensor(prefix + "/loss", torch.tensor(actual["loss"]), torch.tensor(reference_step["loss"]), TOLERANCES["loss"], errors)
                for row in rows:
                    compare_tensor(f"{prefix}/logits/{row}", actual["logits"][row], reference_step["logits"][row], TOLERANCES["logits"], errors)
                for key in ("gradients", "weights", "exp_avg", "exp_avg_sq"):
                    if set(actual[key]) != set(reference_step[key]):
                        raise AssertionError(f"{key}: parameter coverage mismatch")
                    for name in sorted(reference_step[key]):
                        compare_tensor(f"{prefix}/{key}/{name}", actual[key][name], reference_step[key][name], TOLERANCES[key], errors)
        if rank == 0:
            rank_zero = data["snapshots"]
    if write_report:
        write_json(cell / "comparisons.json", {"status": "pass", "checks": len(errors), "maximum_absolute_errors": errors})
    return rank_zero, len(errors)


def launch(command, log, environment, timeout):
    started = time.monotonic()
    with Path(log).open("w") as stream:
        child = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT,
                                 env=environment, start_new_session=True)
        try:
            code = child.wait(timeout=timeout)
        except BaseException:
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
            raise
    return code, time.monotonic() - started


def run_matrix(args):
    require_cuda_authorization(args.device)
    if args.device == "cpu" and args.strategy != "data-parallel":
        raise ValueError("CPU placement adapter only supports DDP")
    cells = matrix(args.world_sizes)
    if 1 not in args.world_sizes or args.world_sizes != sorted(args.world_sizes):
        raise ValueError("matrix must include world size 1 first, followed by ascending sizes")
    fixture = verify_fixture(args.fixture)
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    summary = dict(schema=1, status="running", device=args.device, strategy=args.strategy,
                   scope="production-worker fixed-batch FP32; CPU uses Gloo placement adapter",
                   cuda_qualified=False, world_sizes=args.world_sizes,
                   tolerances=TOLERANCES, fixture_sha256=digest(Path(args.fixture) / "MANIFEST.json"),
                   sources={p: digest(ROOT / p) for p in SOURCE_PATHS},
                   started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                   cells=[], gpu_seconds=0.0)
    write_json(out / "summary.json", summary)
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", CUBLAS_WORKSPACE_CONFIG=":4096:8")
    if args.device == "cpu":
        env["CUDA_VISIBLE_DEVICES"] = ""
    try:
        load_runtime()
        if args.device == "cuda":
            if not os.environ.get("CUDA_VISIBLE_DEVICES"):
                raise ValueError("set CUDA_VISIBLE_DEVICES to the explicitly allocated GPUs")
            if torch.cuda.device_count() != max(args.world_sizes):
                raise ValueError("visible GPU count must exactly match the largest requested world size")
            summary["gpu_names"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
            summary["nccl"] = list(torch.cuda.nccl.version())
        summary["torch"] = str(torch.__version__)
        summary["cuda_runtime"] = torch.version.cuda
        summary["python"] = sys.version
        references = {case: reference(fixture, args.fixture, case) for case in fixture["cases"]}
        torch.save(references, out / "oracle.pt")
        baselines = {}
        for case, world, accumulation in cells:
            cell = out / f"{case}-w{world}-a{accumulation}"
            cell.mkdir()
            command = [sys.executable, "-m", "torch.distributed.run", "--standalone", "--local-addr=127.0.0.1",
                       "--nnodes=1", f"--nproc-per-node={world}", str(Path(__file__).resolve()),
                       "worker", "--fixture", str(Path(args.fixture).resolve()), "--output", str(cell),
                       "--device", args.device, "--strategy", args.strategy, "--case", case,
                       "--accumulation", str(accumulation)]
            timeout = args.timeout_seconds
            if args.device == "cuda":
                timeout = min(timeout, int((args.gpu_hour_budget * 3600 - summary["gpu_seconds"]) / world))
                if timeout < 1:
                    raise RuntimeError("aggregate GPU budget exhausted")
            receipt = dict(case=case, world_size=world, accumulation=accumulation,
                           command=command, timeout_seconds=timeout, status="running")
            summary["cells"].append(receipt)
            write_json(out / "summary.json", summary)
            cell_started = time.monotonic()
            try:
                code, elapsed = launch(command, cell / "worker.log", env, timeout)
                receipt.update(exit_code=code, elapsed_seconds=elapsed)
                if code:
                    raise RuntimeError(f"{cell.name}: worker exit {code}; see worker.log")
                snapshots, checks = check_cell(fixture, args, case, world, accumulation, references[case], baselines.get(case))
                if world == 1 and accumulation == 1:
                    baselines[case] = snapshots
                receipt.update(status="pass", tensor_checks=checks)
            except BaseException as error:
                receipt.update(status="fail", error=f"{type(error).__name__}: {error}")
                raise
            finally:
                receipt["elapsed_seconds"] = time.monotonic() - cell_started
                if args.device == "cuda":
                    summary["gpu_seconds"] += receipt["elapsed_seconds"] * world
        summary["status"] = "pass"
        summary["cuda_qualified"] = args.device == "cuda" and args.world_sizes == [1, 2, 4]
    except BaseException as error:
        summary.update(status="fail", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        summary["finished_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        write_json(out / "summary.json", summary)
        files = sorted(p for p in out.rglob("*") if p.is_file() and p.name != "SHA256SUMS")
        (out / "SHA256SUMS").write_text("".join(f"{digest(p)}  {p.relative_to(out)}\n" for p in files))
    print(json.dumps({"status": summary["status"], "cuda_qualified": summary["cuda_qualified"], "output": str(out)}))


def verify_inventory(directory):
    root = Path(directory).resolve()
    recorded = set()
    for line in (root / "SHA256SUMS").read_text().splitlines():
        expected, name = line.split("  ", 1)
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts or name in recorded:
            raise ValueError("invalid or duplicated evidence path")
        path = root / relative
        if path.is_symlink() or not path.resolve().is_relative_to(root) or digest(path) != expected:
            raise ValueError(f"evidence integrity failure: {name}")
        recorded.add(name)
    actual = {str(p.relative_to(root)) for p in root.rglob("*") if p.is_file() and p.name != "SHA256SUMS"}
    if not recorded or recorded != actual:
        raise ValueError("evidence inventory is incomplete")


def verify_results(args):
    verify_inventory(args.output)
    fixture = verify_fixture(args.fixture)
    summary = json.loads((Path(args.output) / "summary.json").read_text())
    if summary["status"] != "pass" or summary["world_sizes"] != [1, 2, 4]:
        raise ValueError("only a complete passing 1/2/4 matrix qualifies")
    args.device, args.strategy = summary["device"], summary["strategy"]
    if summary["cuda_qualified"] != (args.device == "cuda"):
        raise ValueError("false device qualification claim")
    expected_cells = matrix([1, 2, 4])
    observed = [(c["case"], c["world_size"], c["accumulation"]) for c in summary["cells"]]
    if observed != expected_cells or any(c["status"] != "pass" or c.get("exit_code") != 0 for c in summary["cells"]):
        raise ValueError("missing, failed, or duplicated cell")
    # Verification always evaluates the oracle on CPU, even for CUDA receipts.
    load_runtime()
    baselines = {}
    references = {case: reference(fixture, args.fixture, case) for case in fixture["cases"]}
    for case, world, accumulation in expected_cells:
        snapshots, _ = check_cell(fixture, args, case, world, accumulation, references[case],
                                  baselines.get(case), write_report=False)
        if world == 1 and accumulation == 1:
            baselines[case] = snapshots
    print(json.dumps({"verified": True, "device": args.device, "cuda_qualified": summary["cuda_qualified"]}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare", help="stage synthetic data and initial CPU weights")
    prep.add_argument("--output", required=True)
    prep.add_argument("--steps", type=int, default=8, choices=range(1, 33))
    verify = commands.add_parser("verify", help="verify staged fixture without importing torch")
    verify.add_argument("--fixture", required=True)
    results = commands.add_parser("verify-results", help="read-only CPU verification of a complete result bundle")
    results.add_argument("--fixture", required=True)
    results.add_argument("--output", required=True)
    for name in ("run", "worker"):
        sub = commands.add_parser(name)
        sub.add_argument("--fixture", required=True)
        sub.add_argument("--output", required=True)
        sub.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
        sub.add_argument("--strategy", choices=("data-parallel", "fully-sharded-data-parallel"), default="data-parallel")
        if name == "run":
            sub.add_argument("--world-sizes", nargs="+", type=int, default=[1, 2, 4])
            sub.add_argument("--timeout-seconds", type=int, default=600)
            sub.add_argument("--gpu-hour-budget", type=float, default=2.0)
        else:
            sub.add_argument("--case", choices=("dense", "uneven"), required=True)
            sub.add_argument("--accumulation", type=int, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args)
    elif args.command == "verify":
        verify_fixture(args.fixture)
        print("Fixture verified; no torch import or device access")
    elif args.command == "worker":
        worker(args)
    elif args.command == "verify-results":
        verify_results(args)
    else:
        if args.timeout_seconds <= 0 or not 0 < args.gpu_hour_budget <= 8:
            parser.error("timeout must be positive and GPU budget must be in (0, 8] hours")
        run_matrix(args)


if __name__ == "__main__":
    main()
