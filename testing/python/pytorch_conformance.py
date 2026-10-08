# Copyright (c) 2026 OpenWALDO Project contributors
# Copyright (c) 2026 CtrlIQ, Inc.
# Copyright (c) 2026 Gregory M. Kurtzer
# SPDX-License-Identifier: Apache-2.0

"""Known-answer numerical conformance for WALDO's shared PyTorch model.

The oracle below is intentionally functional and independent of WALDO's model
classes. It uses explicit attention, RoPE, RMS normalization, SwiGLU, loss, and
AdamW equations. WALDO continues to use its production implementation from
internal/pytorchruntime/model.py, including scaled_dot_product_attention.
"""

import argparse
import importlib.util
import io
import json
import math
import pathlib
import sys

import torch
from torch.nn import functional


def load_waldo_model(path):
    spec = importlib.util.spec_from_file_location("waldo_shared_pytorch_model", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load WALDO model source {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def rotate_half(value):
    midpoint = value.shape[-1] // 2
    return torch.cat((-value[..., midpoint:], value[..., :midpoint]), dim=-1)


def reference_rms_norm(value, weight, epsilon=1e-5):
    promoted = value.float()
    scale = torch.rsqrt(promoted.pow(2).mean(dim=-1, keepdim=True) + epsilon)
    return (promoted * scale).to(value.dtype) * weight


def reference_rope(value):
    length = value.shape[2]
    head_dimension = value.shape[-1]
    positions = torch.arange(length, device=value.device, dtype=torch.float32)
    dimensions = torch.arange(0, head_dimension, 2, device=value.device, dtype=torch.float32)
    frequencies = torch.pow(torch.tensor(10000.0, device=value.device, dtype=torch.float32), -dimensions / head_dimension)
    angles = torch.outer(positions, frequencies)
    angles = torch.cat((angles, angles), dim=-1).to(value.dtype)[None, None, :, :]
    return value * torch.cos(angles) + rotate_half(value) * torch.sin(angles)


def reference_attention(value, parameters, prefix, architecture):
    batch, length, hidden = value.shape
    heads = architecture["attention_heads"]
    key_value_heads = architecture["key_value_heads"]
    head_dimension = hidden // heads
    query = functional.linear(value, parameters[prefix + "q_proj.weight"])
    key = functional.linear(value, parameters[prefix + "k_proj.weight"])
    val = functional.linear(value, parameters[prefix + "v_proj.weight"])
    query = query.reshape(batch, length, heads, head_dimension).transpose(1, 2)
    key = key.reshape(batch, length, key_value_heads, head_dimension).transpose(1, 2)
    val = val.reshape(batch, length, key_value_heads, head_dimension).transpose(1, 2)
    query = reference_rope(query)
    key = reference_rope(key)
    if architecture.get("qk_normalization", False):
        query = query * torch.rsqrt(query.float().pow(2).mean(dim=-1, keepdim=True) + 1e-6).to(query.dtype)
        key = key * torch.rsqrt(key.float().pow(2).mean(dim=-1, keepdim=True) + 1e-6).to(key.dtype)
    if heads != key_value_heads:
        repeats = heads // key_value_heads
        key = key.repeat_interleave(repeats, dim=1)
        val = val.repeat_interleave(repeats, dim=1)

    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(head_dimension)
    future = torch.triu(torch.ones((length, length), device=value.device, dtype=torch.bool), diagonal=1)
    scores = scores.masked_fill(future, -torch.inf)
    probabilities = torch.softmax(scores, dim=-1)
    attended = torch.matmul(probabilities, val)
    attended = attended.transpose(1, 2).reshape(batch, length, hidden)
    return functional.linear(attended, parameters[prefix + "o_proj.weight"])


def reference_logits(tokens, parameters, architecture):
    value = functional.embedding(tokens, parameters["embedding.weight"])
    for layer in range(architecture["layers"]):
        prefix = f"layers.{layer}."
        normalized = reference_rms_norm(value, parameters[prefix + "attention_norm.weight"])
        value = value + reference_attention(normalized, parameters, prefix + "attention.", architecture)
        normalized = reference_rms_norm(value, parameters[prefix + "ffn_norm.weight"])
        gate = functional.linear(normalized, parameters[prefix + "feed_forward.gate.weight"])
        up = functional.linear(normalized, parameters[prefix + "feed_forward.up.weight"])
        value = value + functional.linear(functional.silu(gate) * up, parameters[prefix + "feed_forward.down.weight"])
    value = reference_rms_norm(value, parameters["norm.weight"])
    output_weight = parameters["embedding.weight"] if architecture["tie_embeddings"] else parameters["output.weight"]
    return functional.linear(value, output_weight)


def masked_reference_loss(logits, targets, mask):
    token_losses = -torch.log_softmax(logits, dim=-1).gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    return (token_losses * mask).sum() / mask.sum()


def masked_waldo_loss(logits, targets, mask):
    token_losses = functional.cross_entropy(
        logits.float().reshape(-1, logits.shape[-1]),
        targets.reshape(-1),
        reduction="none",
    ).reshape_as(mask)
    mask = mask.float()
    return (token_losses * mask).sum() / mask.sum()


def deterministic_parameters(model):
    with torch.no_grad():
        for index, (name, parameter) in enumerate(model.named_parameters()):
            values = torch.arange(parameter.numel(), device=parameter.device, dtype=parameter.dtype).reshape(parameter.shape)
            if name.endswith("norm.weight") or name == "norm.weight":
                parameter.copy_(1.0 + 0.01 * torch.cos(values + index))
            else:
                parameter.copy_(0.03 * torch.sin(values * 0.17 + index))


def maximum_error(left, right):
    return float((left.detach().double().cpu() - right.detach().double().cpu()).abs().max().item())


def assert_close(name, left, right, absolute_tolerance, relative_tolerance, measurements):
    error = maximum_error(left, right)
    measurements[name] = error
    if not torch.allclose(left, right, atol=absolute_tolerance, rtol=relative_tolerance):
        raise AssertionError(f"{name} differs: maximum absolute error {error:.12g}")


def run(model_source, device_name):
    device = torch.device(device_name)
    dtype = torch.float64 if device.type == "cpu" else torch.float32
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA conformance requested but torch.cuda.is_available() is false")
    if device.type == "cpu":
        absolute_tolerance, relative_tolerance = 2e-7, 1e-6
        loss_tolerance, gradient_tolerance, update_tolerance = 2e-6, 3e-6, 1e-10
    else:
        absolute_tolerance, relative_tolerance = 8e-5, 2e-4
        loss_tolerance, gradient_tolerance, update_tolerance = 2e-5, 4e-4, 2e-5

    architecture = {
        "vocabulary_size": 19,
        "hidden_size": 8,
        "intermediate_size": 24,
        "layers": 2,
        "attention_heads": 2,
        "key_value_heads": 1,
        "dropout": 0.0,
        "qk_normalization": False,
        "initialization": "normal",
        "tie_embeddings": True,
    }
    module = load_waldo_model(model_source)
    model = module.DecoderLM(architecture).to(device=device, dtype=dtype)
    deterministic_parameters(model)
    model.train()

    token_rows = [
        [1, 4, 7, 3, 9, 2, 5],
        [1, 8, 6, 4, 3, 11, 2],
    ]
    tokens = torch.tensor(token_rows, device=device, dtype=torch.long)
    inputs, targets = tokens[:, :-1], tokens[:, 1:]
    mask = torch.tensor(
        [[1, 1, 1, 1, 1, 1], [1, 1, 1, 0, 0, 0]],
        device=device,
        dtype=dtype,
    )

    reference_parameters = {
        name: parameter.detach().clone().requires_grad_(True)
        for name, parameter in model.named_parameters()
    }
    waldo_logits = model(inputs)
    oracle_logits = reference_logits(inputs, reference_parameters, architecture)
    measurements = {}
    assert_close("logits", waldo_logits, oracle_logits, absolute_tolerance, relative_tolerance, measurements)

    waldo_loss = masked_waldo_loss(waldo_logits, targets, mask)
    oracle_loss = masked_reference_loss(oracle_logits.float(), targets, mask.float())
    assert_close("masked_loss", waldo_loss, oracle_loss, loss_tolerance, relative_tolerance, measurements)

    waldo_loss.backward()
    oracle_loss.backward()
    for name, parameter in model.named_parameters():
        assert_close(
            "gradient/" + name,
            parameter.grad,
            reference_parameters[name].grad,
            gradient_tolerance,
            relative_tolerance,
            measurements,
        )

    learning_rate = 0.0017
    beta1, beta2 = 0.9, 0.95
    epsilon = 1e-8
    weight_decay = 0.1
    before = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
    optimizer_gradients = {name: parameter.grad.detach().clone() for name, parameter in model.named_parameters()}
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        betas=(beta1, beta2),
        eps=epsilon,
        weight_decay=weight_decay,
    )
    optimizer.step()

    for name, parameter in model.named_parameters():
        gradient = optimizer_gradients[name]
        first_moment = (1.0 - beta1) * gradient
        second_moment = (1.0 - beta2) * gradient.square()
        corrected_first = first_moment / (1.0 - beta1)
        corrected_second = second_moment / (1.0 - beta2)
        expected = before[name] * (1.0 - learning_rate * weight_decay)
        expected = expected - learning_rate * corrected_first / (torch.sqrt(corrected_second) + epsilon)
        assert_close("adamw_parameter/" + name, parameter, expected, update_tolerance, relative_tolerance, measurements)
        state = optimizer.state[parameter]
        assert_close("adamw_first_moment/" + name, state["exp_avg"], first_moment, update_tolerance, relative_tolerance, measurements)
        assert_close("adamw_second_moment/" + name, state["exp_avg_sq"], second_moment, update_tolerance, relative_tolerance, measurements)
        step = state["step"].item() if hasattr(state["step"], "item") else state["step"]
        if float(step) != 1.0:
            raise AssertionError(f"AdamW step for {name} is {state['step']}, want 1")

    model.eval()
    with torch.no_grad():
        post_update_logits = model(inputs)
    payload = io.BytesIO()
    torch.save(model.state_dict(), payload)
    payload.seek(0)
    try:
        reloaded_state = torch.load(payload, map_location=device, weights_only=True)
    except TypeError:
        reloaded_state = torch.load(payload, map_location=device)
    reloaded = module.DecoderLM(architecture).to(device=device, dtype=dtype)
    missing, unexpected = reloaded.load_state_dict(reloaded_state, strict=True)
    if missing or unexpected:
        raise AssertionError(f"reload mismatch: missing={missing}, unexpected={unexpected}")
    reloaded.eval()
    with torch.no_grad():
        reloaded_logits = reloaded(inputs)
    assert_close("save_reload_logits", post_update_logits, reloaded_logits, 0.0, 0.0, measurements)

    updated_reference = {name: parameter.detach().clone().requires_grad_(False) for name, parameter in model.named_parameters()}
    oracle_post_update = reference_logits(inputs, updated_reference, architecture)
    assert_close("post_update_logits", post_update_logits, oracle_post_update, absolute_tolerance, relative_tolerance, measurements)
    if not torch.equal(post_update_logits[:, -1].argmax(dim=-1), oracle_post_update[:, -1].argmax(dim=-1)):
        raise AssertionError("next-token argmax differs after AdamW update")

    return {
        "status": "pass",
        "device": str(device),
        "dtype": str(dtype).removeprefix("torch."),
        "torch": torch.__version__,
        "architecture": architecture,
        "checks": len(measurements) + 1,
        "maximum_errors": {
            "logits": measurements["logits"],
            "masked_loss": measurements["masked_loss"],
            "gradients": max(value for name, value in measurements.items() if name.startswith("gradient/")),
            "adamw_parameters": max(value for name, value in measurements.items() if name.startswith("adamw_parameter/")),
            "adamw_state": max(value for name, value in measurements.items() if name.startswith("adamw_") and not name.startswith("adamw_parameter/")),
            "save_reload_logits": measurements["save_reload_logits"],
            "post_update_logits": measurements["post_update_logits"],
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-source", required=True, type=pathlib.Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    arguments = parser.parse_args()
    try:
        result = run(arguments.model_source, arguments.device)
    except Exception as error:
        print(json.dumps({"status": "fail", "device": arguments.device, "error": str(error)}, sort_keys=True), file=sys.stderr)
        raise
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
