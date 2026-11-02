# Copyright (c) 2026 OpenWALDO Project contributors
# SPDX-License-Identifier: Apache-2.0
"""The supported recipe surface, defined here and emitted by probe."""
from copy import deepcopy

ARCHITECTURE_DEFAULTS = dict(
    vocab_size=151936, hidden_size=1024, intermediate_size=3072,
    num_hidden_layers=12, num_attention_heads=16, num_key_value_heads=8,
    head_dim=64, max_position_embeddings=32768, rms_norm_eps=1e-6,
    rope_theta=1000000.0, attention_bias=False, attention_dropout=0.0,
    tie_word_embeddings=True, hidden_act="silu", initializer_range=0.02,
    use_cache=False, bos_token_id=151643, eos_token_id=151645,
    pad_token_id=151643,
)
PARAMETER_DEFAULTS = dict(
    learning_rate=0.0003, lr_scheduler_type="cosine", warmup_steps=0,
    weight_decay=0.1, adam_beta1=0.9, adam_beta2=0.999, adam_epsilon=1e-8,
    max_grad_norm=1.0, optim="adamw_torch", bf16=False, fp16=False,
    tf32=False, label_smoothing_factor=0.0, init_seed=42,
    per_device_train_batch_size=1, gradient_accumulation_steps=1,
    gradient_checkpointing=False, torch_compile=False,
    attn_implementation="eager", logging_steps=1, report_to=[],
)
EXECUTION = {
    "per_device_train_batch_size", "gradient_accumulation_steps",
    "gradient_checkpointing", "torch_compile", "attn_implementation",
    "logging_steps", "report_to",
}


def schema(defaults, execution=()):
    properties = {}
    for key, value in defaults.items():
        typ = {bool: "boolean", int: "integer", float: "number", str: "string", list: "array"}[type(value)]
        entry = {"type": typ, "default": deepcopy(value)}
        if key in execution:
            entry["x-waldo-role"] = "execution"
        if typ in ("integer", "number"):
            entry["minimum"] = 0
        properties[key] = entry
    return {"$schema": "https://json-schema.org/draft/2020-12/schema", "type": "object",
            "additionalProperties": False, "properties": properties}


ARCHITECTURE = schema(ARCHITECTURE_DEFAULTS)
for key in ("vocab_size", "hidden_size", "intermediate_size", "num_hidden_layers",
            "num_attention_heads", "num_key_value_heads", "head_dim", "max_position_embeddings"):
    ARCHITECTURE["properties"][key]["minimum"] = 1
ARCHITECTURE["properties"]["hidden_act"]["enum"] = ["silu"]
PARAMETERS = schema(PARAMETER_DEFAULTS, EXECUTION)
for key in ("per_device_train_batch_size", "gradient_accumulation_steps", "logging_steps"):
    PARAMETERS["properties"][key]["minimum"] = 1
for key in ("adam_beta1", "adam_beta2", "label_smoothing_factor"):
    PARAMETERS["properties"][key]["exclusiveMaximum"] = 1
PARAMETERS["properties"]["optim"]["enum"] = ["adamw_torch"]
PARAMETERS["properties"]["lr_scheduler_type"]["enum"] = ["linear", "cosine", "constant"]
PARAMETERS["properties"]["attn_implementation"]["enum"] = ["eager", "sdpa"]
PARAMETERS["properties"]["report_to"]["maxItems"] = 0
# Compilation is reserved until separately numerically qualified.
PARAMETERS["properties"]["torch_compile"]["const"] = False
TOKENIZER = schema({})


def validate(values, document):
    """Validate our deliberately flat schema for standalone command use too."""
    if not isinstance(values, dict):
        raise ValueError("configuration must be an object")
    for key, value in values.items():
        rule = document["properties"].get(key)
        if rule is None:
            raise ValueError(f"unknown configuration property: {key}")
        types = {"boolean": (bool,), "integer": (int,), "number": (int, float),
                 "string": (str,), "array": (list,)}
        if type(value) not in types[rule["type"]]:
            raise ValueError(f"invalid type for {key}")
        if "minimum" in rule and value < rule["minimum"]:
            raise ValueError(f"{key} below minimum")
        if "exclusiveMaximum" in rule and value >= rule["exclusiveMaximum"]:
            raise ValueError(f"{key} above maximum")
        if "enum" in rule and value not in rule["enum"]:
            raise ValueError(f"unsupported {key}")
        if "const" in rule and value != rule["const"]:
            raise ValueError(f"unsupported {key}")
        if "maxItems" in rule and len(value) > rule["maxItems"]:
            raise ValueError(f"unsupported {key}")


def architecture(values):
    validate(values, ARCHITECTURE)
    result = ARCHITECTURE_DEFAULTS | values
    if result["num_attention_heads"] % result["num_key_value_heads"]:
        raise ValueError("query heads must be divisible by key/value heads")
    if result["head_dim"] % 2:
        raise ValueError("head_dim must be even for RoPE")
    return result
