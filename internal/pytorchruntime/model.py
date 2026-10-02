# Copyright (c) 2026 OpenWALDO Project contributors
# Copyright (c) 2026 CtrlIQ, Inc.
# Copyright (c) 2026 Gregory M. Kurtzer
# SPDX-License-Identifier: Apache-2.0

import math

import torch
from torch import nn
from torch.nn import functional
from torch.utils.checkpoint import checkpoint


class RMSNorm(nn.Module):
    def __init__(self, hidden, epsilon=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden))
        self.epsilon = epsilon

    def forward(self, value):
        normalized = value.float() * torch.rsqrt(value.float().pow(2).mean(-1, keepdim=True) + self.epsilon)
        return normalized.to(value.dtype) * self.weight


def rotate_half(value):
    first = value[..., : value.shape[-1] // 2]
    second = value[..., value.shape[-1] // 2 :]
    return torch.cat((-second, first), dim=-1)


class Attention(nn.Module):
    def __init__(self, hidden, heads, kv_heads, qk_normalization=False):
        super().__init__()
        self.heads = heads
        self.kv_heads = kv_heads
        self.head_dim = hidden // heads
        self.qk_normalization = qk_normalization
        kv_width = self.head_dim * kv_heads
        self.q_proj = nn.Linear(hidden, hidden, bias=False)
        self.k_proj = nn.Linear(hidden, kv_width, bias=False)
        self.v_proj = nn.Linear(hidden, kv_width, bias=False)
        self.o_proj = nn.Linear(hidden, hidden, bias=False)

    def rope(self, value):
        length = value.shape[2]
        positions = torch.arange(length, device=value.device, dtype=torch.float32)
        frequencies = 1.0 / (10000.0 ** (torch.arange(0, self.head_dim, 2, device=value.device, dtype=torch.float32) / self.head_dim))
        angles = torch.outer(positions, frequencies)
        angles = torch.cat((angles, angles), dim=-1).to(value.dtype)[None, None, :, :]
        return value * angles.cos() + rotate_half(value) * angles.sin()

    def forward(self, value):
        batch, length, _ = value.shape
        query = self.q_proj(value).reshape(batch, length, self.heads, self.head_dim).transpose(1, 2)
        key = self.k_proj(value).reshape(batch, length, self.kv_heads, self.head_dim).transpose(1, 2)
        val = self.v_proj(value).reshape(batch, length, self.kv_heads, self.head_dim).transpose(1, 2)
        query = self.rope(query)
        key = self.rope(key)
        if self.qk_normalization:
            query = query * torch.rsqrt(query.float().pow(2).mean(-1, keepdim=True) + 1e-6).to(query.dtype)
            key = key * torch.rsqrt(key.float().pow(2).mean(-1, keepdim=True) + 1e-6).to(key.dtype)
        if self.heads != self.kv_heads:
            repeats = self.heads // self.kv_heads
            key = key.repeat_interleave(repeats, dim=1)
            val = val.repeat_interleave(repeats, dim=1)
        attended = functional.scaled_dot_product_attention(query, key, val, is_causal=True)
        attended = attended.transpose(1, 2).reshape(batch, length, -1)
        return self.o_proj(attended)


class FeedForward(nn.Module):
    def __init__(self, hidden, intermediate):
        super().__init__()
        self.gate = nn.Linear(hidden, intermediate, bias=False)
        self.up = nn.Linear(hidden, intermediate, bias=False)
        self.down = nn.Linear(intermediate, hidden, bias=False)

    def forward(self, value):
        return self.down(functional.silu(self.gate(value)) * self.up(value))


class DecoderBlock(nn.Module):
    def __init__(self, hidden, intermediate, heads, kv_heads, dropout, qk_normalization):
        super().__init__()
        self.attention_norm = RMSNorm(hidden)
        self.attention = Attention(hidden, heads, kv_heads, qk_normalization)
        self.ffn_norm = RMSNorm(hidden)
        self.feed_forward = FeedForward(hidden, intermediate)
        self.residual_dropout = nn.Dropout(dropout)

    def forward(self, value):
        value = value + self.residual_dropout(self.attention(self.attention_norm(value)))
        return value + self.residual_dropout(self.feed_forward(self.ffn_norm(value)))


class DecoderLM(nn.Module):
    def __init__(self, architecture):
        super().__init__()
        self.architecture = architecture
        vocabulary = architecture["vocabulary_size"]
        hidden = architecture["hidden_size"]
        self.tie_embeddings = architecture["tie_embeddings"]
        self.activation_checkpointing = False
        self.embedding = nn.Embedding(vocabulary, hidden)
        self.layers = nn.ModuleList([
            DecoderBlock(
                hidden,
                architecture["intermediate_size"],
                architecture["attention_heads"],
                architecture["key_value_heads"],
                architecture.get("dropout", 0.0),
                architecture.get("qk_normalization", False),
            )
            for _ in range(architecture["layers"])
        ])
        self.norm = RMSNorm(hidden)
        if not self.tie_embeddings:
            self.output = nn.Linear(hidden, vocabulary, bias=False)

    def initialize(self):
        for module in self.modules():
            if isinstance(module, (nn.Embedding, nn.Linear)):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, RMSNorm):
                nn.init.ones_(module.weight)
        if self.architecture.get("initialization", "normal") == "depth-scaled":
            residual_std = 0.02 / math.sqrt(2 * len(self.layers))
            for layer in self.layers:
                nn.init.normal_(layer.attention.o_proj.weight, mean=0.0, std=residual_std)
                nn.init.normal_(layer.feed_forward.down.weight, mean=0.0, std=residual_std)

    def forward(self, tokens):
        value = self.embedding(tokens)
        for layer in self.layers:
            if self.activation_checkpointing and self.training:
                value = checkpoint(layer, value, use_reentrant=False)
            else:
                value = layer(value)
        value = self.norm(value)
        if self.tie_embeddings:
            return functional.linear(value, self.embedding.weight)
        return self.output(value)


# Worker scripts may retain older local definitions while schema-1 artifacts
# migrate. These aliases make the shared implementation authoritative.
WALDO_SHARED_RMS_NORM = RMSNorm
WALDO_SHARED_ATTENTION = Attention
WALDO_SHARED_FEED_FORWARD = FeedForward
WALDO_SHARED_DECODER_BLOCK = DecoderBlock
WALDO_SHARED_DECODER_LM = DecoderLM
