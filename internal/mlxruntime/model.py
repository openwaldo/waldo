# Copyright (c) 2026 OpenWALDO Project contributors
# Copyright (c) 2026 CtrlIQ, Inc.
# Copyright (c) 2026 Gregory M. Kurtzer
# SPDX-License-Identifier: Apache-2.0

import mlx.core as mx
import mlx.nn as nn


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
        self.rope = nn.RoPE(self.head_dim, traditional=False, base=10000)

    def project(self, value, offset=0):
        batch, length, _ = value.shape
        query = self.q_proj(value).reshape(batch, length, self.heads, self.head_dim).transpose(0, 2, 1, 3)
        key = self.k_proj(value).reshape(batch, length, self.kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        val = self.v_proj(value).reshape(batch, length, self.kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        query, key = self.rope(query, offset=offset), self.rope(key, offset=offset)
        if self.qk_normalization:
            query = query * mx.rsqrt(mx.mean(query * query, axis=-1, keepdims=True) + 1e-6)
            key = key * mx.rsqrt(mx.mean(key * key, axis=-1, keepdims=True) + 1e-6)
        return query, key, val

    def __call__(self, value):
        query, key, val = self.project(value)
        attended = mx.fast.scaled_dot_product_attention(
            query, key, val, scale=self.head_dim ** -0.5, mask="causal"
        )
        attended = attended.transpose(0, 2, 1, 3).reshape(value.shape[0], value.shape[1], -1)
        return self.o_proj(attended)

    def generate(self, value, cache=None):
        offset = 0 if cache is None else cache[0].shape[2]
        query, key, val = self.project(value, offset)
        if cache is not None:
            key = mx.concatenate([cache[0], key], axis=2)
            val = mx.concatenate([cache[1], val], axis=2)
        if value.shape[1] > 1:
            attended = mx.fast.scaled_dot_product_attention(
                query, key, val, scale=self.head_dim ** -0.5, mask="causal"
            )
        else:
            attended = mx.fast.scaled_dot_product_attention(
                query, key, val, scale=self.head_dim ** -0.5
            )
        attended = attended.transpose(0, 2, 1, 3).reshape(value.shape[0], value.shape[1], -1)
        return self.o_proj(attended), (key, val)


class FeedForward(nn.Module):
    def __init__(self, hidden, intermediate):
        super().__init__()
        self.gate = nn.Linear(hidden, intermediate, bias=False)
        self.up = nn.Linear(hidden, intermediate, bias=False)
        self.down = nn.Linear(intermediate, hidden, bias=False)

    def __call__(self, value):
        return self.down(nn.silu(self.gate(value)) * self.up(value))


class DecoderBlock(nn.Module):
    def __init__(self, hidden, intermediate, heads, kv_heads, dropout=0.0, qk_normalization=False):
        super().__init__()
        self.attention_norm = nn.RMSNorm(hidden, eps=1e-5)
        self.attention = Attention(hidden, heads, kv_heads, qk_normalization)
        self.ffn_norm = nn.RMSNorm(hidden, eps=1e-5)
        self.feed_forward = FeedForward(hidden, intermediate)
        self.residual_dropout = nn.Dropout(dropout)

    def __call__(self, value):
        value = value + self.residual_dropout(self.attention(self.attention_norm(value)))
        return value + self.residual_dropout(self.feed_forward(self.ffn_norm(value)))

    def generate(self, value, cache=None):
        attended, next_cache = self.attention.generate(self.attention_norm(value), cache)
        value = value + self.residual_dropout(attended)
        return value + self.residual_dropout(self.feed_forward(self.ffn_norm(value))), next_cache


class DecoderLM(nn.Module):
    def __init__(self, architecture):
        super().__init__()
        self.architecture = architecture
        vocabulary = architecture["vocabulary_size"]
        hidden = architecture["hidden_size"]
        self.tie_embeddings = architecture["tie_embeddings"]
        self.embedding = nn.Embedding(vocabulary, hidden)
        self.layers = [
            DecoderBlock(
                hidden,
                architecture["intermediate_size"],
                architecture["attention_heads"],
                architecture["key_value_heads"],
                architecture.get("dropout", 0.0),
                architecture.get("qk_normalization", False),
            )
            for _ in range(architecture["layers"])
        ]
        self.norm = nn.RMSNorm(hidden, eps=1e-5)
        if not self.tie_embeddings:
            self.output = nn.Linear(hidden, vocabulary, bias=False)

    def initialize(self):
        normal = nn.init.normal(mean=0.0, std=0.02)
        self.embedding.weight = normal(self.embedding.weight)
        for layer in self.layers:
            for projection in (layer.attention.q_proj, layer.attention.k_proj, layer.attention.v_proj, layer.attention.o_proj, layer.feed_forward.gate, layer.feed_forward.up, layer.feed_forward.down):
                projection.weight = normal(projection.weight)
        if not self.tie_embeddings:
            self.output.weight = normal(self.output.weight)
        if self.architecture.get("initialization", "normal") == "depth-scaled":
            residual = nn.init.normal(mean=0.0, std=0.02 / (2 * len(self.layers)) ** 0.5)
            for layer in self.layers:
                layer.attention.o_proj.weight = residual(layer.attention.o_proj.weight)
                layer.feed_forward.down.weight = residual(layer.feed_forward.down.weight)

    def logits(self, value):
        value = self.norm(value)
        if self.tie_embeddings:
            return self.embedding.as_linear(value)
        return self.output(value)

    def __call__(self, tokens):
        value = self.embedding(tokens)
        for layer in self.layers:
            value = layer(value)
        return self.logits(value)

    def generate(self, tokens, cache=None):
        value = self.embedding(tokens)
        next_cache = []
        for index, layer in enumerate(self.layers):
            layer_cache = None if cache is None else cache[index]
            value, layer_cache = layer.generate(value, layer_cache)
            next_cache.append(layer_cache)
        return self.logits(value), next_cache


class ByteTokenizer:
    pad_id = 0
    bos_id = 1
    eos_id = 2
    byte_offset = 3

    def encode(self, text, add_eos=True):
        tokens = [byte + self.byte_offset for byte in text.encode("utf-8")]
        if add_eos:
            tokens.append(self.eos_id)
        return tokens

    def decode_token(self, token):
        if self.byte_offset <= token < self.byte_offset + 256:
            return bytes([token - self.byte_offset])
        return b""


# Worker scripts may retain older local definitions while schema-1 artifacts
# migrate. These aliases make the shared implementation authoritative.
WALDO_SHARED_ATTENTION = Attention
WALDO_SHARED_FEED_FORWARD = FeedForward
WALDO_SHARED_DECODER_BLOCK = DecoderBlock
WALDO_SHARED_DECODER_LM = DecoderLM
WALDO_SHARED_BYTE_TOKENIZER = ByteTokenizer
