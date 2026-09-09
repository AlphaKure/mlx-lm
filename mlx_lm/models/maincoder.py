# Copyright © 2026 Apple Inc.

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import mlx.core as mx
from mlx import nn

from .base import BaseModelArgs, create_attention_mask, scaled_dot_product_attention
from .rope_utils import initialize_rope


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str = "maincoder"
    vocab_size: int = 151936
    hidden_size: int = 1536
    head_dim: int = 96
    num_attention_heads: int = 16
    num_key_value_heads: int = 4
    use_qk_norm: bool = True
    rope_scaling: Optional[Dict] = field(default_factory=dict)
    rope_theta: float = 1000000.0
    max_position_embeddings: int = 2048
    rms_norm_eps: float = 1e-05
    intermediate_size_mlp: int = 4096
    num_hidden_layers: int = 32


class MaincoderAttention(nn.Module):

    def __init__(self, config: ModelArgs):
        super().__init__()

        self.head_dim = config.head_dim
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.use_qk_norm = config.use_qk_norm
        self.rms_norm_eps = config.rms_norm_eps

        self.q_proj = nn.Linear(
            config.hidden_size, self.head_dim * self.num_attention_heads, bias=False
        )
        self.k_proj = nn.Linear(
            config.hidden_size, self.head_dim * self.num_key_value_heads, bias=False
        )
        self.v_proj = nn.Linear(
            config.hidden_size, self.head_dim * self.num_key_value_heads, bias=False
        )
        self.o_proj = nn.Linear(
            self.head_dim * self.num_attention_heads, config.hidden_size, bias=False
        )

        if self.use_qk_norm:
            self.q_norm = nn.RMSNorm(config.head_dim, eps=config.rms_norm_eps)
            self.k_norm = nn.RMSNorm(config.head_dim, eps=config.rms_norm_eps)

        self.rope = initialize_rope(
            dims=self.head_dim,
            base=config.rope_theta,
            scaling_config=config.rope_scaling,
            traditional=True,
            max_position_embeddings=config.max_position_embeddings,
        )

    def __call__(
        self,
        hidden_states: mx.array,
        cache: Optional[Any] = None,
        mask: Optional[Any] = None,
    ) -> mx.array:

        batch_size, query_length, hidden_dimesion = hidden_states.shape

        queries = (
            self.q_proj(hidden_states)
            .reshape(batch_size, query_length, self.num_attention_heads, -1)
            .transpose(0, 2, 1, 3)
        )
        keys = (
            self.k_proj(hidden_states)
            .reshape(batch_size, query_length, self.num_key_value_heads, -1)
            .transpose(0, 2, 1, 3)
        )
        values = (
            self.v_proj(hidden_states)
            .reshape(batch_size, query_length, self.num_key_value_heads, -1)
            .transpose(0, 2, 1, 3)
        )

        offset = 0 if cache is None else cache.offset

        # RoPE
        queries = self.rope(queries, offset=offset)
        keys = self.rope(keys, offset=offset)

        # QK norm
        if self.use_qk_norm:
            queries = self.q_norm(queries)
            keys = self.k_norm(keys)

        if cache is not None:
            keys, values = cache.update_and_fetch(keys, values)

        output = scaled_dot_product_attention(
            queries, keys, values, cache=cache, scale=self.scaling, mask=mask
        )

        output = output.transpose(0, 2, 1, 3).reshape(batch_size, query_length, -1)
        return self.o_proj(output)


class MaincoderMLP(nn.Module):

    def __init__(self, config: ModelArgs):
        super().__init__()

        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size_mlp

        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)

    def __call__(self, hidden_states: mx.array) -> mx.array:
        return self.down_proj(
            nn.silu(self.gate_proj(hidden_states)) * self.up_proj(hidden_states)
        )


class MaincoderDecoderLayer(nn.Module):

    def __init__(self, config: ModelArgs):
        super().__init__()

        self.self_attn = MaincoderAttention(config=config)
        self.feed_forward = MaincoderMLP(config=config)
        self.input_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def __call__(
        self,
        hidden_states: mx.array,
        cache: Optional[Any] = None,
        mask: Optional[Any] = None,
    ) -> mx.array:

        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, cache=cache, mask=mask)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.feed_forward(hidden_states)

        return residual + hidden_states


class MaincoderModel(nn.Module):

    def __init__(self, config: ModelArgs):
        super().__init__()

        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(self.vocab_size, dims=config.hidden_size)
        self.layers = [
            MaincoderDecoderLayer(config=config)
            for _ in range(config.num_hidden_layers)
        ]

        self.norm = nn.RMSNorm(dims=config.hidden_size, eps=config.rms_norm_eps)

    def __call__(
        self, hidden_states: mx.array, cache: Optional[Any] = None
    ) -> mx.array:

        hidden_states = self.embed_tokens(hidden_states)

        if cache is None:
            cache = [None] * len(self.layers)

        mask = create_attention_mask(hidden_states, cache=cache[0])

        for c, block in zip(cache, self.layers):
            hidden_states = block(hidden_states=hidden_states, cache=c, mask=mask)

        return self.norm(hidden_states)


class Model(nn.Module):

    def __init__(self, config: ModelArgs):
        super().__init__()

        self.model_type = config.model_type
        self.args = config

        self.model = MaincoderModel(config=config)

    def __call__(
        self, hidden_states: mx.array, cache: Optional[Any] = None
    ) -> mx.array:

        hidden_states = self.model(hidden_states=hidden_states, cache=cache)
        return self.model.embed_tokens.as_linear(hidden_states)

    @property
    def layers(self):
        return self.model.layers
