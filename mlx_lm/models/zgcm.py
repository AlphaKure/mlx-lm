# Copyright © 2026 Apple Inc.

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import mlx.core as mx
from mlx import nn

from .base import BaseModelArgs, create_attention_mask, scaled_dot_product_attention
from .cache import KVCache, RotatingKVCache
from .rope_utils import initialize_rope


@dataclass
class ModelArgs(BaseModelArgs):

    model_type: str = "zgcm"
    vocab_size: int = 155136
    hidden_size: int = 4096
    num_attention_heads: int = 32
    num_key_value_heads: int = 8
    head_dim: int = 128
    layer_types: List[str] = field(default_factory=list)
    sliding_window: int = 128
    attention_gate_layers: List[bool] = field(default=list)
    rms_norm_eps: float = 1e-06
    rope_theta: float = 10000000.0
    rope_scaling: Optional[Dict] = None
    max_position_embeddings: int = 262144
    intermediate_size: int = 11008
    num_hidden_layers: int = 32
    attention_bias: bool = False
    # tie_word_embeddings: bool = False
    partial_rotary_factor: float = 1.0


class ZgcmAttention(nn.Module):

    def __init__(self, config: ModelArgs, layer_idx: int):
        super().__init__()

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.scaling = self.head_dim**-0.5
        self.has_gate = config.attention_gate_layers[layer_idx]
        self.rms_norm_eps = config.rms_norm_eps
        self.use_bias = config.attention_bias

        self.q_proj = nn.Linear(
            self.hidden_size, self.head_dim * self.num_heads, bias=self.use_bias
        )
        self.k_proj = nn.Linear(
            self.hidden_size,
            self.head_dim * self.num_key_value_heads,
            bias=self.use_bias,
        )
        self.v_proj = nn.Linear(
            self.hidden_size,
            self.head_dim * self.num_key_value_heads,
            bias=self.use_bias,
        )
        self.o_proj = nn.Linear(
            self.head_dim * self.num_heads, self.hidden_size, bias=self.use_bias
        )

        self.q_norm = nn.RMSNorm(self.head_dim, eps=self.rms_norm_eps)
        self.k_norm = nn.RMSNorm(self.head_dim, eps=self.rms_norm_eps)

        if self.has_gate:
            self.g_proj = nn.Linear(
                self.hidden_size, self.head_dim * self.num_heads, bias=False
            )

        self.rope_base = config.rope_theta
        self.rope_scaling = config.rope_scaling
        self.max_position_embeddings = config.max_position_embeddings
        self.rope = initialize_rope(
            dims=int(self.head_dim * config.partial_rotary_factor),
            base=self.rope_base,
            traditional=False,
            scaling_config=self.rope_scaling,
            max_position_embeddings=self.max_position_embeddings,
        )

    def __call__(
        self,
        hidden_states: mx.array,
        cache: Optional[Any] = None,
        mask: Optional[Any] = None,
    ) -> mx.array:

        batch_size, query_length, _ = hidden_states.shape

        queries = self.q_proj(hidden_states).reshape(
            batch_size, query_length, self.num_heads, -1
        )
        keys = self.k_proj(hidden_states).reshape(
            batch_size, query_length, self.num_key_value_heads, -1
        )
        values = (
            self.v_proj(hidden_states)
            .reshape(batch_size, query_length, self.num_key_value_heads, -1)
            .transpose(0, 2, 1, 3)
        )

        queries = self.q_norm(queries).transpose(0, 2, 1, 3)
        keys = self.k_norm(keys).transpose(0, 2, 1, 3)

        offset = 0
        if cache is not None:
            offset = cache.offset

        queries = self.rope(queries, offset)
        keys = self.rope(keys, offset)

        if cache is not None:
            keys, values = cache.update_and_fetch(keys, values)

        outputs = scaled_dot_product_attention(
            queries, keys, values, cache=cache, scale=self.scaling, mask=mask
        )

        outputs = outputs.transpose(0, 2, 1, 3).reshape(batch_size, query_length, -1)
        if self.has_gate:
            gates = nn.sigmoid(self.g_proj(hidden_states)).astype(outputs.dtype)
            outputs = gates * outputs

        return self.o_proj(outputs)


class ZgcmMLP(nn.Module):

    def __init__(self, config: ModelArgs):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size

        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)

    def __call__(self, hidden_states: mx.array) -> mx.array:

        return self.down_proj(
            nn.silu(self.gate_proj(hidden_states)) * self.up_proj(hidden_states)
        )


class ZgcmDecoderLayer(nn.Module):

    def __init__(self, config: ModelArgs, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.rms_norm_eps = config.rms_norm_eps
        self.use_sliding = config.layer_types[layer_idx] == "sliding_attention"

        self.self_attn = ZgcmAttention(config=config, layer_idx=layer_idx)
        self.post_attention_layernorm = nn.RMSNorm(self.hidden_size, self.rms_norm_eps)
        self.post_feedforward_layernorm = nn.RMSNorm(
            self.hidden_size, self.rms_norm_eps
        )
        self.mlp = ZgcmMLP(config=config)

    def __call__(
        self,
        hidden_states: mx.array,
        cache: Optional[Any] = None,
        mask: Optional[Any] = None,
    ) -> mx.array:

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        attn_outputs = self.self_attn(hidden_states, cache=cache, mask=mask)
        hidden_states = residual + attn_outputs

        residual = hidden_states
        hidden_states = self.post_feedforward_layernorm(hidden_states)
        hidden_states = residual + self.mlp(hidden_states)
        return hidden_states


class ZgcmModel(nn.Module):

    def __init__(self, config: ModelArgs):
        super().__init__()

        self.vocab_size = config.vocab_size
        self.hidden_size = config.hidden_size
        self.num_layers = config.num_hidden_layers
        self.rms_norm_eps = config.rms_norm_eps
        self.fa_idx = config.layer_types.index("full_attention")
        self.swa_idx = config.layer_types.index("sliding_attention")
        self.sliding_window = config.sliding_window

        self.embed_tokens = nn.Embedding(self.vocab_size, self.hidden_size)

        self.layers = [
            ZgcmDecoderLayer(config=config, layer_idx=idx)
            for idx in range(self.num_layers)
        ]

        self.norm = nn.RMSNorm(self.hidden_size, self.rms_norm_eps)

    def __call__(
        self, hidden_states: mx.array, cache: Optional[Any] = None
    ) -> mx.array:

        hidden_states = self.embed_tokens(hidden_states)

        if cache is None:

            cache = [None] * len(self.layers)

        full_mask = (
            create_attention_mask(hidden_states, cache=cache[self.fa_idx])
            if self.fa_idx is not None
            else None
        )
        swa_mask = (
            create_attention_mask(
                hidden_states,
                cache=cache[self.swa_idx],
                window_size=self.sliding_window,
            )
            if self.swa_idx is not None
            else None
        )

        for c, layer in zip(cache, self.layers):
            hidden_states = layer(
                hidden_states=hidden_states,
                cache=c,
                mask=swa_mask if layer.use_sliding else full_mask,
            )

        return self.norm(hidden_states)


class Model(nn.Module):

    def __init__(self, config: ModelArgs):
        super().__init__()

        self.model_type = config.model_type
        self.args = config

        self.model = ZgcmModel(config=config)
        self.lm_head = nn.Linear(
            self.args.hidden_size, self.args.vocab_size, bias=False
        )

    def __call__(self, inputs: mx.array, cache: Optional[Any] = None) -> mx.array:

        hidden_states = self.model(inputs, cache=cache)

        return self.lm_head(hidden_states)

    @property
    def layers(self):
        return self.model.layers

    def make_cache(self):
        caches = []
        for layer in self.layers:
            if layer.use_sliding:
                caches.append(RotatingKVCache(max_size=self.model.sliding_window))
            else:
                caches.append(KVCache())
        return caches
