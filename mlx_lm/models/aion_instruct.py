# Copyright © 2026 Apple Inc.

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional

import mlx.core as mx
from mlx import nn

from .base import BaseModelArgs, create_attention_mask, scaled_dot_product_attention
from .rope_utils import initialize_rope


@dataclass
class ModelArgs(BaseModelArgs):
    """
    "initializer_range": 0.02,
    "tie_word_embeddings": true,
    """

    model_type: str = "aion_instruct"
    vocab_size: int = 200029
    hidden_size: int = 2048
    head_dim: int = 128
    num_attention_heads: int = 24
    num_key_value_heads: int = 8
    attention_bias: bool = False
    rms_norm_eps: float = 1e-06
    rope_parameters: Dict = field(default_factory=dict)
    max_position_embeddings: int = 32768
    layer_types: List = field(default_factory=list)
    intermediate_size: int = 8192
    num_hidden_layers: int = 28
    sliding_window: int = 2048


class AionInstructAttention(nn.Module):

    def __init__(self, config: ModelArgs, layer_types: Literal["fa", "swa"]):
        super().__init__()

        self.hidden_size = config.hidden_size
        self.head_dim = config.head_dim
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads

        self.scaling = self.head_dim**-0.5

        self.q_proj = nn.Linear(
            self.hidden_size,
            self.head_dim * self.num_attention_heads,
            bias=config.attention_bias,
        )
        self.k_proj = nn.Linear(
            self.hidden_size,
            self.head_dim * self.num_key_value_heads,
            bias=config.attention_bias,
        )
        self.v_proj = nn.Linear(
            self.hidden_size,
            self.head_dim * self.num_key_value_heads,
            bias=config.attention_bias,
        )
        self.o_proj = nn.Linear(
            self.head_dim * self.num_attention_heads,
            self.hidden_size,
            bias=config.attention_bias,
        )

        self.q_norm = nn.RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = nn.RMSNorm(self.head_dim, eps=config.rms_norm_eps)

        if layer_types == "fa":
            rope_parameters = config.rope_parameters["full_attention"]
        else:
            rope_parameters = config.rope_parameters["sliding_attention"]
        rope_theta = rope_parameters["rope_theta"]
        self.max_position_embeddings = config.max_position_embeddings
        self.rope = initialize_rope(
            dims=self.head_dim,
            base=rope_theta,
            traditional=False,
            scaling_config=rope_parameters,
            max_position_embeddings=self.max_position_embeddings,
        )

    def __call__(
        self,
        hidden_states: mx.array,
        cache: Optional[Any] = None,
        mask: Optional[Any] = None,
    ) -> mx.array:

        batch_size, query_length, _ = hidden_states.shape

        queries = self.q_norm(
            self.q_proj(hidden_states)
            .reshape(batch_size, query_length, self.num_attention_heads, -1)
            .transpose(0, 2, 1, 3)
        )
        keys = self.k_norm(
            self.k_proj(hidden_states)
            .reshape(batch_size, query_length, self.num_key_value_heads, -1)
            .transpose(0, 2, 1, 3)
        )
        values = (
            self.v_proj(hidden_states)
            .reshape(batch_size, query_length, self.num_key_value_heads, -1)
            .transpose(0, 2, 1, 3)
        )

        offset = 0
        if cache is not None:
            offset = cache.offset

        queries = self.rope(queries, offset)
        keys = self.rope(keys, offset)

        if cache is not None:
            keys, values = cache.update_and_fetch(keys, values)

        outputs = scaled_dot_product_attention(
            queries, keys, values, cache=cache, mask=mask, scale=self.scaling
        )

        outputs = outputs.transpose(0, 2, 1, 3).reshape(batch_size, query_length, -1)

        return self.o_proj(outputs)


class AionInstructMLP(nn.Module):

    def __init__(self, config: ModelArgs):
        super().__init__()

        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size

        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)

    def __call__(self, hidden_states: mx.array) -> mx.array:
        return self.down_proj(
            nn.silu(self.gate_proj(hidden_states)) * self.up_proj(hidden_states)
        )


class AionInstructDecoderLayer(nn.Module):

    def __init__(self, config: ModelArgs, layer_idx: int):
        super().__init__()
        self.layer_types = (
            "fa" if config.layer_types[layer_idx] == "full_attention" else "swa"
        )

        self.self_attn = AionInstructAttention(
            config=config, layer_types=self.layer_types
        )
        self.mlp = AionInstructMLP(config=config)
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
        attn_outputs = self.self_attn(hidden_states, cache, mask)
        hidden_states = residual + attn_outputs

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        mlp_output = self.mlp(hidden_states)
        return residual + mlp_output


class AionInstructModel(nn.Module):

    def __init__(self, config: ModelArgs):
        super().__init__()
        self.config = config

        self.vocab_size = config.vocab_size
        self.hidden_size = config.hidden_size

        self.embed_tokens = nn.Embedding(self.vocab_size, self.hidden_size)
        self.layers = [
            AionInstructDecoderLayer(config=config, layer_idx=idx)
            for idx in range(config.num_hidden_layers)
        ]
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.fa_idx = config.layer_types.index("full_attention")
        self.swa_idx = config.layer_types.index("sliding_attention")

    def __call__(
        self, hidden_states: mx.array, cache: Optional[Any] = None
    ) -> mx.array:

        hidden_states = self.embed_tokens(hidden_states)

        if cache is None:
            cache = [None] * len(self.layers)

        full_mask = create_attention_mask(h=hidden_states, cache=cache[self.fa_idx])
        sliding_mask = create_attention_mask(
            h=hidden_states,
            cache=cache[self.swa_idx],
            window_size=self.config.sliding_window,
        )

        for c, layer in zip(cache, self.layers):
            mask = full_mask if layer.layer_types == "fa" else sliding_mask
            hidden_states = layer(hidden_states, cache=c, mask=mask)

        return self.norm(hidden_states)


class Model(nn.Module):

    def __init__(self, config: ModelArgs):
        super().__init__()
        self.args = config
        self.model_type = config.model_type
        self.model = AionInstructModel(config=config)

    def __call__(
        self, hidden_states: mx.array, cache: Optional[Any] = None
    ) -> mx.array:

        hidden_states = self.model(hidden_states=hidden_states, cache=cache)
        return self.model.embed_tokens.as_linear(hidden_states)

    @property
    def layers(self):
        return self.model.layers
