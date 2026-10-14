# Copyright © 2026 Apple Inc.

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import mlx.core as mx
from mlx import nn

from .base import BaseModelArgs, create_attention_mask, scaled_dot_product_attention
from .llama import MLP as LlamaMLP
from .rope_utils import initialize_rope


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str = "deci"
    vocab_size: int = 32000
    hidden_size: int = 4096
    num_attention_heads: int = 32
    num_key_value_heads_per_layer: List[int] = field(default_factory=list)
    rope_theta: Optional[int] = None
    rope_scaling: Dict = field(default_factory=dict)
    max_position_embeddings: int = 8192
    intermediate_size: int = 14336
    num_hidden_layers: int = 32
    rms_norm_eps: float = 1e-05
    tie_word_embeddings: bool = False


class DeciAttention(nn.Module):

    def __init__(self, config: ModelArgs, layer_idx: int):
        super().__init__()

        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads_per_layer[layer_idx]
        self.scale = self.head_dim**-0.5  # Not sure

        self.rope_theta = (
            config.rope_theta if config.rope_theta is not None else 10000
        )  # Not sure
        self.rope_scaling = config.rope_scaling
        self.max_position_embeddings = config.max_position_embeddings
        self.rope = initialize_rope(
            dims=self.head_dim,
            base=self.rope_theta,
            traditional=False,
            scaling_config=self.rope_scaling,
            max_position_embeddings=self.max_position_embeddings,
        )

        self.q_proj = nn.Linear(
            self.hidden_size, self.num_attention_heads * self.head_dim, bias=False
        )
        self.k_proj = nn.Linear(
            self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False
        )
        self.v_proj = nn.Linear(
            self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False
        )
        self.o_proj = nn.Linear(
            self.num_attention_heads * self.head_dim, self.hidden_size, bias=False
        )

    def __call__(
        self,
        hidden_states: mx.array,
        cache: Optional[Any] = None,
        mask: Optional[Any] = None,
    ) -> mx.array:

        batch_size, query_length, _ = hidden_states.shape

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

        offset = 0
        if cache is not None:
            offset = cache.offset

        queries = self.rope(queries, offset)
        keys = self.rope(keys, offset)

        if cache is not None:
            keys, values = cache.update_and_fetch(keys, values)

        outputs = scaled_dot_product_attention(
            queries, keys, values, cache=cache, scale=self.scale, mask=mask
        )
        outputs = outputs.transpose(0, 2, 1, 3).reshape(batch_size, query_length, -1)

        return self.o_proj(outputs)


class DeciMLP(LlamaMLP):

    pass


class DeciDecoderLayer(nn.Module):

    def __init__(self, config: ModelArgs, layer_idx: int):
        super().__init__()

        self.hidden_size = config.hidden_size
        self.rms_norm_eps = config.rms_norm_eps

        self.self_attn = DeciAttention(config=config, layer_idx=layer_idx)
        self.mlp = DeciMLP(args=config)
        self.input_layernorm = nn.RMSNorm(self.hidden_size, eps=self.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(
            self.hidden_size, eps=self.rms_norm_eps
        )

    def __call__(
        self,
        hidden_states: mx.array,
        cache: Optional[Any] = None,
        mask: Optional[Any] = None,
    ) -> mx.array:

        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states=hidden_states, cache=cache, mask=mask
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


class DeciModel(nn.Module):

    def __init__(self, config: ModelArgs):
        super().__init__()

        self.vocab_size = config.vocab_size
        self.hidden_size = config.hidden_size
        self.rms_norm_eps = config.rms_norm_eps

        self.embed_tokens = nn.Embedding(self.vocab_size, self.hidden_size)
        self.layers = [
            DeciDecoderLayer(config=config, layer_idx=idx)
            for idx in range(config.num_hidden_layers)
        ]  # TODO
        self.norm = nn.RMSNorm(self.hidden_size, eps=self.rms_norm_eps)

    def __call__(
        self, hidden_states: mx.array, cache: Optional[Any] = None
    ) -> mx.array:

        hidden_states = self.embed_tokens(hidden_states)

        if cache is None:
            cache = [None] * len(self.layers)

        mask = create_attention_mask(hidden_states, cache[0])

        for c, block in zip(cache, self.layers):
            hidden_states = block(hidden_states=hidden_states, cache=c, mask=mask)

        return self.norm(hidden_states)


class Model(nn.Module):

    def __init__(self, config: ModelArgs):
        super().__init__()
        self.args = config
        self.model_type = config.model_type
        self.model = DeciModel(config=config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def __call__(
        self, hidden_states: mx.array, cache: Optional[Any] = None
    ) -> mx.array:

        hidden_states = self.model(hidden_states=hidden_states, cache=cache)
        if self.args.tie_word_embeddings:
            return self.model.word_embeddings.as_linear(hidden_states)
        else:
            return self.lm_head(hidden_states)

    @property
    def layers(self):
        return self.model.layers
