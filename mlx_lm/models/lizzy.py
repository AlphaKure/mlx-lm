from dataclasses import dataclass, field
from typing import Any, Optional

import mlx.core as mx
from mlx import nn

from .base import BaseModelArgs, create_attention_mask, scaled_dot_product_attention
from .rope_utils import initialize_rope


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str = "lizzy"
    vocab_size: int = 100278
    hidden_size: int = 4096
    layer_types: list = field(
        default_factory=list
    )  # slidling attention, full attention
    # layer_layouts: list = field(default_factory= list) # decoder_postnorm
    num_attention_heads: int = 32
    num_key_value_heads: int = 32
    head_dim: int = 128
    rope_layer_flags: list = field(default_factory=list)  # true false
    rope_type_overrides: dict = field(default_factory=dict)
    rope_theta: int = 500000
    rope_scaling: dict = field(default_factory=dict)
    max_position_embeddings: int = 65536
    norm_eps: float = 1e-06
    use_qk_norm: bool = True
    intermediate_size: int = 11008
    use_pre_attn_norm: bool = False
    use_pre_mlp_norm: bool = False
    use_post_attn_norm: bool = True
    use_post_mlp_norm: bool = True
    num_hidden_layers: int = 32
    sliding_window: int = 4096


class LizzyAttention(nn.Module):

    def __init__(self, layer_index: int, config: ModelArgs):
        super().__init__()

        self.layer_type = config.layer_types[layer_index]

        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.head_dim = config.head_dim
        self.hidden_size = config.hidden_size
        self.scaling = self.head_dim**-0.5
        self.use_rope = config.rope_layer_flags[layer_index]
        self.rope_override = config.rope_type_overrides.get(self.layer_type, None)
        if self.use_rope:

            rope_type = self.rope_override or config.rope_scaling.get(
                "rope_type", "default"
            )
            if rope_type == "dynamic":
                # dynamic
                self.rope = initialize_rope(
                    dims=self.head_dim,
                    base=config.rope_theta,
                    traditional=False,
                    scaling_config=config.rope_scaling,
                    max_position_embeddings=config.max_position_embeddings,
                )
            else:
                # non-dynamic RoPE (default, YaRN, etc)
                self.rope = initialize_rope(
                    dims=self.head_dim,
                    base=config.rope_theta,
                    traditional=False,
                    scaling_config=config.rope_scaling,
                )

        # linears
        self.q_proj = nn.Linear(
            self.hidden_size, self.num_heads * self.head_dim, bias=False
        )
        self.k_proj = nn.Linear(
            self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False
        )
        self.v_proj = nn.Linear(
            self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim, self.hidden_size, bias=False
        )

        # norms
        self.use_qk_norm = config.use_qk_norm
        if self.use_qk_norm:
            self.q_norm = nn.RMSNorm(
                self.num_heads * self.head_dim, eps=config.norm_eps
            )
            self.k_norm = nn.RMSNorm(
                self.num_key_value_heads * self.head_dim, eps=config.norm_eps
            )

    def __call__(
        self,
        inputs: mx.array,
        cache: Optional[Any] = None,
        mask: Optional[Any] = None,
    ) -> mx.array:

        batch_size, query_length, _ = inputs.shape

        queries = self.q_proj(inputs)
        keys = self.k_proj(inputs)
        values = self.v_proj(inputs)

        if self.use_qk_norm:
            queries = self.q_norm(queries)
            keys = self.k_norm(keys)

        queries = queries.reshape(
            batch_size, query_length, self.num_heads, -1
        ).transpose(0, 2, 1, 3)
        keys = keys.reshape(
            batch_size, query_length, self.num_key_value_heads, -1
        ).transpose(0, 2, 1, 3)
        values = values.reshape(
            batch_size, query_length, self.num_key_value_heads, -1
        ).transpose(0, 2, 1, 3)

        if cache is not None:
            if self.use_rope:
                queries = self.rope(queries, offset=cache.offset)
                keys = self.rope(keys, offset=cache.offset)
            keys, values = cache.update_and_fetch(keys, values)
        else:
            if self.use_rope:
                queries = self.rope(queries)
                keys = self.rope(keys)

        output = scaled_dot_product_attention(
            queries, keys, values, cache=cache, scale=self.scaling, mask=mask
        )

        output = output.transpose(0, 2, 1, 3).reshape(batch_size, query_length, -1)
        return self.o_proj(output)


class LizzyMLP(nn.Module):

    def __init__(self, config: ModelArgs):
        super().__init__()

        self.gate_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=False
        )
        self.up_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=False
        )
        self.down_proj = nn.Linear(
            config.intermediate_size, config.hidden_size, bias=False
        )

    def __call__(
        self,
        inputs: mx.array,
    ) -> mx.array:

        return self.down_proj(nn.silu(self.gate_proj(inputs)) * self.up_proj(inputs))


class LizzyDecoderLayer(nn.Module):

    def __init__(self, layer_index: int, config: ModelArgs):
        super().__init__()

        self.self_attn = LizzyAttention(layer_index=layer_index, config=config)
        self.mlp = LizzyMLP(config=config)

        self.pre_attn_norm = (
            nn.RMSNorm(dims=config.hidden_size, eps=config.norm_eps)
            if config.use_pre_attn_norm
            else None
        )
        self.pre_mlp_norm = (
            nn.RMSNorm(dims=config.hidden_size, eps=config.norm_eps)
            if config.use_pre_mlp_norm
            else None
        )
        self.post_attn_norm = (
            nn.RMSNorm(dims=config.hidden_size, eps=config.norm_eps)
            if config.use_post_attn_norm
            else None
        )
        self.post_mlp_norm = (
            nn.RMSNorm(dims=config.hidden_size, eps=config.norm_eps)
            if config.use_post_mlp_norm
            else None
        )

    def __call__(
        self,
        inputs: mx.array,
        cache: Optional[Any] = None,
        mask: Optional[Any] = None,
    ) -> mx.array:

        residual = inputs
        attn_inputs = (
            self.pre_attn_norm(inputs) if self.pre_attn_norm is not None else inputs
        )
        attn_outputs = self.self_attn(attn_inputs, cache=cache, mask=mask)
        if self.post_attn_norm is not None:
            attn_outputs = self.post_attn_norm(attn_outputs)
        hidden_states = residual + attn_outputs

        residual = hidden_states
        mlp_inputs = (
            self.pre_mlp_norm(hidden_states)
            if self.pre_mlp_norm is not None
            else hidden_states
        )
        mlp_outputs = self.mlp(mlp_inputs)
        if self.post_mlp_norm is not None:
            mlp_outputs = self.post_mlp_norm(mlp_outputs)

        hidden_states = residual + mlp_outputs
        return hidden_states


class LizzyModel(nn.Module):

    def __init__(self, config: ModelArgs):
        super().__init__()

        self.config = config
        self.vocab_size = config.vocab_size
        self.hidden_size = config.hidden_size

        self.embed_tokens = nn.Embedding(self.vocab_size, self.hidden_size)

        self.layers = [
            LizzyDecoderLayer(layer_index=index, config=config)
            for index in range(config.num_hidden_layers)
        ]
        self.norm = nn.RMSNorm(dims=config.hidden_size, eps=config.norm_eps)

    def __call__(
        self,
        inputs: mx.array,
        cache: Optional[Any] = None,
    ) -> mx.array:

        hidden_status = self.embed_tokens(inputs)

        if cache is None:
            cache = [None] * len(self.layers)

        full_mask = create_attention_mask(h=hidden_status, cache=cache[0])
        sliding_mask = create_attention_mask(
            h=hidden_status, cache=cache[0], window_size=self.config.sliding_window
        )

        index = 0
        for c, layer in zip(cache, self.layers):
            mask = (
                full_mask
                if self.config.layer_types[index] == "full_attention"
                else sliding_mask
            )
            hidden_status = layer(hidden_status, cache=c, mask=mask)
            index += 1

        return self.norm(hidden_status)


class Model(nn.Module):

    def __init__(self, config: ModelArgs):
        super().__init__()

        self.model_type = "lizzy"
        self.model = LizzyModel(config=config)

        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def __call__(
        self,
        inputs: mx.array,
        cache: Optional[Any] = None,
    ) -> mx.array:

        hidden_states = self.model(inputs=inputs, cache=cache)
        return self.lm_head(hidden_states)

    @property
    def layers(self):
        return self.model.layers
