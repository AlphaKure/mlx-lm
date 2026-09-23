# Copyright © 2026 Apple Inc.

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Tuple

import mlx.core as mx
from mlx import nn

from .base import BaseModelArgs, create_attention_mask, scaled_dot_product_attention
from .rope_utils import initialize_rope
from .switch_layers import SwitchGLU, SwitchLinear, swiglu


def split_to_interleaved(x: mx.array) -> mx.array:
    # Inputs:  x0 x1 x2 x3 ... y0 y1 y2 y3 ...
    # Outputs: x0 y0 x1 y1 x2 y2 x3 y3 ...
    shape = x.shape
    return x.reshape(*shape[:-1], 2, -1).transpose(-1, -2).reshape(shape)


def interleaved_to_split(x: mx.array) -> mx.array:
    # Inputs: x0 y0 x1 y1 x2 y2 x3 y3 ...
    # Outputs: x0 x1 x2 x3 ... y0 y1 y2 y3 ...
    shape = x.shape
    return x.reshape(*shape[:-1], -1, 2).transpose(-1, -2).reshape(shape)


@dataclass
class ModelArgs(BaseModelArgs):
    """
    "initializer_range": 0.02,
    "output_router_logits": false,
    "router_aux_loss_coef": 0.001,
    "sliding_window": null,
    "tie_word_embeddings": false,
    "use_sliding_window": false,
    """

    model_type: str = "k2_horizon"
    vocab_size: int = 250624
    hidden_size: int = 2560
    mlp_only_layers: List[int] = field(default_factory=list)
    num_experts: int = 100
    decoder_sparse_step: int = 1
    mova_num_experts: int = 64
    head_dim: int = 128
    num_attention_heads: int = 32
    num_key_value_heads: int = 8
    rope_head_dim: Optional[int] = None
    rope_parameters: Dict = field(default_factory=dict)
    max_position_embeddings: int = 524288
    attention_bias: bool = False
    attention_gate_func: Optional[str] = None
    query_key_norm: bool = False
    rms_norm_eps: float = 1e-06
    moe_gate_bias: bool = True
    router_scaling_factor: Optional[float] = None
    router_score_func: str = "sigmoid"
    num_experts_per_tok: int = 8
    mova_num_experts_per_tok: int = 4
    intermediate_size: int = 6144
    norm_topk_prob: bool = True
    num_shared_experts: int = 1
    moe_intermediate_size: int = 768
    layernorm_num_groups: int = 2
    num_hidden_layers: int = 48
    rope_theta: Optional[float] = None

    def __post_init__(self):

        if self.rope_theta is None:

            assert self.rope_parameters.get("rope_theta", None)
            self.rope_theta = float(self.rope_parameters.get("rope_theta"))


@mx.compile
def calc_router_weights(
    logits: mx.array,
    score_func: Literal["softmax", "sigmoid"],
    top_k: int,
    scaling_factor: Optional[float],
    bias: Optional[mx.array] = None,
    norm_topk_prob: Optional[bool] = False,
) -> Tuple[mx.array, mx.array]:

    if score_func == "sigmoid":
        routing_score = nn.sigmoid(logits).astype(mx.float32)
    elif score_func == "softmax":
        routing_score = nn.softmax(logits).astype(mx.float32)

    selection_scores = routing_score
    if bias is not None:
        selection_scores = selection_scores + bias

    selection_indices = mx.argpartition(-selection_scores, kth=top_k - 1, axis=-1)[
        ..., :top_k
    ]
    selection_indices = mx.stop_gradient(selection_indices)
    routing_weights = mx.take_along_axis(routing_score, selection_indices, axis=-1)
    if top_k > 1 or norm_topk_prob:
        routing_weights = routing_weights / routing_weights.sum(axis=-1, keepdims=True)
    if scaling_factor is not None:
        routing_weights = routing_weights * scaling_factor
    return routing_weights, selection_indices


class K2HorizonRMSNorm(nn.Module):
    # Group RMS Norm
    def __init__(self, hidden_size: int, groups: int, eps: float):
        super().__init__()
        self.hidden_size = hidden_size
        self.eps = eps
        self.n_groups = groups
        assert hidden_size % groups == 0

        self.weight = mx.ones((hidden_size,))

    def __call__(self, hidden_states: mx.array) -> mx.array:

        weight = self.weight.reshape((self.n_groups, -1))
        original_shape = hidden_states.shape
        hidden_states = mx.unflatten(hidden_states, axis=-1, shape=(self.n_groups, -1))
        hidden_states = mx.fast.rms_norm(hidden_states, weight=None, eps=self.eps)
        hidden_states = hidden_states * weight
        return hidden_states.reshape(original_shape)


class K2HorizonAttention(nn.Module):

    def __init__(self, config: ModelArgs):
        super().__init__()

        self.head_dim = config.head_dim
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.use_bias = config.attention_bias
        self.scaling = self.head_dim**-0.5

        self.rope_head_dim = (
            config.rope_head_dim
            if config.rope_head_dim is not None
            else config.head_dim
        )
        self.rope_base = config.rope_theta
        self.rope_parameters = config.rope_parameters
        self.max_position_embeddings = config.max_position_embeddings
        self.rope = initialize_rope(
            dims=self.rope_head_dim,
            base=self.rope_base,
            traditional=False,
            scaling_config=self.rope_parameters,
            max_position_embeddings=self.max_position_embeddings,
        )

        self.q_proj = nn.Linear(
            self.hidden_size,
            self.head_dim * self.num_attention_heads,
            bias=self.use_bias,
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
            self.head_dim * self.num_attention_heads,
            self.hidden_size,
            bias=self.use_bias,
        )

        self.gate_func = config.attention_gate_func
        if self.gate_func is not None:
            self.gate_proj = nn.Linear(
                config.hidden_size,
                config.num_attention_heads * config.head_dim,
                bias=False,
            )

        self.use_qk_norm = config.query_key_norm
        if self.use_qk_norm:
            self.q_norm = K2HorizonRMSNorm(
                config.num_attention_heads * config.head_dim,
                groups=config.num_attention_heads,
                eps=config.rms_norm_eps,
            )
            self.k_norm = K2HorizonRMSNorm(
                config.num_key_value_heads * config.head_dim,
                groups=config.num_key_value_heads,
                eps=config.rms_norm_eps,
            )

    def __call__(
        self,
        hidden_states: mx.array,
        cache: Optional[Any] = None,
        mask: Optional[Any] = None,
    ) -> mx.array:

        batch_size, query_length, _ = hidden_states.shape

        if self.use_qk_norm:
            queries = (
                self.q_norm(self.q_proj(hidden_states))
                .reshape(batch_size, query_length, self.num_attention_heads, -1)
                .transpose(0, 2, 1, 3)
            )
            keys = (
                self.k_norm(self.k_proj(hidden_states))
                .reshape(batch_size, query_length, self.num_key_value_heads, -1)
                .transpose(0, 2, 1, 3)
            )
        else:
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

        if self.rope_head_dim == self.head_dim:
            queries = self.rope(queries, offset)
            keys = self.rope(keys, offset)
        else:
            q_interleaved = split_to_interleaved(queries)
            k_interleaved = split_to_interleaved(keys)

            q_rot, q_pass = mx.split(q_interleaved, [self.rope_head_dim], axis=-1)

            k_rot, k_pass = mx.split(k_interleaved, [self.rope_head_dim], axis=-1)

            q_rot = self.rope(interleaved_to_split(q_rot), offset)
            k_rot = self.rope(interleaved_to_split(k_rot), offset)

            queries = interleaved_to_split(
                mx.concatenate([split_to_interleaved(q_rot), q_pass], axis=-1)
            )
            keys = interleaved_to_split(
                mx.concatenate([split_to_interleaved(k_rot), k_pass], axis=-1)
            )

        if cache is not None:
            keys, values = cache.update_and_fetch(keys, values)

        outputs = scaled_dot_product_attention(
            queries=queries,
            keys=keys,
            values=values,
            cache=cache,
            scale=self.scaling,
            mask=mask,
        )

        if self.gate_func is not None:

            gates = (
                self.gate_proj(hidden_states)
                .reshape(batch_size, query_length, -1, self.head_dim)
                .transpose(0, 2, 1, 3)
            )
            if self.gate_func == "silu":
                gates = nn.silu(gates)
            else:
                assert self.gate_func == "softplus"
                # mx base use 1
                # offical code use log2
                beta = math.log(2)
                gates = mx.logaddexp(mx.array(0.0), beta * gates) / beta

            outputs = outputs * gates

        outputs = outputs.transpose(0, 2, 1, 3).reshape(batch_size, query_length, -1)
        return self.o_proj(outputs)


class K2HorizonMoVAAttention(nn.Module):

    def __init__(self, config: ModelArgs):
        super().__init__()

        self.head_dim = config.head_dim
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.mova_num_experts = config.mova_num_experts
        self.use_bias = config.attention_bias
        self.scaling = self.head_dim**-0.5
        self.router_score_func = config.router_score_func
        self.router_scaling_factor = config.router_scaling_factor
        self.num_experts_per_tok = config.mova_num_experts_per_tok

        self.rope_head_dim = (
            config.rope_head_dim
            if config.rope_head_dim is not None
            else config.head_dim
        )
        self.rope_base = config.rope_theta
        self.rope_parameters = config.rope_parameters
        self.max_position_embeddings = config.max_position_embeddings
        self.rope = initialize_rope(
            dims=self.rope_head_dim,
            base=self.rope_base,
            traditional=False,
            scaling_config=self.rope_parameters,
            max_position_embeddings=self.max_position_embeddings,
        )

        self.q_proj = nn.Linear(
            self.hidden_size,
            self.head_dim * self.num_attention_heads,
            bias=self.use_bias,
        )
        self.k_proj = nn.Linear(
            self.hidden_size,
            self.head_dim * self.num_key_value_heads,
            bias=self.use_bias,
        )
        self.o_proj = nn.Linear(
            self.head_dim * self.num_attention_heads,
            self.hidden_size,
            bias=self.use_bias,
        )
        self.v_router = nn.Linear(
            self.hidden_size, self.mova_num_experts, bias=config.moe_gate_bias
        )
        self.v_experts = SwitchLinear(
            input_dims=self.hidden_size,
            output_dims=self.num_key_value_heads * self.head_dim,
            num_experts=self.mova_num_experts,
            bias=False,
        )

        self.gate_func = config.attention_gate_func
        if self.gate_func is not None:
            self.gate_proj = nn.Linear(
                config.hidden_size,
                config.num_attention_heads * config.head_dim,
                bias=False,
            )

        self.use_qk_norm = config.query_key_norm
        if self.use_qk_norm:
            self.q_norm = K2HorizonRMSNorm(
                config.num_attention_heads * config.head_dim,
                groups=config.num_attention_heads,
                eps=config.rms_norm_eps,
            )
            self.k_norm = K2HorizonRMSNorm(
                config.num_key_value_heads * config.head_dim,
                groups=config.num_key_value_heads,
                eps=config.rms_norm_eps,
            )

    def __call__(
        self,
        hidden_states: mx.array,
        cache: Optional[Any] = None,
        mask: Optional[Any] = None,
    ) -> mx.array:

        batch_size, query_length, _ = hidden_states.shape

        flat_hidden_stats = hidden_states.reshape(-1, hidden_states.shape[-1])

        logits = self.v_router(flat_hidden_stats)

        bias = None
        if self.v_router.bias is not None:
            bias = self.v_router.bias
            logits = logits - bias

        routers, inds = calc_router_weights(
            logits=logits,
            score_func=self.router_score_func,
            top_k=self.num_experts_per_tok,
            bias=bias,
            scaling_factor=self.router_scaling_factor,
        )

        expert_inputs = mx.expand_dims(flat_hidden_stats, axis=(-2, -3))
        values = self.v_experts(expert_inputs, inds)
        values = values.squeeze(-2)
        values = nn.silu(values)
        values = (values * routers[..., None]).sum(axis=-2)
        values = values.reshape(
            batch_size, query_length, self.num_key_value_heads, -1
        ).transpose(0, 2, 1, 3)

        if self.use_qk_norm:
            queries = (
                self.q_norm(self.q_proj(hidden_states))
                .reshape(batch_size, query_length, self.num_attention_heads, -1)
                .transpose(0, 2, 1, 3)
            )
            keys = (
                self.k_norm(self.k_proj(hidden_states))
                .reshape(batch_size, query_length, self.num_key_value_heads, -1)
                .transpose(0, 2, 1, 3)
            )
        else:
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

        offset = 0
        if cache is not None:
            offset = cache.offset

        if self.rope_head_dim == self.head_dim:
            queries = self.rope(queries, offset)
            keys = self.rope(keys, offset)
        else:
            q_interleaved = split_to_interleaved(queries)
            k_interleaved = split_to_interleaved(keys)

            q_rot, q_pass = mx.split(q_interleaved, [self.rope_head_dim], axis=-1)

            k_rot, k_pass = mx.split(k_interleaved, [self.rope_head_dim], axis=-1)

            q_rot = self.rope(interleaved_to_split(q_rot), offset)
            k_rot = self.rope(interleaved_to_split(k_rot), offset)

            queries = interleaved_to_split(
                mx.concatenate([split_to_interleaved(q_rot), q_pass], axis=-1)
            )
            keys = interleaved_to_split(
                mx.concatenate([split_to_interleaved(k_rot), k_pass], axis=-1)
            )

        if cache is not None:
            keys, values = cache.update_and_fetch(keys, values)

        outputs = scaled_dot_product_attention(
            queries=queries,
            keys=keys,
            values=values,
            cache=cache,
            scale=self.scaling,
            mask=mask,
        )

        if self.gate_func is not None:

            gates = (
                self.gate_proj(hidden_states)
                .reshape(batch_size, query_length, -1, self.head_dim)
                .transpose(0, 2, 1, 3)
            )
            if self.gate_func == "silu":
                gates = nn.silu(gates)
            else:
                assert self.gate_func == "softplus"
                # mx base use 1
                # offical code use log2
                beta = math.log(2)
                gates = mx.logaddexp(mx.array(0.0), beta * gates) / beta

            outputs = outputs * gates

        outputs = outputs.transpose(0, 2, 1, 3).reshape(batch_size, query_length, -1)
        return self.o_proj(outputs)


class K2HorizonMoe(nn.Module):

    def __init__(self, config: ModelArgs):
        super().__init__()

        self.hidden_size = config.hidden_size
        self.gate_use_bias = config.moe_gate_bias
        self.intermediate_size = config.moe_intermediate_size

        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_tok
        self.norm_topk_prob = config.norm_topk_prob
        self.num_shared_experts = config.num_shared_experts
        self.router_score_func = config.router_score_func
        self.router_scaling_factor = config.router_scaling_factor

        self.gate = nn.Linear(self.hidden_size, self.num_experts, self.gate_use_bias)
        self.experts = SwitchGLU(
            input_dims=self.hidden_size,
            hidden_dims=self.intermediate_size,
            num_experts=self.num_experts,
            bias=False,
        )

        if self.num_shared_experts > 0:
            self.shared_experts = K2HorizonMLP(
                config=config,
                intermediate_size=self.intermediate_size * self.num_shared_experts,
            )

    def __call__(self, hidden_states: mx.array) -> mx.array:

        logits = self.gate(hidden_states)

        bias = None
        if self.gate.bias is not None:
            bias = self.gate.bias
            logits = logits - bias

        routers, inds = calc_router_weights(
            logits=logits,
            bias=bias,
            score_func=self.router_score_func,
            top_k=self.top_k,
            scaling_factor=self.router_scaling_factor,
            norm_topk_prob=self.norm_topk_prob,
        )

        experts = self.experts(hidden_states, inds)
        experts = (experts * routers[..., None]).sum(axis=-2)
        if self.num_shared_experts > 0:
            return experts + self.shared_experts(hidden_states)
        else:
            return experts


class K2HorizonMLP(nn.Module):

    def __init__(self, config: ModelArgs, intermediate_size: Optional[int] = None):

        super().__init__()
        self.intermediate_size = (
            intermediate_size
            if intermediate_size is not None
            else config.intermediate_size
        )
        self.hidden_size = config.hidden_size

        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)

    def __call__(self, hidden_states: mx.array) -> mx.array:

        return self.down_proj(
            swiglu(self.gate_proj(hidden_states), self.up_proj(hidden_states))
        )


class K2HorizonDecoderLayer(nn.Module):

    def __init__(self, config: ModelArgs, layer_idx: int):

        super().__init__()
        self.hidden_size = config.hidden_size
        self.layernorm_num_groups = config.layernorm_num_groups
        self.rms_norm_eps = config.rms_norm_eps

        self.is_sparse_layer = (layer_idx not in config.mlp_only_layers) and (
            config.num_experts > 0 and (layer_idx + 1) % config.decoder_sparse_step == 0
        )

        if self.is_sparse_layer and config.mova_num_experts > 0:
            self.self_attn = K2HorizonMoVAAttention(config=config)
        else:
            self.self_attn = K2HorizonAttention(config=config)

        if self.is_sparse_layer:
            self.mlp = K2HorizonMoe(config=config)
        else:
            self.mlp = K2HorizonMLP(config=config)

        self.input_layernorm = K2HorizonRMSNorm(
            hidden_size=self.hidden_size,
            groups=self.layernorm_num_groups,
            eps=self.rms_norm_eps,
        )
        self.post_attention_layernorm = K2HorizonRMSNorm(
            hidden_size=self.hidden_size,
            groups=self.layernorm_num_groups,
            eps=self.rms_norm_eps,
        )

    def __call__(
        self,
        hidden_states: mx.array,
        cache: Optional[Any] = None,
        mask: Optional[Any] = None,
    ):

        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        attn_outpus = self.self_attn(
            hidden_states=hidden_states, cache=cache, mask=mask
        )
        hidden_states = residual + attn_outpus

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        mlp_outputs = self.mlp(hidden_states)
        hidden_states = residual + mlp_outputs

        return hidden_states


class K2HorizonModel(nn.Module):

    def __init__(self, config: ModelArgs):

        super().__init__()
        self.vocab_size = config.vocab_size
        self.hidden_size = config.hidden_size
        self.num_norm_groups = config.layernorm_num_groups
        self.rms_norm_eps = config.rms_norm_eps

        self.embed_tokens = nn.Embedding(self.vocab_size, self.hidden_size)
        self.layers = [
            K2HorizonDecoderLayer(config=config, layer_idx=idx)
            for idx in range(config.num_hidden_layers)
        ]

        self.norm = K2HorizonRMSNorm(
            hidden_size=self.hidden_size,
            groups=self.num_norm_groups,
            eps=self.rms_norm_eps,
        )

    def __call__(
        self, hidden_states: mx.array, cache: Optional[Any] = None
    ) -> mx.array:

        hidden_states = self.embed_tokens(hidden_states)

        if cache is None:
            cache = [None] * len(self.layers)

        mask = create_attention_mask(hidden_states, cache[0])

        for c, layer in zip(cache, self.layers):
            hidden_states = layer(hidden_states, cache=c, mask=mask)

        return self.norm(hidden_states)


class Model(nn.Module):

    def __init__(self, config: ModelArgs):
        super().__init__()

        self.args = config
        self.model_type = config.model_type
        self.model = K2HorizonModel(config=config)
        self.lm_head = nn.Linear(
            self.args.hidden_size, self.args.vocab_size, bias=False
        )

    def __call__(
        self, hidden_states: mx.array, cache: Optional[Any] = None
    ) -> mx.array:

        hidden_states = self.model(hidden_states=hidden_states, cache=cache)
        return self.lm_head(hidden_states)

    @property
    def layers(self):
        return self.model.layers

    def sanitize(self, weights):

        for layer_idx in range(self.args.num_hidden_layers):

            if not self.layers[layer_idx].is_sparse_layer:

                continue

            prefix = f"model.layers.{layer_idx}.mlp.experts"

            for name in ["gate_proj", "up_proj", "down_proj"]:
                key = f"{prefix}.0.{name}.weight"
                if key not in weights:
                    continue

                expert_weights = [
                    weights.pop(f"{prefix}.{expert_idx}.{name}.weight")
                    for expert_idx in range(self.args.num_experts)
                ]
                weights[f"{prefix}.{name}.weight"] = mx.stack(expert_weights)

            prefix = f"model.layers.{layer_idx}.self_attn.v_experts"

            if f"{prefix}.0.weight" not in weights:
                continue

            mova_expert_weights = [
                weights.pop(f"{prefix}.{expert_idx}.weight")
                for expert_idx in range(self.args.mova_num_experts)
            ]

            weights[f"{prefix}.weight"] = mx.stack(mova_expert_weights)

        return weights
