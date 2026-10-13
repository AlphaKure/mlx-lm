# Copyright © 2026 Apple Inc.

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import mlx.core as mx
from mlx import nn

from .base import BaseModelArgs, create_attention_mask, scaled_dot_product_attention
from .rope_utils import initialize_rope
from .switch_layers import SwitchGLU, swiglu


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str = "param2moe"
    vocab_size: int = 128000
    hidden_size: int = 2048
    num_attention_heads: int = 32
    head_dim: Optional[int] = None
    partial_rotary_factor: Optional[float] = None
    num_key_value_heads: int = 8
    rope_theta: int = 500000
    rope_parameters: Dict = field(default_factory=dict)
    max_position_embeddings: int = 32768
    use_qkv_bias: bool = False
    use_qk_norm: bool = True
    rms_norm_eps: float = 1e-06
    num_experts: Optional[int] = None
    first_k_dense_replace: int = 1
    intermediate_size: int = 9216
    moe_intermediate_size: int = 2048
    num_experts_per_tok: int = 6
    n_group: int = 1
    topk_group: int = 1
    routed_scaling_factor: float = 2.5
    norm_topk_prob: bool = True
    num_shared_experts: Optional[int] = None
    num_hidden_layers: int = 21
    moe_shared_expert_intermediate_size: int = 4096


class Param2MoEAttention(nn.Module):

    def __init__(self, config: ModelArgs):
        super().__init__()

        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.head_dim = (
            config.head_dim
            if config.head_dim is not None
            else self.hidden_size // self.num_attention_heads
        )
        self.num_key_value_heads = config.num_key_value_heads
        self.scaling = self.head_dim**-0.5

        partial_rotary_factor = (
            config.partial_rotary_factor
            if config.partial_rotary_factor is not None
            else 1.0
        )
        self.rope_dim = int(self.head_dim * partial_rotary_factor)
        self.rope_base = config.rope_theta
        self.rope_parameters = config.rope_parameters
        self.max_position_embeddings = config.max_position_embeddings
        self.rope = initialize_rope(
            dims=self.rope_dim,
            base=self.rope_base,
            traditional=False,
            scaling_config=self.rope_parameters,
            max_position_embeddings=self.max_position_embeddings,
        )

        self.use_qkv_bias = config.use_qkv_bias
        self.query_key_value = nn.Linear(
            self.hidden_size,
            (self.num_attention_heads + 2 * self.num_key_value_heads) * self.head_dim,
            bias=self.use_qkv_bias,
        )

        self.use_qk_norm = config.use_qk_norm
        self.rms_norm_eps = config.rms_norm_eps
        if self.use_qk_norm:
            self.query_layernorm = nn.RMSNorm(self.head_dim, eps=self.rms_norm_eps)
            self.key_layernorm = nn.RMSNorm(self.head_dim, eps=self.rms_norm_eps)

        self.dense = nn.Linear(
            self.num_attention_heads * self.head_dim,
            self.hidden_size,
            bias=self.use_qkv_bias,
        )

    def __call__(
        self,
        hidden_states: mx.array,
        cache: Optional[Any] = None,
        mask: Optional[Any] = None,
    ) -> mx.array:

        batch_size, query_length, _ = hidden_states.shape

        qkv = self.query_key_value(hidden_states).reshape(
            batch_size,
            query_length,
            self.num_attention_heads + 2 * self.num_key_value_heads,
            -1,
        )

        queries, keys, values = mx.split(
            qkv,
            [
                self.num_attention_heads,
                self.num_attention_heads + self.num_key_value_heads,
            ],
            axis=2,
        )
        queries = queries.transpose(0, 2, 1, 3)
        keys = keys.transpose(0, 2, 1, 3)
        values = values.transpose(0, 2, 1, 3)

        if self.use_qk_norm:

            queries = self.query_layernorm(queries)
            keys = self.key_layernorm(keys)

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
        return self.dense(outputs)


class Param2MoEMLP(nn.Module):

    def __init__(self, config: ModelArgs, intermediate_size: Optional[int] = None):
        super().__init__()

        self.hidden_size = config.hidden_size
        self.intermediate_size = (
            intermediate_size
            if intermediate_size is not None
            else config.intermediate_size
        )

        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)

    def __call__(self, hidden_states: mx.array) -> mx.array:
        return self.down_proj(
            swiglu(self.gate_proj(hidden_states), self.up_proj(hidden_states))
        )


@mx.compile
def group_expert_select(
    gates,
    e_score_correction_bias,
    top_k,
    n_group,
    topk_group,
    routed_scaling_factor,
    norm_topk_prob,
):

    scores = mx.sigmoid(gates.astype(mx.float32))
    orig_scores = scores
    scores = scores + e_score_correction_bias
    if n_group > 1:
        scores = mx.unflatten(scores, axis=-1, shape=(n_group, -1))
        group_scores = mx.topk(scores, 2, axis=-1).sum(axis=-1, keepdims=True)
        k = n_group - topk_group
        group_idx = mx.argpartition(group_scores, kth=k - 1, axis=-2)[..., :k, :]
        scores = mx.put_along_axis(
            scores, mx.stop_gradient(group_idx), mx.array(0.0), axis=-2
        )
        scores = mx.flatten(scores, -2, -1)

    k = top_k
    inds = mx.argpartition(-scores, kth=k - 1, axis=-1)[..., :k]
    inds = mx.stop_gradient(inds)
    scores = mx.take_along_axis(orig_scores, inds, axis=-1)
    if top_k > 1 and norm_topk_prob:
        denominator = scores.sum(axis=-1, keepdims=True)
        scores = scores / (denominator + 1e-20)
    scores = scores * routed_scaling_factor

    return inds, scores


class Param2MoEGate(nn.Module):

    def __init__(self, config: ModelArgs):
        super().__init__()

        self.top_k = config.num_experts_per_tok
        self.num_experts = config.num_experts
        self.routed_scaling_factor = config.routed_scaling_factor
        self.n_group = config.n_group
        self.topk_group = config.topk_group
        self.norm_topk_prob = config.norm_topk_prob

        self.weight = mx.zeros((self.num_experts, config.hidden_size))
        self.expert_bias = mx.zeros((self.num_experts,))

    def __call__(self, hidden_status: mx.array) -> mx.array:
        return group_expert_select(
            hidden_status @ self.weight.T,
            self.expert_bias,
            self.top_k,
            self.n_group,
            self.topk_group,
            self.routed_scaling_factor,
            self.norm_topk_prob,
        )


class Param2MoESparseMoEBlock(nn.Module):

    def __init__(self, config: ModelArgs):
        super().__init__()

        self.hidden_size = config.hidden_size
        self.experts_hidden_size = config.moe_intermediate_size
        self.num_experts = config.num_experts
        self.intermediate_size = config.intermediate_size

        self.experts = SwitchGLU(
            input_dims=self.hidden_size,
            hidden_dims=self.experts_hidden_size,
            num_experts=self.num_experts,
            bias=False,
        )
        self.gate = Param2MoEGate(config=config)
        self.num_shared_experts = config.num_shared_experts
        if self.num_shared_experts is not None:
            self.shared_experts = Param2MoEMLP(
                config=config,
                intermediate_size=config.moe_shared_expert_intermediate_size,
            )

    def __call__(self, hidden_states: mx.array) -> mx.array:

        inds, scores = self.gate(hidden_states)
        expert_outputs = self.experts(hidden_states, indices=inds)
        expert_outputs = (
            (expert_outputs * scores[..., None])
            .sum(axis=-2)
            .astype(expert_outputs.dtype)
        )
        if self.num_shared_experts is not None:
            expert_outputs = expert_outputs + self.shared_experts(hidden_states)

        return expert_outputs


class Param2MoEDecoderLayer(nn.Module):

    def __init__(self, config: ModelArgs, layer_idx: int):
        super().__init__()

        self.hidden_size = config.hidden_size
        self.rms_norm_eps = config.rms_norm_eps

        self.attention = Param2MoEAttention(config=config)
        if config.num_experts is not None and layer_idx >= config.first_k_dense_replace:
            self.mlp = Param2MoESparseMoEBlock(config=config)
        else:
            self.mlp = Param2MoEMLP(config=config)

        self.input_layernorm = nn.RMSNorm(self.hidden_size, self.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(self.hidden_size, self.rms_norm_eps)

    def __call__(
        self,
        hidden_states: mx.array,
        cache: Optional[Any] = None,
        mask: Optional[Any] = None,
    ) -> mx.array:

        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.attention(
            hidden_states=hidden_states, cache=cache, mask=mask
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


class Param2MoEModel(nn.Module):

    def __init__(self, config: ModelArgs):
        super().__init__()

        self.vocab_size = config.vocab_size
        self.hidden_size = config.hidden_size
        self.rms_norm_eps = config.rms_norm_eps

        self.word_embeddings = nn.Embedding(self.vocab_size, self.hidden_size)
        self.layers = [
            Param2MoEDecoderLayer(config=config, layer_idx=idx)
            for idx in range(config.num_hidden_layers)
        ]
        self.norm = nn.RMSNorm(self.hidden_size, self.rms_norm_eps)

    def __call__(
        self, hidden_states: mx.array, cache: Optional[Any] = None
    ) -> mx.array:

        hidden_states = self.word_embeddings(hidden_states)

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
        self.model = Param2MoEModel(config=config)

    def __call__(
        self, hidden_states: mx.array, cache: Optional[Any] = None
    ) -> mx.array:

        hidden_states = self.model(hidden_states=hidden_states, cache=cache)
        return self.model.word_embeddings.as_linear(hidden_states)

    def sanitize(self, weights):
        for layer_idx in range(self.args.num_hidden_layers):

            if layer_idx < self.args.first_k_dense_replace:
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

        return weights

    @property
    def layers(self):
        return self.model.layers
