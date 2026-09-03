from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import mlx.core as mx
from mlx import nn

from .activations import swiglu
from .base import BaseModelArgs, create_attention_mask, scaled_dot_product_attention
from .rope_utils import initialize_rope
from .switch_layers import SwitchGLU


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
    scores = mx.take_along_axis(orig_scores, inds, axis=-1)
    if top_k > 1 and norm_topk_prob:
        denominator = scores.sum(axis=-1, keepdims=True)
        scores = scores / denominator
    scores = scores * routed_scaling_factor

    return inds, scores


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str = "mobilemoe"
    vocab_size: int = 128256
    hidden_size: int = 1280
    layer_types: List = field(default_factory=list)
    head_dim: int = 64
    num_attention_heads: int = 20
    num_key_value_heads: int = 4
    attention_bias: bool = False
    use_qk_norm: bool = True
    no_rope_layers: List = field(default_factory=list)
    rms_norm_eps: float = 1e-05
    rope_theta: float = 500000.0
    rope_scaling: Dict = field(default_factory=dict)
    max_position_embeddings: int = 8192
    interleave_moe_layer_step: int = 1
    num_hidden_layers: int = 32
    intermediate_size: int = 640
    num_experts_per_tok: int = 4
    num_local_experts: int = 60
    attn_temperature_tuning: bool = False
    floor_scale: int = 8192
    attn_scale: float = 0.1
    routed_scaling_factor: float = 2.5
    norm_topk_prob: bool = True
    intermediate_size_mlp: int = 2560

    def __post_init__(self):
        self.moe_layers = list(
            range(
                self.interleave_moe_layer_step - 1,
                self.num_hidden_layers,
                self.interleave_moe_layer_step,
            )
        )


class MobileMoEAttention(nn.Module):

    def __init__(self, config: ModelArgs, layer_index: int):
        super().__init__()

        self.head_dim = config.head_dim
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.rms_norm_eps = config.rms_norm_eps

        self.num_key_value_groups = self.num_attention_heads // self.num_key_value_heads
        self.attn_temperature_tuning = config.attn_temperature_tuning
        self.floor_scale = config.floor_scale
        self.attn_scale = config.attn_scale
        self.use_rope = config.no_rope_layers[layer_index]
        self.use_qk_norm = config.use_qk_norm

        self.q_proj = nn.Linear(
            config.hidden_size,
            self.head_dim * self.num_attention_heads,
            bias=config.attention_bias,
        )
        self.k_proj = nn.Linear(
            config.hidden_size,
            self.head_dim * self.num_key_value_heads,
            bias=config.attention_bias,
        )
        self.v_proj = nn.Linear(
            config.hidden_size,
            self.head_dim * self.num_key_value_heads,
            bias=config.attention_bias,
        )
        self.o_proj = nn.Linear(
            self.head_dim * self.num_attention_heads,
            config.hidden_size,
            bias=config.attention_bias,
        )

        if self.use_rope:
            self.rope = initialize_rope(
                dims=self.head_dim,
                traditional=False,
                base=config.rope_theta,
                scaling_config=config.rope_scaling,
                max_position_embeddings=config.max_position_embeddings,
            )

    def __call__(
        self,
        hidden_states: mx.array,
        cache: Optional[Any] = None,
        mask: Optional[Any] = None,
    ) -> mx.array:

        batch_size, query_length, hidden_dim = hidden_states.shape

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

        if cache is not None:
            offset = cache.offset
        else:
            offset = 0

        if self.use_rope:
            queries = self.rope(queries, offset=offset)
            keys = self.rope(keys, offset=offset)

        if self.use_qk_norm and self.use_rope:
            queries = mx.fast.rms_norm(x=queries, weight=None, eps=self.rms_norm_eps)
            keys = mx.fast.rms_norm(x=keys, weight=None, eps=self.rms_norm_eps)

        if self.attn_temperature_tuning and not self.use_rope:

            # Use from llama4
            attn_scales = (
                mx.log(
                    mx.floor(
                        mx.arange(offset + 1, offset + query_length + 1)
                        / self.floor_scale
                    )
                    + 1.0
                )
                * self.attn_scale
                + 1.0
            )
            attn_scales = attn_scales[:, None]
            queries = (queries * attn_scales).astype(queries.dtype)

        if cache is not None:
            keys, values = cache.update_and_fetch(keys, values)

        attn_outputs = scaled_dot_product_attention(
            queries=queries,
            keys=keys,
            values=values,
            cache=cache,
            mask=mask,
            scale=self.scaling,
        )

        attn_outputs = attn_outputs.transpose(0, 2, 1, 3).reshape(
            batch_size, query_length, -1
        )
        return self.o_proj(attn_outputs)


class MobileMoEMLP(nn.Module):

    def __init__(self, config: ModelArgs, intermediate_size: Optional[int] = None):
        super().__init__()

        if intermediate_size is None:
            intermediate_size = config.intermediate_size

        self.gate_proj = nn.Linear(config.hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, config.hidden_size, bias=False)

    def __call__(self, hidden_states: mx.array) -> mx.array:

        return self.down_proj(
            swiglu(self.gate_proj(hidden_states), self.up_proj(hidden_states))
        )


class MobileMoETextMoE(nn.Module):

    def __init__(self, config: ModelArgs):
        super().__init__()

        self.top_k = config.num_experts_per_tok
        self.hidden_dim = config.hidden_size
        self.num_experts = config.num_local_experts
        self.routed_scaling_factor = config.routed_scaling_factor
        self.norm_topk_prob = config.norm_topk_prob

        self.experts = SwitchGLU(
            input_dims=self.hidden_dim,
            hidden_dims=config.intermediate_size,
            num_experts=self.num_experts,
            bias=False,
        )
        self.router = nn.Linear(
            config.hidden_size, config.num_local_experts, bias=False
        )
        self.e_score_correction_bias = mx.zeros((self.num_experts,))

        self.shared_expert = MobileMoEMLP(
            config=config, intermediate_size=config.intermediate_size_mlp
        )

    def __call__(self, hidden_states: mx.array) -> mx.array:
        inds, scores = group_expert_select(
            gates=self.router(hidden_states),
            e_score_correction_bias=self.e_score_correction_bias,
            top_k=self.top_k,
            n_group=1,
            topk_group=1,
            routed_scaling_factor=self.routed_scaling_factor,
            norm_topk_prob=self.norm_topk_prob,
        )
        expert_outputs = self.experts(hidden_states, inds)
        expert_outputs = (
            (expert_outputs * scores[..., None])
            .sum(axis=-2)
            .astype(expert_outputs.dtype)
        )

        return expert_outputs + self.shared_expert(hidden_states)


class MobileMoEDecoderLayer(nn.Module):

    def __init__(self, config: ModelArgs, layer_index: int):
        super().__init__()

        self.hidden_size = config.hidden_size
        self.attention_type = config.layer_types[layer_index]
        self.is_moe_layer = layer_index in config.moe_layers

        self.self_attn = MobileMoEAttention(config=config, layer_index=layer_index)
        if self.is_moe_layer:
            self.feed_forward = MobileMoETextMoE(config)
        else:
            self.feed_forward = MobileMoEMLP(
                config=config, intermediate_size=config.intermediate_size_mlp
            )

        self.input_layernorm = nn.RMSNorm(self.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(
            self.hidden_size, eps=config.rms_norm_eps
        )

    def __call__(
        self,
        hidden_states: mx.array,
        cache: Optional[Any] = None,
        mask: Optional[Any] = None,
    ) -> mx.array:

        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        attn_output = self.self_attn(hidden_states, cache=cache, mask=mask)
        hidden_states = residual + attn_output

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.feed_forward(hidden_states)

        return residual + hidden_states


class MobileMoEModel(nn.Module):

    def __init__(self, config: ModelArgs):
        super().__init__()

        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(self.vocab_size, config.hidden_size)
        self.layers = [
            MobileMoEDecoderLayer(config=config, layer_index=index)
            for index in range(config.num_hidden_layers)
        ]
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

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

        self.model = MobileMoEModel(config=config)

    def __call__(self, inputs: mx.array, cache: Optional[Any] = None) -> mx.array:

        hidden_states = self.model(inputs, cache)

        return self.model.embed_tokens.as_linear(hidden_states)

    @property
    def layers(self):
        return self.model.layers

    def sanitize(self, weights):
        for l in range(self.args.num_hidden_layers):

            if f"model.layers.{l}.feed_forward.expert_bias" in weights:
                bias = weights.pop(f"model.layers.{l}.feed_forward.expert_bias")
                weights[f"model.layers.{l}.feed_forward.e_score_correction_bias"] = bias

            prefix = f"model.layers.{l}.feed_forward.experts"
            if f"{prefix}.gate_up_proj" in weights:
                v = weights.pop(f"{prefix}.gate_up_proj")
                gate_k = f"{prefix}.gate_proj.weight"
                up_k = f"{prefix}.up_proj.weight"
                gate_proj, up_proj = mx.split(v, 2, axis=-1)
                weights[gate_k] = mx.swapaxes(gate_proj, 1, 2)
                weights[up_k] = mx.swapaxes(up_proj, 1, 2)
            if f"{prefix}.down_proj" in weights:
                down_proj = weights.pop(f"{prefix}.down_proj")
                weights[f"{prefix}.down_proj.weight"] = mx.swapaxes(down_proj, 1, 2)
        return weights
