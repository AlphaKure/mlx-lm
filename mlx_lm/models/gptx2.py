from dataclasses import dataclass
from typing import Any, Optional

import mlx.core as mx
from mlx import nn

from .activations import swiglu
from .base import BaseModelArgs, create_attention_mask, scaled_dot_product_attention


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str = "gptx2"
    vocab_size: int = 32770
    hidden_size: int = 576
    rms_norm_eps: float = 0.000001
    num_attention_heads: int = 9
    num_key_value_heads: int = 3
    head_dim: int = 64
    rope_theta: float = 100000.0
    xsa_projection: bool = True
    intermediate_size: int = 1728
    num_hidden_layers: int = 30
    tie_word_embeddings: bool = True


class GPTX2Attention(nn.Module):

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_head = args.num_attention_heads
        self.n_kv_heads = args.num_key_value_heads
        self.head_dim = args.head_dim
        self.n_rep = self.n_head // self.n_kv_heads
        self.xsa_projection = args.xsa_projection
        self.scale = self.head_dim**-0.5

        self.q_proj = nn.Linear(
            args.hidden_size, self.n_head * self.head_dim, bias=False
        )
        self.k_proj = nn.Linear(
            args.hidden_size, self.n_kv_heads * self.head_dim, bias=False
        )
        self.v_proj = nn.Linear(
            args.hidden_size, self.n_kv_heads * self.head_dim, bias=False
        )
        self.o_proj = nn.Linear(
            self.n_head * self.head_dim, args.hidden_size, bias=False
        )
        self.o_proj.NANOGPT_SCALE_INIT = 1

        self.rope = nn.RoPE(
            dims=self.head_dim,
            traditional=True,
            base=args.rope_theta,
        )

    def __call__(
        self,
        hidden_states: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:

        batch_size, query_length, _ = hidden_states.shape

        queries, keys, values = (
            self.q_proj(hidden_states),
            self.k_proj(hidden_states),
            self.v_proj(hidden_states),
        )

        queries = queries.reshape(batch_size, query_length, self.n_head, -1).transpose(
            0, 2, 1, 3
        )
        keys = keys.reshape(batch_size, query_length, self.n_kv_heads, -1).transpose(
            0, 2, 1, 3
        )
        values = values.reshape(
            batch_size, query_length, self.n_kv_heads, -1
        ).transpose(0, 2, 1, 3)
        current_values = values  # save value for xsa

        if cache is not None:
            queries = self.rope(queries, offset=cache.offset)
            keys = self.rope(keys, offset=cache.offset)
            keys, values = cache.update_and_fetch(keys, values)
        else:
            queries = self.rope(queries)
            keys = self.rope(keys)

        k_repeated = mx.repeat(keys, self.n_rep, axis=1)
        v_repeated = mx.repeat(values, self.n_rep, axis=1)

        output = scaled_dot_product_attention(
            queries, k_repeated, v_repeated, cache=cache, scale=self.scale, mask=mask
        )

        if self.xsa_projection:
            output = output.reshape(
                batch_size, self.n_kv_heads, self.n_rep, query_length, self.head_dim
            )
            v_grouped = current_values[:, :, None, :, :]
            denominator = (v_grouped * v_grouped).sum(axis=-1, keepdims=True)
            denominator = mx.maximum(denominator, 1e-6)
            projection = (
                (output * v_grouped).sum(axis=-1, keepdims=True) / denominator
            ) * v_grouped
            output = output - projection
            output = output.reshape(
                batch_size, self.n_head, query_length, self.head_dim
            )
        output = output.transpose(0, 2, 1, 3).reshape(batch_size, query_length, -1)
        return self.o_proj(output)


class GPTX2MLP(nn.Module):

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.w_up = nn.Linear(args.hidden_size, args.intermediate_size, bias=False)
        self.w_gate = nn.Linear(args.hidden_size, args.intermediate_size, bias=False)
        self.w_down = nn.Linear(args.intermediate_size, args.hidden_size, bias=False)
        self.w_down.NANOGPT_SCALE_INIT = 1

    def __call__(self, hidden_states: mx.array) -> mx.array:
        return self.w_down(swiglu(self.w_gate(hidden_states), self.w_up(hidden_states)))


class GPTX2Block(nn.Module):

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.ln_1 = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.attn = GPTX2Attention(args=args)
        self.ln_2 = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.mlp = GPTX2MLP(args=args)

    def __call__(
        self,
        hidden_states: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:

        hidden_states = hidden_states + self.attn(
            self.ln_1(hidden_states), mask=mask, cache=cache
        )

        return hidden_states + self.mlp(self.ln_2(hidden_states))


class GPTX2Model(nn.Module):

    def __init__(self, args: ModelArgs):
        super().__init__()

        self.wte = nn.Embedding(args.vocab_size, dims=args.hidden_size)
        self.h = [GPTX2Block(args=args) for _ in range(args.num_hidden_layers)]
        self.ln_f = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

    def __call__(
        self,
        hidden_states: mx.array,
        cache: Optional[Any] = None,
    ) -> mx.array:

        batch_size, query_length = hidden_states.shape

        hidden_states = self.wte(hidden_states)

        if cache is None:
            cache = [None] * len(self.h)

        mask = create_attention_mask(hidden_states, cache[0])

        for layer, c in zip(self.h, cache):
            hidden_states = layer(hidden_states, mask, cache=c)

        return self.ln_f(hidden_states)


class Model(nn.Module):

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = GPTX2Model(args=args)
        if not args.tie_word_embeddings:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(self, inputs: mx.array, cache: Optional[Any] = None) -> mx.array:

        hidden_states = self.model(inputs, cache=cache)
        if self.args.tie_word_embeddings:
            return self.model.wte.as_linear(hidden_states)
        else:
            return self.lm_head(hidden_states)

    def sanitize(self, weights):
        new_weights = {}
        for weight in weights:
            if weight.startswith("transformer."):
                new_weights[weight.replace("transformer.", "model.")] = weights[weight]
            else:
                new_weights[weight] = weights[weight]
        return new_weights

    @property
    def layers(self):
        return self.model.h
