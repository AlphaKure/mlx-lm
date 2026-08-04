from dataclasses import dataclass
from typing import Any, Optional

import mlx.core as mx
from mlx import nn

from .base import BaseModelArgs, create_attention_mask, scaled_dot_product_attention


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str = "bloom"
    n_embed: int = 14336
    num_attention_heads: int = 112
    vocab_size: int = 250880
    layer_norm_epsilon: float = 1e-05
    n_layer: int = 70


class BloomAttention(nn.Module):

    def __init__(self, args: ModelArgs):
        super().__init__()

        self.hidden_size = args.n_embed
        self.num_heads = args.num_attention_heads

        self.head_dim = self.hidden_size // self.num_heads

        if self.head_dim * self.num_heads != self.hidden_size:
            raise ValueError(
                f"`hidden_size` must be divisible by num_heads (got `hidden_size`: {self.hidden_size} and `num_heads`:"
                f" {self.num_heads})."
            )

        self.inv_norm_factor = self.head_dim**-0.5

        self.query_key_value = nn.Linear(
            self.hidden_size, 3 * self.hidden_size, bias=True
        )
        self.dense = nn.Linear(self.hidden_size, self.hidden_size, bias=True)

    def __call__(
        self,
        hidden_states: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:

        batch_size, length, _ = hidden_states.shape

        qkv = self.query_key_value(hidden_states)
        qkv = qkv.reshape(batch_size, length, self.num_heads, 3, self.head_dim)

        queries = qkv[..., 0, :].transpose(0, 2, 1, 3)
        keys = qkv[..., 1, :].transpose(0, 2, 1, 3)
        values = qkv[..., 2, :].transpose(0, 2, 1, 3)

        offset = 0
        if cache is not None:
            offset = cache.offset
            keys, values = cache.update_and_fetch(keys, values)

        score = nn.ALiBi.create_alibi_matrix(
            q_sequence_length=length + offset,
            k_sequence_length=keys.shape[2],
            num_heads=self.num_heads,
            offset=offset,
            dtype=queries.dtype,
        )

        if mask is not None:
            if mask.dtype == mx.bool_:
                mask = mx.where(
                    mask, mx.array(0.0, dtype=score.dtype), mx.finfo(score.dtype).min
                )
            mask = score + mask
        else:
            mask = score

        output = scaled_dot_product_attention(
            queries, keys, values, cache=cache, scale=self.inv_norm_factor, mask=mask
        )
        output = output.transpose(0, 2, 1, 3).reshape(batch_size, length, -1)
        return self.dense(output)


class BloomMLP(nn.Module):

    def __init__(self, args: ModelArgs):
        super().__init__()

        hidden_size = args.n_embed

        self.dense_h_to_4h = nn.Linear(hidden_size, 4 * hidden_size, bias=True)
        self.dense_4h_to_h = nn.Linear(4 * hidden_size, hidden_size, bias=True)

    def __call__(self, hidden_state: mx.array) -> mx.array:

        hidden_state = self.dense_h_to_4h(hidden_state)
        hidden_state = nn.gelu_approx(hidden_state)
        return self.dense_4h_to_h(hidden_state)


class BloomBlock(nn.Module):

    def __init__(self, args: ModelArgs):
        super().__init__()

        self.hidden_size = args.n_embed

        self.input_layernorm = nn.LayerNorm(
            self.hidden_size, eps=args.layer_norm_epsilon, bias=True
        )
        self.self_attention = BloomAttention(args=args)
        self.post_attention_layernorm = nn.LayerNorm(
            self.hidden_size, eps=args.layer_norm_epsilon, bias=True
        )
        self.mlp = BloomMLP(args=args)

    def __call__(
        self,
        hidden_states: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:

        layernorm_output = self.input_layernorm(hidden_states)

        residual = hidden_states

        attention_output = self.self_attention(
            hidden_states=layernorm_output, mask=mask, cache=cache
        )

        attention_output = residual + attention_output

        layernorm_output = self.post_attention_layernorm(attention_output)
        residual = attention_output

        output = self.mlp(layernorm_output) + attention_output

        return output


class BloomModel(nn.Module):

    def __init__(self, args: ModelArgs):
        super().__init__()

        self.embed_dim = args.n_embed
        self.num_heads = args.num_attention_heads

        self.word_embeddings = nn.Embedding(args.vocab_size, self.embed_dim)
        self.word_embeddings_layernorm = nn.LayerNorm(
            self.embed_dim, eps=args.layer_norm_epsilon, bias=True
        )
        self.h = [BloomBlock(args=args) for _ in range(args.n_layer)]
        self.ln_f = nn.LayerNorm(self.embed_dim, eps=args.layer_norm_epsilon)

    def __call__(
        self,
        inputs: mx.array,
        cache: Optional[Any] = None,
    ) -> mx.array:

        input_embed = self.word_embeddings(inputs)

        if cache is None:
            cache = [None] * len(self.h)

        hidden_state = self.word_embeddings_layernorm(input_embed)

        mask = create_attention_mask(h=hidden_state, cache=cache[0], return_array=True)

        for c, block in zip(cache, self.h):
            hidden_state = block(hidden_states=hidden_state, mask=mask, cache=c)

        return self.ln_f(hidden_state)


class Model(nn.Module):

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = BloomModel(args)

    def __call__(
        self,
        inputs: mx.array,
        cache=None,
    ):
        outputs = self.model(inputs, cache)
        outputs = self.model.word_embeddings.as_linear(outputs)
        return outputs

    def sanitize(self, weights):
        new_weights = {}
        for weight in weights:
            if not weight.startswith("model."):
                new_weights[f"model.{weight}"] = weights[weight]
            else:
                new_weights[weight] = weights[weight]
        return new_weights

    @property
    def layers(self):
        return self.model.h
