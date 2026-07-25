from dataclasses import dataclass, field
from typing import Optional, Any

from .base import BaseModelArgs, scaled_dot_product_attention, create_attention_mask

from mlx import nn
import mlx.core as mx

@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str = "gpt_neo"
    hidden_size: int = 2048
    vocab_size: int = 50257
    max_position_embeddings: int = 2048
    layer_norm_epsilon: float = 1e-05
    num_heads: int = 16
    num_layers: int = 24
    attention_layers: list = field(default_factory= list)
    window_size: int = 256


class GPTNeoAttention(nn.Module):

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.embed_dim = args.hidden_size
        self.num_heads = args.num_heads
        self.head_dim = self.embed_dim // self.num_heads 
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:"
                f" {self.num_heads})."
            )

        self.scaling = 1.0  # GPT-Neo does NOT scale attention scores

        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim, bias= False)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim, bias= False)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim, bias= False)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim, bias= True)

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:

        B, L, D = x.shape

        queries = self.q_proj(x)
        keys = self.k_proj(x)
        values = self.v_proj(x)

        queries = queries.reshape(B, L, self.num_heads, -1).transpose(0, 2, 1, 3)
        keys = keys.reshape(B, L, self.num_heads, -1).transpose(0, 2, 1, 3)
        values = values.reshape(B, L, self.num_heads, -1).transpose(0, 2, 1, 3)

        if cache is not None:
            keys, values = cache.update_and_fetch(keys, values)

        output = scaled_dot_product_attention(
            queries, keys, values, cache=cache, scale=self.scaling, mask=mask
        )

        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.out_proj(output)

class GPTNeoMLP(nn.Module):

    def __init__(self, args: ModelArgs):
        super().__init__()
        embed_dim = args.hidden_size

        self.c_fc = nn.Linear(embed_dim, 4*embed_dim , bias= True)
        self.c_proj = nn.Linear(4*embed_dim, embed_dim, bias= True)

    def __call__(self, x: mx.array) -> mx.array:
        hidden_states = self.c_fc(x)
        hidden_states = nn.gelu_approx(hidden_states) # gelu_new
        hidden_states = self.c_proj(hidden_states)
        return hidden_states

class GPTNeoBlock(nn.Module):

    def __init__(self, args: ModelArgs, layer_id: int):
        super().__init__()

        # global local attention
        self.isLocal = args.attention_layers[layer_id] == "local"

        self.ln_1 = nn.LayerNorm(args.hidden_size, eps= args.layer_norm_epsilon)
        self.attn = GPTNeoAttention(args= args)
        self.ln_2 = nn.LayerNorm(args.hidden_size, eps= args.layer_norm_epsilon)
        self.mlp = GPTNeoMLP(args= args)

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:

        residual = x 
        hidden_states = self.ln_1(x)
        attn_output = self.attn(x= hidden_states, mask= mask, cache= cache)

        hidden_states = attn_output + residual

        residual = hidden_states
        hidden_states = self.ln_2(hidden_states)
        feed_forward_hidden_states = self.mlp(hidden_states)

        hidden_states = residual + feed_forward_hidden_states

        return hidden_states

class GPTNeoModel(nn.Module):

    def __init__(self, args: ModelArgs):
        super().__init__()

        self.args = args
        self.embed_dim = args.hidden_size
        self.wte = nn.Embedding(args.vocab_size, self.embed_dim)
        self.wpe = nn.Embedding(args.max_position_embeddings, self.embed_dim)
        self.h = [GPTNeoBlock(args= args, layer_id = layer_id) for layer_id in range(args.num_layers)]
        self.ln_f = nn.LayerNorm(self.embed_dim, eps= args.layer_norm_epsilon)

    def __call__(
        self,
        inputs: mx.array,
        cache: Optional[Any]= None,
    ) -> mx.array:

        batch_size, input_length = inputs.shape
        
        if cache is None:
            cache = [None] * len(self.h)

        offset = 0
        if cache[0] is not None:
            offset = cache[0].offset

        offset = mx.array(offset)
        position_ids = mx.arange(input_length) + offset[..., None]

        inputs_embed = self.wte(inputs)
        position_embed = self.wpe(position_ids)
        hidden_status = inputs_embed + position_embed

        for c, block in zip(cache, self.h):
            if block.isLocal:
                # local attention
                mask = create_attention_mask(
                    hidden_status, c, window_size= self.args.window_size
                )
            else:
                # global attention
                mask = create_attention_mask(
                    hidden_status, c
                )
            hidden_status = block(
                x= hidden_status,
                mask= mask,
                cache= c
            )

        return self.ln_f(hidden_status)

class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = GPTNeoModel(args)

    def __call__(self, inputs, cache=None):

        output = self.model(
            inputs=inputs,
            cache=cache
        )

        logits = self.model.wte.as_linear(output)
        return logits

    def sanitize(self, weights):
        new_weights = {}
        for i in range(self.args.num_layers):
            if f"transformer.h.{i}.attn.attention.bias" in weights:
                del weights[f"transformer.h.{i}.attn.attention.bias"]
            if f"transformer.h.{i}.attn.attention.masked_bias" in weights:
                del weights[f"transformer.h.{i}.attn.attention.masked_bias"]
        for weight in weights:
            if not weight.startswith("model."):
                new_weights[f"model.{weight.replace('transformer.', '').replace('.attn.attention.', '.attn.')}"] = weights[weight]
            else:
                new_weights[weight] = weights[weight]
        return new_weights

    @property
    def layers(self):
        return self.model.h