import math

import jax
import jax.numpy as jnp
from flax import linen as nn

from .config import GPT2Config


class NewGELU(nn.Module):
    """
    GELU activation function with the exact GPT-2 approximation.
    """

    def __call__(self, x: jax.Array) -> jax.Array:
        return (
            0.5
            * x
            * (
                1.0
                + jnp.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * jnp.power(x, 3)))
            )
        )


class CausalSelfAttention(nn.Module):
    """
    Multi-head causal self-attention mechanism.
    """

    config: GPT2Config

    @nn.compact
    def __call__(self, x: jax.Array, deterministic: bool = True) -> jax.Array:
        B, T, C = x.shape  # Batch, Time, Channels
        n_head = self.config.n_head
        head_dim = C // n_head
        scale = 1.0 / jnp.sqrt(head_dim)

        # Combined projection for Q, K, V
        qkv = nn.Dense(
            3 * C,
            name="c_attn",
            kernel_init=nn.initializers.normal(stddev=0.02),
        )(
            x
        )  # (B, T, 3 * C)

        # Reshape and split into query, key, value: (B, T, 3, n_head, head_dim)
        qkv = qkv.reshape(B, T, 3, n_head, head_dim)
        q, k, v = jnp.split(qkv, 3, axis=2)
        q = jnp.squeeze(q, axis=2)  # (B, T, n_head, head_dim)
        k = jnp.squeeze(k, axis=2)
        v = jnp.squeeze(v, axis=2)

        # Compute attention scores
        att = jnp.einsum("bthd,bThd->bhtT", q, k) * scale  # (B, n_head, T, T)

        # Create causal mask and apply it
        mask = jnp.tril(jnp.ones((T, T))).reshape(1, 1, T, T)
        att = att * mask - 1e10 * (1 - mask)
        att = nn.softmax(att, axis=-1)

        # Weighted sum of values
        y = jnp.einsum("bhtT,bThd->bthd", att, v)  # (B, T, n_head, head_dim)
        y = y.reshape(B, T, C)  # Recombine heads

        # Output projection with scaled initialization
        scale_init = 0.02 / math.sqrt(2 * self.config.n_layer)
        y = nn.Dense(
            C,
            name="c_proj",
            kernel_init=nn.initializers.normal(stddev=scale_init),
        )(y)
        return y


class MLP(nn.Module):
    """
    GPT-2 MLP block.
    """

    config: GPT2Config

    @nn.compact
    def __call__(self, x: jax.Array, deterministic: bool = True) -> jax.Array:
        # First dense layer
        h = nn.Dense(
            4 * self.config.n_embd,
            name="c_fc",
            kernel_init=nn.initializers.normal(stddev=0.02),
        )(x)
        h = NewGELU()(h)
        # Second dense layer with scaled initialization
        scale_init = 0.02 / math.sqrt(2 * self.config.n_layer)
        h = nn.Dense(
            self.config.n_embd,
            name="c_proj",
            kernel_init=nn.initializers.normal(stddev=scale_init),
        )(h)
        return h


class Block(nn.Module):
    """
    A single transformer block with pre–layer normalization.
    """

    config: GPT2Config

    @nn.compact
    def __call__(self, x: jax.Array, deterministic: bool = True) -> jax.Array:
        # Self-attention sub-block
        sa_input = nn.LayerNorm(epsilon=1e-5, name="ln_1")(x)
        sa_output = CausalSelfAttention(self.config, name="attn")(
            sa_input, deterministic
        )
        x = x + sa_output

        # MLP sub-block
        mlp_input = nn.LayerNorm(epsilon=1e-5, name="ln_2")(x)
        mlp_output = MLP(self.config, name="mlp")(mlp_input, deterministic)
        x = x + mlp_output
        return x


class GPT2Model(nn.Module):
    """
    Full GPT-2 model with token & positional embeddings, transformer blocks, and output projection.
    """

    config: GPT2Config

    @nn.compact
    def __call__(self, input_ids: jax.Array, deterministic: bool = True) -> jax.Array:
        B, T = input_ids.shape
        C = self.config.n_embd

        # Token embeddings (with padded vocabulary size)
        wte = nn.Embed(
            num_embeddings=self.config.padded_vocab_size,
            features=C,
            name="wte",
            embedding_init=nn.initializers.normal(stddev=0.02),
        )
        # Positional embeddings as an embedding layer (to mirror PyTorch)
        wpe = nn.Embed(
            num_embeddings=self.config.n_positions,
            features=C,
            name="wpe",
            embedding_init=nn.initializers.normal(stddev=0.02),
        )
        tok_emb = wte(input_ids)  # (B, T, C)
        pos_ids = jnp.arange(T)[None, :]  # shape (1, T)
        pos_emb = wpe(pos_ids)  # (1, T, C)
        x = tok_emb + pos_emb  # broadcasting over batch

        # Transformer blocks
        for i in range(self.config.n_layer):
            x = Block(self.config, name=f"h.{i}")(x, deterministic=deterministic)

        # Final layer norm
        x = nn.LayerNorm(epsilon=1e-5, name="ln_f")(x)

        # Output projection with weight tying: use the first `vocab_size` rows of wte.embedding
        logits = x @ wte.embedding[: self.config.vocab_size, :].T
        return logits
