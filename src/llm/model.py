import jax
import jax.numpy as jnp
from flax import linen as nn

from .config import GPT2Config


class CausalSelfAttention(nn.Module):
    config: GPT2Config

    @nn.compact
    def __call__(self, x, deterministic=True):
        B, T, C = x.shape
        head_dim = C // self.config.n_head
        # Project x into query, key, and value (concatenated)
        qkv = nn.Dense(3 * C)(x)
        # Reshape to (B, T, 3, n_head, head_dim)
        qkv = qkv.reshape(B, T, 3, self.config.n_head, head_dim)
        # Split into q, k, and v; each becomes (B, T, n_head, head_dim)
        q, k, v = jnp.split(qkv, 3, axis=2)
        q = jnp.squeeze(q, axis=2)
        k = jnp.squeeze(k, axis=2)
        v = jnp.squeeze(v, axis=2)

        # Compute scaled dot-product attention
        scale = 1.0 / jnp.sqrt(head_dim)
        att = jnp.einsum("bthd,bThd->bhtT", q, k) * scale

        # Create lower-triangular mask to prevent attending to future tokens
        mask = jnp.tril(jnp.ones((T, T)))
        mask = mask.reshape(1, 1, T, T)
        att = att * mask - 1e10 * (1 - mask)
        att = nn.softmax(att, axis=-1)
        att = nn.Dropout(0.1)(att, deterministic=deterministic)

        # Compute the attention output
        y = jnp.einsum("bhtT,bThd->bthd", att, v)
        y = y.reshape(B, T, C)
        y = nn.Dense(C)(y)
        y = nn.Dropout(0.1)(y, deterministic=deterministic)
        return y


class MLP(nn.Module):
    config: GPT2Config

    @nn.compact
    def __call__(self, x, deterministic=True):
        hidden = nn.Dense(4 * self.config.n_embd)(x)
        hidden = nn.gelu(hidden)
        hidden = nn.Dense(self.config.n_embd)(hidden)
        hidden = nn.Dropout(0.1)(hidden, deterministic=deterministic)
        return hidden


class Block(nn.Module):
    config: GPT2Config

    @nn.compact
    def __call__(self, x, deterministic=True):
        # Self-Attention sub-block with residual connection
        ln1 = nn.LayerNorm()(x)
        attn_out = CausalSelfAttention(self.config)(ln1, deterministic=deterministic)
        x = x + attn_out
        # Feed-forward (MLP) sub-block with residual connection
        ln2 = nn.LayerNorm()(x)
        mlp_out = MLP(self.config)(ln2, deterministic=deterministic)
        x = x + mlp_out
        return x


class GPT2Model(nn.Module):
    config: GPT2Config

    @nn.compact
    def __call__(self, input_ids, deterministic=True):
        B, T = input_ids.shape
        # Token embeddings
        wte = nn.Embed(
            num_embeddings=self.config.vocab_size,
            features=self.config.n_embd,
            name="wte",
        )
        # Learnable positional embeddings: shape (n_positions, n_embd)
        wpe = self.param(
            "wpe",
            nn.initializers.normal(stddev=0.02),
            (self.config.n_positions, self.config.n_embd),
        )
        x = wte(input_ids)  # (B, T, n_embd)
        x = x + wpe[:T, :]  # add positional embeddings

        # Pass through transformer blocks
        for _ in range(self.config.n_layer):
            x = Block(self.config)(x, deterministic=deterministic)
        x = nn.LayerNorm()(x)

        # Output projection using weight tying with token embeddings
        logits = x @ wte.embedding.T  # (B, T, vocab_size)
        return logits
