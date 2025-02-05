import math

import jax
import jax.numpy as jnp


class Inferencer:
    """
    A simple inferencer class for autoregressive text generation.
    Given an initial prompt (token IDs) and a number of new tokens to generate,
    it iteratively runs the model to generate text.
    """

    def __init__(self, model, config):
        self.model = model
        self.config = config

    def generate(
        self,
        params,
        input_ids,
        max_new_tokens,
        temperature=1.0,
        top_k=None,
        rng_key=None,
    ):
        """
        Generates new tokens autoregressively.

        Args:
            params: Model parameters.
            input_ids: jnp.ndarray of shape (batch_size, seq_len) containing the initial tokens.
            max_new_tokens: Number of tokens to generate.
            temperature: Temperature for sampling (default 1.0).
            top_k: If provided, only the top_k tokens are considered at each step.
            rng_key: Optional JAX PRNGKey; if None, a default key is used.

        Returns:
            jnp.ndarray of shape (batch_size, seq_len + max_new_tokens)
        """
        if rng_key is None:
            rng_key = jax.random.PRNGKey(0)
        batch_size, seq_len = input_ids.shape

        for _ in range(max_new_tokens):
            # Ensure that the input is within the maximum allowed sequence length.
            if seq_len > self.config.n_positions:
                input_ids = input_ids[:, -self.config.n_positions :]
                seq_len = self.config.n_positions

            # Run the model in deterministic (inference) mode.
            logits = self.model.apply({"params": params}, input_ids, deterministic=True)
            # Get logits for the last token in the sequence.
            logits = logits[:, -1, :]  # shape (batch_size, vocab_size)
            # Adjust by temperature.
            logits = logits / temperature

            # Optionally apply top-k filtering.
            if top_k is not None:
                kth = jnp.sort(logits, axis=-1)[:, -top_k][:, None]
                logits = jnp.where(logits < kth, -1e10, logits)

            # Compute probabilities.
            probs = jax.nn.softmax(logits, axis=-1)
            # Update the random key and sample the next token.
            rng_key, subkey = jax.random.split(rng_key)
            next_token = jax.random.categorical(subkey, logits, axis=-1)
            next_token = next_token[:, None]  # shape (batch_size, 1)
            # Append the new token to the sequence.
            input_ids = jnp.concatenate([input_ids, next_token], axis=1)
            seq_len += 1

        return input_ids
