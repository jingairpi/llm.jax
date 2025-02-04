# tests/test_model.py

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from transformers import FlaxGPT2LMHeadModel
from transformers import GPT2Config as HFConfig

from llm.config import GPT2Config
from llm.model import GPT2Model


def convert_hf_to_my_params(hf_params: dict, config: GPT2Config) -> dict:
    """
    Convert Hugging Face FlaxGPT2LMHeadModel parameters into the structure
    expected by our GPT2Model.

    The HF model stores the weights for the Dense projections (for attention and the MLP)
    in a “Conv1D” style (with kernel shape (out_features, in_features)), but our model’s
    nn.Dense expects kernels of shape (in_features, out_features). Hence we transpose
    the four weight matrices in: attn.c_attn, attn.c_proj, mlp.c_fc, and mlp.c_proj.
    """
    new_params = {}
    # Copy token and position embeddings.
    new_params["wte"] = hf_params["transformer"]["wte"]
    new_params["wpe"] = hf_params["transformer"]["wpe"]

    # Process each transformer block.
    for i in range(config.n_layer):
        # HF stores transformer blocks under transformer.h with string keys.
        block = hf_params["transformer"]["h"][str(i)].copy()

        # --- Process the attention submodule ---
        if "attn" in block:
            attn = block["attn"].copy()
            # For the combined QKV projection (c_attn): transpose the kernel.
            if "c_attn" in attn:
                c_attn = attn["c_attn"].copy()
                if "kernel" in c_attn:
                    # HF kernel shape is (3 * n_embd, n_embd); we want (n_embd, 3 * n_embd)
                    c_attn["kernel"] = c_attn["kernel"].T
                attn["c_attn"] = c_attn
            # For the output projection (c_proj): transpose the kernel.
            if "c_proj" in attn:
                c_proj = attn["c_proj"].copy()
                if "kernel" in c_proj:
                    c_proj["kernel"] = c_proj["kernel"].T
                attn["c_proj"] = c_proj
            block["attn"] = attn

        # --- Process the MLP submodule ---
        if "mlp" in block:
            mlp = block["mlp"].copy()
            # For the first MLP dense (c_fc): transpose the kernel.
            if "c_fc" in mlp:
                c_fc = mlp["c_fc"].copy()
                if "kernel" in c_fc:
                    c_fc["kernel"] = c_fc["kernel"].T
                mlp["c_fc"] = c_fc
            # For the second MLP dense (c_proj): transpose the kernel.
            if "c_proj" in mlp:
                c_proj = mlp["c_proj"].copy()
                if "kernel" in c_proj:
                    c_proj["kernel"] = c_proj["kernel"].T
                mlp["c_proj"] = c_proj
            block["mlp"] = mlp

        # Place the converted block under our naming scheme.
        new_params[f"h.{i}"] = block

    # Copy the final layer norm.
    new_params["ln_f"] = hf_params["transformer"]["ln_f"]

    return {"params": new_params}


def test_forward_shape():
    """
    Test that our GPT2Model produces logits with the correct shape.
    """
    # Use a small configuration for testing.
    config = GPT2Config(
        vocab_size=100,
        padded_vocab_size=100,  # for simplicity, no extra padding
        n_positions=20,
        n_embd=16,
        n_layer=2,
        n_head=2,
    )
    model = GPT2Model(config)
    batch_size = 1
    seq_length = 10
    dummy_input = jnp.ones((batch_size, seq_length), dtype=jnp.int32)

    # Initialize model parameters with a fixed PRNG key.
    rng = jax.random.PRNGKey(0)
    params = model.init(rng, dummy_input)

    # Compute logits.
    logits = model.apply(params, dummy_input, deterministic=True)
    expected_shape = (batch_size, seq_length, config.vocab_size)
    assert (
        logits.shape == expected_shape
    ), f"Expected logits shape {expected_shape}, but got {logits.shape}"


def test_compare_with_hf():
    """
    Compare our GPT2Model (with parameters converted from HF's FlaxGPT2LMHeadModel)
    to the HF model's output. For an identical small configuration and fixed input,
    the outputs should nearly match.
    """
    # Define a small configuration.
    vocab_size = 100
    config = GPT2Config(
        vocab_size=vocab_size,
        padded_vocab_size=vocab_size,
        n_positions=20,
        n_embd=16,
        n_layer=2,
        n_head=2,
    )
    my_model = GPT2Model(config)
    # Create a dummy input (e.g. a sequence of integers 0 to 9).
    dummy_input = jnp.arange(10)[None, :].astype(jnp.int32)
    rng = jax.random.PRNGKey(0)
    _ = my_model.init(rng, dummy_input)  # initialize (parameters will be replaced)

    # Create an HF configuration that matches.
    hf_config = HFConfig(
        vocab_size=vocab_size,
        n_positions=20,
        n_ctx=20,  # context length, same as n_positions
        n_embd=16,
        n_layer=2,
        n_head=2,
    )
    hf_model = FlaxGPT2LMHeadModel(hf_config, seed=0, dtype=jnp.float32)
    hf_params = hf_model.params

    # Convert HF parameters to the structure expected by our model.
    converted_params = convert_hf_to_my_params(hf_params, config)

    # Compute outputs from our model using the converted parameters.
    my_logits = my_model.apply(converted_params, dummy_input, deterministic=True)

    # Compute outputs from the HF model.
    hf_outputs = hf_model(dummy_input, params=hf_params)
    hf_logits = hf_outputs.logits

    # Ensure that the shapes agree.
    assert (
        my_logits.shape == hf_logits.shape
    ), f"Output shapes differ: {my_logits.shape} vs {hf_logits.shape}"

    # Compare the numerical values (they should match very closely).
    np.testing.assert_allclose(
        np.array(my_logits),
        np.array(hf_logits),
        rtol=1e-5,
        atol=1e-5,
        err_msg="The logits from the custom model and HF model do not match.",
    )
