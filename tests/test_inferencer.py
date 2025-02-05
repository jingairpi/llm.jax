import jax
import jax.numpy as jnp
import pytest

from llm.config import GPT2Config
from llm.inferencer import Inferencer
from llm.model import GPT2Model


# Use a minimal configuration for testing.
@pytest.fixture
def config():
    return GPT2Config(
        vocab_size=100,  # small vocab for testing
        n_positions=32,  # small sequence length
        n_embd=16,  # small embedding dimension
        n_layer=1,  # single layer
        n_head=2,  # two attention heads
    )


@pytest.fixture
def model(config):
    return GPT2Model(config)


@pytest.fixture
def params(model, config):
    rng = jax.random.PRNGKey(0)
    # Use a dummy input: shape (1, n_positions-1) so that inputs shape is (1, 31)
    dummy_input = jnp.ones((1, config.n_positions - 1), dtype=jnp.int32)
    return model.init(rng, dummy_input)["params"]


def test_inferencer_output_shape(model, config, params):
    """
    Test that Inferencer.generate produces an output sequence with the expected shape.
    For an initial prompt with shape (1, 1) and max_new_tokens=5,
    the output should have shape (1, 6).
    """
    inferencer = Inferencer(model, config)
    initial_prompt = jnp.array([[1]], dtype=jnp.int32)  # a single token prompt
    generated_ids = inferencer.generate(
        params,
        initial_prompt,
        max_new_tokens=5,
        temperature=1.0,
        top_k=10,
        rng_key=jax.random.PRNGKey(42),
    )
    expected_shape = (1, 1 + 5)  # (1, 6)
    assert (
        generated_ids.shape == expected_shape
    ), f"Expected generated shape {expected_shape}, got {generated_ids.shape}"
    # Optionally check that the generated IDs are valid integers in the range [0, vocab_size).
    assert (
        generated_ids.min() >= 0 and generated_ids.max() < config.vocab_size
    ), "Generated token IDs are outside the expected vocabulary range."
