import os

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from llm.config import GPT2Config
from llm.data import DataLoader
from llm.evaluator import Evaluator
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
def dummy_data_file(tmp_path):
    """
    Create a temporary binary file with dummy tokenized data.
    For example, if batch_size=2 and seq_len=32, we need at least 2*32 tokens.
    Here we generate 128 tokens, ensuring that they fall within the vocabulary range.
    """
    file_path = tmp_path / "dummy.bin"
    # Ensure tokens are in the range [0, 100) since our config vocab_size is 100.
    data = (np.arange(128) % 100).astype(np.int32)
    data.tofile(file_path)
    return str(file_path)


@pytest.fixture
def dummy_data_loader(dummy_data_file, config):
    # Create a DataLoader that does not repeat (for evaluation, one epoch is enough).
    # (DataLoader yields batches where inputs are shape (batch_size, seq_len-1) and targets are the same.)
    return DataLoader(dummy_data_file, batch_size=2, seq_len=32, repeat=False)


def test_evaluator_loss(model, config, dummy_data_loader):
    """
    Test that Evaluator.evaluate returns a scalar loss (non-negative float)
    when running a couple of evaluation steps.
    """
    evaluator = Evaluator(model, dummy_data_loader)
    rng = jax.random.PRNGKey(0)
    # Initialize model parameters with a dummy input of shape (batch_size, seq_len-1).
    dummy_input = jnp.ones((2, 31), dtype=jnp.int32)
    params = model.init(rng, dummy_input)["params"]

    avg_loss = evaluator.evaluate(params, num_steps=2)

    # Check that the loss is a float and non-negative.
    assert isinstance(avg_loss, float), "Expected loss to be a float."
    assert avg_loss >= 0.0, "Expected loss to be non-negative."
