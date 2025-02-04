import re

import jax
import numpy as np
import pytest

from llm.config import GPT2Config
from llm.data import DataLoader
from llm.model import GPT2Model
from llm.trainer import Trainer


@pytest.fixture
def sample_train_file(tmp_path):
    """
    Creates a temporary binary file with fake tokenized data.
    For example, if batch_size=2 and seq_len=16, each batch needs 32 tokens.
    Here we create 128 tokens (4 full batches).
    """
    file_path = tmp_path / "sample_train.bin"
    data = np.arange(128, dtype=np.int32)
    data.tofile(file_path)
    return str(file_path)


@pytest.fixture
def trainer(sample_train_file):
    """
    Instantiates a Trainer object using a small configuration, model, and DataLoader.
    """
    config = GPT2Config(
        vocab_size=50257,  # usual vocab size
        n_positions=64,  # shorter positions for testing
        n_embd=32,  # smaller embedding dimension
        n_layer=1,  # one transformer block
        n_head=2,  # two attention heads
    )
    model = GPT2Model(config)
    optimizer_config = {
        "learning_rate": 1e-3,
        "batch_size": 2,  # small batch size for testing.
    }
    batch_size = optimizer_config["batch_size"]
    seq_len = 16
    data_loader_instance = DataLoader(sample_train_file, batch_size, seq_len)
    rng = jax.random.PRNGKey(0)
    return Trainer(model, config, optimizer_config, data_loader_instance, rng)


def test_trainer_runs(trainer, capsys):
    """
    Runs the training loop for a few steps using the Trainer.
    Verifies that training steps and final diagnostics are printed in the expected format.
    """
    num_steps = 3
    trainer.train(num_steps)

    captured = capsys.readouterr().out
    # Basic check for training step output.
    assert "Step" in captured, "Expected training step output not found."

    # Use regex to verify each printed training step format.
    step_pattern = re.compile(r"Step (\d+): Loss = ([\d\.]+), Time = ([\d\.]+) ms")
    step_matches = step_pattern.findall(captured)
    assert (
        len(step_matches) >= num_steps
    ), f"Expected at least {num_steps} training steps, but found {len(step_matches)}."
    for step_str, loss_str, time_str in step_matches:
        step_int = int(step_str)
        loss_val = float(loss_str)
        time_val = float(time_str)
        assert step_int >= 0, "Step number should be non-negative."
        assert loss_val >= 0, "Loss should be non-negative."
        assert time_val > 0, "Elapsed time should be positive."

    # Check for a summary message.
    summary_pattern = re.compile(r"Training complete", re.IGNORECASE)
    assert (
        summary_pattern.search(captured) is not None
    ), "Expected training completion message not found."
