import numpy as np
import pytest

from llm.data import DataLoader, load_binary_data


@pytest.fixture
def sample_data_file(tmp_path):
    """Creates a temporary binary file with sample tokenized data."""
    file_path = tmp_path / "sample_data.bin"
    data = np.arange(1000, dtype=np.int32)  # Fake tokenized dataset
    data.tofile(file_path)
    return str(file_path)


def test_load_binary_data(sample_data_file):
    """Ensures binary data loads correctly from file."""
    data = load_binary_data(sample_data_file)
    assert isinstance(data, np.ndarray), "Data should be a NumPy array"
    assert data.dtype == np.int32, "Data type should be int32"


def test_data_loader(sample_data_file):
    """Tests the DataLoader for correct batch shapes."""
    batch_size, seq_len = 4, 64
    loader = DataLoader(sample_data_file, batch_size, seq_len)

    # Get the first batch
    inputs, targets = next(iter(loader))
    expected_shape = (batch_size, seq_len - 1)
    assert (
        inputs.shape == expected_shape
    ), f"Incorrect input shape: expected {expected_shape}, got {inputs.shape}"
    assert (
        targets.shape == expected_shape
    ), f"Incorrect target shape: expected {expected_shape}, got {targets.shape}"

    # Verify that the targets are shifted versions of the inputs.
    # (For example, if a batch row is [0, 1, 2, ... 62] then targets should be [1, 2, ... 63])
    np.testing.assert_array_equal(inputs[0] + 1, targets[0])


def test_dataloader_reset(sample_data_file):
    """
    Test that the DataLoader correctly resets its internal counter.
    We will iterate through one epoch, record the first batch,
    call reset(), and then confirm that the next first batch is the same.
    """
    batch_size = 4
    seq_len = 16  # Each batch will have shape (batch_size, 16) then split to inputs and targets.

    # Instantiate the DataLoader with repeat=False (so it stops after one epoch).
    loader = DataLoader(sample_data_file, batch_size, seq_len, repeat=False)

    # Convert the loader to an iterator.
    it = iter(loader)

    # Get the first batch.
    first_batch = next(it)

    # Consume the remaining batches (simulate one full epoch).
    batches = [first_batch]
    for batch in it:
        batches.append(batch)

    # Now reset the DataLoader.
    loader.reset()

    # Get the first batch after reset.
    reset_first_batch = next(iter(loader))

    # The first batch before and after reset should be identical.
    # We'll compare both the inputs and targets.
    first_inputs, first_targets = first_batch
    reset_inputs, reset_targets = reset_first_batch

    np.testing.assert_array_equal(
        first_inputs,
        reset_inputs,
        err_msg="Inputs of the first batch after reset do not match the original.",
    )
    np.testing.assert_array_equal(
        first_targets,
        reset_targets,
        err_msg="Targets of the first batch after reset do not match the original.",
    )
