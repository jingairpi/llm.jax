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
