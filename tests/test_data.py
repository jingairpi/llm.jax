import numpy as np
import pytest

from llm.data import data_loader, load_binary_data


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
    """Tests the data loader for correct batch shapes."""
    batch_size, seq_len = 4, 64
    loader = data_loader(sample_data_file, batch_size, seq_len)

    inputs, targets = next(loader)
    assert inputs.shape == (batch_size, seq_len - 1), "Incorrect input shape"
    assert targets.shape == (batch_size, seq_len - 1), "Incorrect target shape"
