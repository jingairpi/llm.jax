import numpy as np


def load_binary_data(file_path):
    """
    Reads the tokenized dataset from a binary file (e.g., llm.c uses `.bin` files).
    Returns a numpy array of integer token IDs.
    """
    with open(file_path, "rb") as f:
        data = np.fromfile(f, dtype=np.int32)
    return data


def data_loader(file_path, batch_size, seq_len):
    """
    Loads tokenized text data from a binary file and returns (inputs, targets) batches.
    The input is a sequence of tokens; targets are the same sequence shifted by one.
    """
    data = load_binary_data(file_path)  # Read raw token IDs
    num_batches = len(data) // (batch_size * seq_len)

    for i in range(num_batches):
        batch = data[i * batch_size * seq_len : (i + 1) * batch_size * seq_len]
        batch = batch.reshape(batch_size, seq_len).astype(np.int32)
        inputs = batch[:, :-1]  # All tokens except the last one
        targets = batch[:, 1:]  # Shifted one token to the right
        yield inputs, targets
