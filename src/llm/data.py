import numpy as np


def data_loader(file_path, batch_size, seq_len):
    """
    Loads tokenized text data from a NumPy file (a 1D array of token ids)
    and returns batches of (inputs, targets) for language modeling.
    Inputs are the sequence excluding the last token; targets are the sequence shifted by one.
    """
    data = np.load(file_path)  # Expecting a 1D numpy array of token IDs
    num_batches = len(data) // (batch_size * seq_len)
    for i in range(num_batches):
        batch = data[i * batch_size * seq_len : (i + 1) * batch_size * seq_len]
        batch = batch.reshape(batch_size, seq_len).astype(np.int32)
        inputs = batch[:, :-1]
        targets = batch[:, 1:]
        yield inputs, targets


def dummy_data_loader(num_batches, batch_size, seq_len, vocab_size):
    """
    Generates dummy batches for language modeling.
    Each batch is a tuple (inputs, targets) where:
      - inputs: shape (batch_size, seq_len - 1)
      - targets: shape (batch_size, seq_len - 1), shifted by one token.
    """
    for _ in range(num_batches):
        batch = np.random.randint(0, vocab_size, size=(batch_size, seq_len)).astype(
            np.int32
        )
        inputs = batch[:, :-1]
        targets = batch[:, 1:]
        yield inputs, targets
