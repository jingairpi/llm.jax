import numpy as np


def load_binary_data(file_path):
    """
    Reads the tokenized dataset from a binary file (e.g., llm.c uses `.bin` files).
    Returns a numpy array of integer token IDs.
    """
    with open(file_path, "rb") as f:
        data = np.fromfile(f, dtype=np.int32)
    return data


class DataLoader:
    """
    A simple DataLoader that reads tokenized text data from a binary file
    and yields (inputs, targets) batches. Inputs are sequences of tokens,
    and targets are the same sequences shifted by one.
    """

    def __init__(self, file_path, batch_size, seq_len):
        """
        Initializes the DataLoader.

        Args:
            file_path (str): Path to the binary data file.
            batch_size (int): Number of sequences per batch.
            seq_len (int): Length of each sequence.
        """
        self.data = load_binary_data(file_path)
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.num_batches = len(self.data) // (batch_size * seq_len)
        self.current_batch = 0

    def __iter__(self):
        """
        Returns the iterator object (self) and resets the batch counter.
        """
        self.current_batch = 0
        return self

    def __next__(self):
        """
        Returns the next (inputs, targets) batch. Each batch is constructed by:
          - taking a slice of the data of length (batch_size * seq_len),
          - reshaping it into (batch_size, seq_len),
          - and then splitting it so that inputs contain all tokens except the last,
            and targets contain all tokens except the first.

        Raises:
            StopIteration: When there are no more batches to yield.
        """
        if self.current_batch >= self.num_batches:
            raise StopIteration

        start = self.current_batch * self.batch_size * self.seq_len
        end = (self.current_batch + 1) * self.batch_size * self.seq_len
        batch = self.data[start:end]
        self.current_batch += 1

        # Reshape to (batch_size, seq_len)
        batch = batch.reshape(self.batch_size, self.seq_len).astype(np.int32)
        # Split into inputs (all tokens except last) and targets (all tokens except first)
        inputs = batch[:, :-1]
        targets = batch[:, 1:]
        return inputs, targets

    def reset(self):
        """
        Resets the internal batch counter so that iteration starts over.
        """
        self.current_batch = 0
