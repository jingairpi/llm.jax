import numpy as np


def load_binary_data(file_path):
    """
    Reads the tokenized dataset from a binary file.
    Returns a numpy array of integer token IDs.
    """
    with open(file_path, "rb") as f:
        data = np.fromfile(f, dtype=np.int32)
    return data


class DataLoader:
    """
    A simple DataLoader that reads tokenized text data from a binary file
    and yields (inputs, targets) batches. One complete pass through the data
    represents one epoch. If repeat is True, the loader cycles indefinitely.
    """

    def __init__(self, file_path, batch_size, seq_len, repeat=False):
        self.data = load_binary_data(file_path)
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.repeat = repeat
        self.num_batches = len(self.data) // (batch_size * seq_len)
        self.current_batch = 0

    def __iter__(self):
        self.current_batch = 0
        return self

    def __next__(self):
        if self.current_batch >= self.num_batches:
            if self.repeat:
                self.reset()
            else:
                raise StopIteration
        start = self.current_batch * self.batch_size * self.seq_len
        end = (self.current_batch + 1) * self.batch_size * self.seq_len
        batch = self.data[start:end]
        self.current_batch += 1
        batch = batch.reshape(self.batch_size, self.seq_len).astype(np.int32)
        # For example, define inputs as all tokens except the last and targets as all tokens except the first.
        inputs = batch[:, :-1]
        targets = batch[:, 1:]
        return inputs, targets

    def reset(self):
        """
        Resets the DataLoader's internal batch counter so that iteration starts from the beginning.
        """
        self.current_batch = 0
