class GPT2Config:
    def __init__(
        self,
        vocab_size=50257,
        padded_vocab_size=50304,
        n_positions=1024,
        n_embd=768,
        n_layer=12,
        n_head=12,
    ):
        self.vocab_size = vocab_size  # vocabulary size
        self.padded_vocab_size = (
            padded_vocab_size  # padded to a multiple of 128 (e.g., 50304)
        )
        self.n_positions = n_positions  # maximum sequence length
        self.n_embd = n_embd  # embedding dimension (channels)
        self.n_layer = n_layer  # number of transformer blocks
        self.n_head = n_head  # number of attention heads
