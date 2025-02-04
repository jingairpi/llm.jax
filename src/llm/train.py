import time

import jax
import jax.numpy as jnp
import optax
from flax.training import train_state

from .config import GPT2Config
from .data import DataLoader
from .model import GPT2Model


def compute_loss(params, apply_fn, batch, rng):
    """
    Computes the cross-entropy loss between model predictions and targets.
    Returns the loss and the logits for further inspection.
    """
    inputs, targets = batch
    logits = apply_fn(
        {"params": params}, inputs, deterministic=False, rngs={"dropout": rng}
    )
    loss = optax.softmax_cross_entropy_with_integer_labels(logits, targets)
    return loss.mean(), logits


@jax.jit
def train_step(state, batch, rng):
    """
    A single training step that computes loss, gradients, and applies an update.
    """
    grad_fn = jax.value_and_grad(compute_loss, has_aux=True)
    (loss, logits), grads = grad_fn(state.params, state.apply_fn, batch, rng)
    new_state = state.apply_gradients(grads=grads)
    return new_state, loss, logits


def train(num_steps=10, batch_size=4, seq_len=64, dataset_path="data/gpt2_train.bin"):
    """
    Runs a training loop for a specified number of steps using data loaded
    from a binary file. Prints the loss and timing information for each step.
    """
    # Initialize configuration and model.
    config = GPT2Config(
        vocab_size=50257, n_positions=1024, n_embd=768, n_layer=12, n_head=12
    )
    model = GPT2Model(config)

    # NOTE: The DataLoader yields batches where each batch is created from a
    # segment of length `seq_len` and then split so that:
    #   - inputs have shape (batch_size, seq_len - 1)
    #   - targets have shape (batch_size, seq_len - 1)
    # Therefore, when initializing the model parameters, we mimic that input shape.
    rng = jax.random.PRNGKey(0)
    dummy_input = jnp.ones((batch_size, seq_len), dtype=jnp.int32)[:, :-1]
    params = model.init(rng, dummy_input)["params"]

    # Set up the optimizer and training state.
    tx = optax.adamw(learning_rate=1e-4)
    state = train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)

    # Create the DataLoader instance.
    loader = DataLoader(dataset_path, batch_size, seq_len)

    step = 0
    for batch in loader:
        if step >= num_steps:
            break
        rng, step_rng = jax.random.split(rng)
        start_time = time.time()
        state, loss, logits = train_step(state, batch, step_rng)
        elapsed = time.time() - start_time
        print(f"Step {step}: Loss = {loss:.4f}, Time = {elapsed*1000:.2f} ms")
        step += 1

    # Final validation: run a forward pass in deterministic mode.
    # Restart the data loader iterator for sampling.
    sample_loader = DataLoader(dataset_path, batch_size, seq_len)
    sample_inputs, _ = next(iter(sample_loader))
    final_logits = model.apply(
        {"params": state.params}, sample_inputs, deterministic=True
    )
    print("Final logits shape:", final_logits.shape)
