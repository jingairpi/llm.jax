import time

import jax
import jax.numpy as jnp
import optax
from flax.training import train_state


# Define a top-level, pure training step function.
def train_step_fn(state, batch, rng, model, compute_loss_fn):
    """
    Performs a single training step given the current state, a batch, and a PRNG key.
    Uses the provided model and loss function.
    """
    grad_fn = jax.value_and_grad(compute_loss_fn, has_aux=True)
    (loss, logits), grads = grad_fn(state.params, batch, rng)
    new_state = state.apply_gradients(grads=grads)
    return new_state, loss, logits


# JIT compile the training step function.
# We mark `model` and `compute_loss_fn` as static arguments since they are not arrays.
train_step_fn = jax.jit(train_step_fn, static_argnums=(3, 4))


class Trainer:
    def __init__(self, model, config, optimizer_config, dataset, rng):
        self.model = model
        self.config = config
        self.dataset = dataset  # Assume this is an iterable (e.g. a DataLoader)
        self.rng = rng

        # Initialize model parameters using a dummy input.
        dummy_input = jnp.ones(
            (optimizer_config["batch_size"], config.n_positions - 1), dtype=jnp.int32
        )
        params = model.init(rng, dummy_input)["params"]

        # Set up optimizer and training state.
        tx = optax.adamw(learning_rate=optimizer_config["learning_rate"])
        self.state = train_state.TrainState.create(
            apply_fn=model.apply, params=params, tx=tx
        )

    def compute_loss(self, params, batch, rng):
        inputs, targets = batch
        logits = self.model.apply(
            {"params": params}, inputs, deterministic=False, rngs={"dropout": rng}
        )
        loss = optax.softmax_cross_entropy_with_integer_labels(logits, targets)
        return loss.mean(), logits

    def train_step(self, state, batch, rng):
        return train_step_fn(state, batch, rng, self.model, self.compute_loss)

    def train(self, num_steps):
        step = 0
        for batch in self.dataset:
            if step >= num_steps:
                break
            self.rng, step_rng = jax.random.split(self.rng)
            start_time = time.time()
            # Call the jitted top-level function.
            self.state, loss, logits = train_step_fn(
                self.state, batch, step_rng, self.model, self.compute_loss
            )
            elapsed = time.time() - start_time
            print(f"Step {step}: Loss = {loss:.4f}, Time = {elapsed*1000:.2f} ms")
            step += 1
        print("Training complete.")
        return self.state
