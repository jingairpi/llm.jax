# llm/evaluator.py

import jax
import jax.numpy as jnp
import optax


class Evaluator:
    """
    A simple evaluator that computes the average cross-entropy loss
    on an evaluation dataset. Assumes that the data loader yields
    (inputs, targets) batches.
    """

    def __init__(self, model, data_loader):
        self.model = model
        self.data_loader = data_loader

    def compute_loss(self, params, batch):
        inputs, targets = batch
        # Run the model in deterministic (eval) mode.
        logits = self.model.apply({"params": params}, inputs, deterministic=True)
        loss = optax.softmax_cross_entropy_with_integer_labels(logits, targets)
        return loss.mean(), logits

    def evaluate(self, params, num_steps=10):
        total_loss = 0.0
        steps = 0
        # Loop over the evaluation dataset for a fixed number of steps.
        for batch in self.data_loader:
            loss, _ = self.compute_loss(params, batch)
            total_loss += loss
            steps += 1
            if steps >= num_steps:
                break
        avg_loss = total_loss / steps if steps > 0 else 0.0
        avg_loss = float(avg_loss)
        return avg_loss
