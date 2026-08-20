import numpy as np
import jax
import jax.numpy as jnp
from jax import lax
from jax.flatten_util import ravel_pytree

from Model_Interface import ModelInterface

"""
File: Model_RNN.py
Minimal recurrent baseline that implements the same interface as classical models.

The model learns trial-wise action probabilities from action/reward history:
- Input at trial t: previous action (one-hot) + previous reward
- Recurrent state update: h_t = tanh(Wxh x_t + Whh h_{t-1} + b_h)
- Policy readout: softmax(Wyh h_t + b_y)

"""


class SimpleRNNModel(ModelInterface):
    def __init__(
        self,
        task_properties: dict,
        hidden_size: int = 16,
        learning_rate: float = 0.05,
        n_epochs: int = 60,
        seed: int = 0,
    ):
        super().__init__(task_properties)
        self.hidden_size = int(hidden_size)
        self.learning_rate = float(learning_rate)
        self.n_epochs = int(n_epochs)
        self.seed = int(seed)
        self.input_size = int(self.n_actions + 1)  # previous action one-hot + previous reward

        self.params = self._initialize_parameters()
        self.loss_history = []
        self.hidden_history = None

    def _initialize_parameters(self):
        rng = np.random.default_rng(self.seed)
        w_scale = 0.1
        return {
            "Wxh": jnp.array(rng.normal(scale=w_scale, size=(self.hidden_size, self.input_size))),
            "Whh": jnp.array(rng.normal(scale=w_scale, size=(self.hidden_size, self.hidden_size))),
            "bh": jnp.zeros((self.hidden_size,)),
            "Wyh": jnp.array(rng.normal(scale=w_scale, size=(self.n_actions, self.hidden_size))),
            "by": jnp.zeros((self.n_actions,)),
        }

    def _sequence_to_inputs(self, actions: np.ndarray, rewards: np.ndarray) -> jnp.ndarray:
        """Build recurrent inputs where each trial sees previous action/reward."""
        actions = np.asarray(actions, dtype=int)
        rewards = np.asarray(rewards, dtype=float)

        n_trials = len(actions)
        x = np.zeros((n_trials, self.input_size), dtype=float)

        for t in range(1, n_trials):
            prev_action = int(actions[t - 1])
            if 0 <= prev_action < self.n_actions:
                x[t, prev_action] = 1.0
            x[t, -1] = rewards[t - 1]

        return jnp.array(x)

    def _forward_logits(self, params, x_seq: jnp.ndarray):
        """Run the recurrent pass and return logits and hidden states."""

        def step_fn(h_prev, x_t):
            h_t = jnp.tanh(params["Wxh"] @ x_t + params["Whh"] @ h_prev + params["bh"])
            logits_t = params["Wyh"] @ h_t + params["by"]
            return h_t, (logits_t, h_t)

        h0 = jnp.zeros((self.hidden_size,))
        _, (logits_seq, hidden_seq) = lax.scan(step_fn, h0, x_seq)
        return logits_seq, hidden_seq

    def _sequence_nll(self, params, x_seq: jnp.ndarray, y_seq: jnp.ndarray):
        logits_seq, _ = self._forward_logits(params, x_seq)
        log_probs = jax.nn.log_softmax(logits_seq, axis=1)
        idx = jnp.arange(y_seq.shape[0])
        return -jnp.sum(log_probs[idx, y_seq])

    def fit(self, actions_list: list, rewards_list: list):
        """Fit recurrent parameters by gradient descent on summed NLL."""
        if len(actions_list) == 0:
            raise ValueError("actions_list is empty; cannot fit model.")

        x_list = []
        y_list = []
        for actions, rewards in zip(actions_list, rewards_list):
            actions = np.asarray(actions, dtype=int)
            rewards = np.asarray(rewards, dtype=float)
            if len(actions) == 0:
                continue
            x_list.append(self._sequence_to_inputs(actions, rewards))
            y_list.append(jnp.array(actions, dtype=int))

        if len(x_list) == 0:
            raise ValueError("No valid sequences available after preprocessing.")

        flat_params, unravel_fn = ravel_pytree(self.params)

        def objective(flat_vector):
            params = unravel_fn(flat_vector)
            total_nll = 0.0
            total_trials = 0
            for x_seq, y_seq in zip(x_list, y_list):
                total_nll = total_nll + self._sequence_nll(params, x_seq, y_seq)
                total_trials += int(y_seq.shape[0])
            return total_nll / jnp.maximum(total_trials, 1)

        grad_fn = jax.grad(objective)

        self.loss_history = []
        current = flat_params
        for _ in range(self.n_epochs):
            loss_value = float(objective(current))
            grad_value = grad_fn(current)

            grad_norm = float(jnp.linalg.norm(grad_value))
            if grad_norm > 5.0:
                grad_value = grad_value * (5.0 / grad_norm)

            current = current - self.learning_rate * grad_value
            self.loss_history.append(loss_value)

        self.params = unravel_fn(current)
        self.is_fitted = True

        final_loss = self.loss_history[-1] if self.loss_history else float(objective(current))
        print(f"RNN fit complete. Mean NLL/trial: {final_loss:.4f}")
        return self

    def get_action_probabilities(self, actions: np.ndarray, rewards: np.ndarray) -> np.ndarray:
        """
        Return P(A_t | history up to t-1) for all trials in a sequence.
        """
        actions = np.asarray(actions, dtype=int)
        rewards = np.asarray(rewards, dtype=float)
        if len(actions) == 0:
            self.hidden_history = np.zeros((0, self.hidden_size), dtype=float)
            return np.zeros((0, self.n_actions), dtype=float)

        x_seq = self._sequence_to_inputs(actions, rewards)
        logits_seq, hidden_seq = self._forward_logits(self.params, x_seq)
        probs = jax.nn.softmax(logits_seq, axis=1)

        self.hidden_history = np.asarray(hidden_seq)
        return np.asarray(probs)

    def get_latent_states(self):
        return self.hidden_history

    def get_num_parameters(self) -> int:
        total = 0
        for value in self.params.values():
            total += int(np.prod(value.shape))
        return total

    def get_model_description(self) -> dict:
        base_desc = super().get_model_description()
        base_desc["description"] = "Minimal recurrent policy model with tanh hidden state and softmax output."
        base_desc["hidden_size"] = f"{self.hidden_size}"
        base_desc["learning_rate"] = f"{self.learning_rate}"
        base_desc["epochs"] = f"{self.n_epochs}"
        return base_desc
