"""A shared move-scoring network with two ReLU hidden layers.

For every candidate: 1353 features -> 32 -> 32 -> one score.
Softmax is applied across all legal candidates in a single position.
"""
import numpy as np


class DeepMoveModel:
    parameter_names = ('W1', 'b1', 'W2', 'b2', 'W3', 'b3')

    def __init__(self, input_size=1353, hidden_size=32, second_hidden_size=32, seed=42):
        if min(input_size, hidden_size, second_hidden_size) < 1:
            raise ValueError('Layer sizes must be positive.')
        rng = np.random.default_rng(seed)
        # He initialization for layers followed by ReLU.
        self.W1 = (rng.standard_normal((input_size, hidden_size))
                   * np.sqrt(2.0 / input_size)).astype(np.float32)
        self.b1 = np.zeros(hidden_size, dtype=np.float32)
        self.W2 = (rng.standard_normal((hidden_size, second_hidden_size))
                   * np.sqrt(2.0 / hidden_size)).astype(np.float32)
        self.b2 = np.zeros(second_hidden_size, dtype=np.float32)
        self.W3 = (rng.standard_normal((second_hidden_size, 1))
                   * np.sqrt(1.0 / second_hidden_size)).astype(np.float32)
        self.b3 = np.zeros(1, dtype=np.float32)

    def forward(self, inputs):
        if inputs.ndim != 2 or inputs.shape[1] != self.W1.shape[0] or len(inputs) == 0:
            raise ValueError(f'Expected a nonempty (moves, {self.W1.shape[0]}) input array.')
        z1 = inputs @ self.W1 + self.b1
        hidden1 = np.maximum(0.0, z1)
        z2 = hidden1 @ self.W2 + self.b2
        hidden2 = np.maximum(0.0, z2)
        scores = (hidden2 @ self.W3 + self.b3).reshape(-1)
        if not np.isfinite(scores).all():
            raise ValueError('Non-finite scores; check inputs or lower the learning rate.')
        shifted_scores = scores - scores.max()
        exp_scores = np.exp(shifted_scores)
        probabilities = exp_scores / exp_scores.sum()
        return probabilities, {
            'inputs': inputs, 'z1': z1, 'hidden1': hidden1,
            'z2': z2, 'hidden2': hidden2,
            'probabilities': probabilities, 'shifted_scores': shifted_scores,
        }

    def loss_and_gradients(self, inputs, target_index):
        probabilities, cache = self.forward(inputs)
        if not 0 <= target_index < len(probabilities):
            raise ValueError('Target index is outside the candidate list.')
        shifted = cache['shifted_scores']
        loss = np.log(np.exp(shifted).sum()) - shifted[target_index]
        d_scores = probabilities.copy()
        d_scores[target_index] -= 1.0
        d_scores = d_scores[:, None]

        # Output layer -> second hidden layer.
        d_W3 = cache['hidden2'].T @ d_scores
        d_b3 = d_scores.sum(axis=0)
        d_hidden2 = d_scores @ self.W3.T
        d_z2 = d_hidden2 * (cache['z2'] > 0)

        # Second hidden layer -> first hidden layer.
        d_W2 = cache['hidden1'].T @ d_z2
        d_b2 = d_z2.sum(axis=0)
        d_hidden1 = d_z2 @ self.W2.T
        d_z1 = d_hidden1 * (cache['z1'] > 0)

        # First hidden layer -> inputs.
        d_W1 = inputs.T @ d_z1
        d_b1 = d_z1.sum(axis=0)
        return float(loss), {
            'W1': d_W1, 'b1': d_b1,
            'W2': d_W2, 'b2': d_b2,
            'W3': d_W3, 'b3': d_b3,
        }

    def train_step(self, inputs, target_index, learning_rate=0.001):
        loss, gradients = self.loss_and_gradients(inputs, target_index)
        # All gradients were computed using the same pre-update weights.
        for name in self.parameter_names:
            getattr(self, name)[:] -= learning_rate * gradients[name]
        return loss

    def save(self, path):
        np.savez(path, **{name: getattr(self, name) for name in self.parameter_names})

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as archive:
            if set(archive.files) != set(cls.parameter_names):
                raise ValueError('Expected a two-hidden-layer checkpoint with W1/b1, W2/b2, W3/b3.')
            parameters = {name: np.array(archive[name], dtype=np.float32)
                          for name in cls.parameter_names}
        w1, w2, w3 = (parameters[name] for name in ('W1', 'W2', 'W3'))
        if w1.ndim != 2 or w2.ndim != 2 or w3.ndim != 2:
            raise ValueError('Weight matrices must be two-dimensional.')
        model = cls(w1.shape[0], w1.shape[1], w2.shape[1])
        for name, value in parameters.items():
            if value.shape != getattr(model, name).shape or not np.isfinite(value).all():
                raise ValueError(f'Invalid checkpoint parameter: {name}.')
            setattr(model, name, value)
        return model
