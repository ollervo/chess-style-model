import numpy as np


class MoveModel:
    def __init__(self, input_size=1349, hidden_size=32, seed=42):
        rng = np.random.default_rng(seed)

        # Small random weights break symmetry between neurons.
        self.W1 = (
            rng.standard_normal((input_size, hidden_size))
            * np.sqrt(2.0 / input_size)
        ).astype(np.float32)

        self.b1 = np.zeros(hidden_size, dtype=np.float32)

        self.W2 = (
            rng.standard_normal((hidden_size, 1))
            * np.sqrt(1.0 / hidden_size)
        ).astype(np.float32)

        self.b2 = np.zeros(1, dtype=np.float32)

    def forward(self, inputs):
        """Return probabilities and intermediate values for one position."""

        if inputs.ndim != 2 or inputs.shape[1] != self.W1.shape[0]:
            raise ValueError(
                f"Expected inputs with shape (number_of_moves, "
                f"{self.W1.shape[0]})."
            )

        if inputs.shape[0] == 0:
            raise ValueError("This position has no legal moves.")

        # First layer.
        z1 = inputs @ self.W1 + self.b1

        # ReLU activation: replace negative values with zero.
        hidden = np.maximum(0.0, z1)

        # One score per candidate move.
        scores = (hidden @ self.W2 + self.b2).reshape(-1)

        # Stable softmax: subtracting the maximum prevents overflow.
        shifted_scores = scores - np.max(scores)

        exp_scores = np.exp(shifted_scores)
        probabilities = exp_scores / np.sum(exp_scores)

        # Store intermediate calculations for backpropagation later.
        cache = {
            "inputs": inputs,
            "z1": z1,
            "hidden": hidden,
            "probabilities": probabilities,
            "shifted_scores": shifted_scores,
        }

        return probabilities, cache

    def loss_and_gradients(self, inputs, target_index):
        """Calculate cross-entropy loss and its parameter gradients."""

        probabilities, cache = self.forward(inputs)

        if not 0 <= target_index < len(probabilities):
            raise ValueError("Target index is outside the candidate list.")

        # Compute -log(probability of the target) directly from scores.
        # This remains stable even if that probability underflows to zero.
        shifted = cache["shifted_scores"]

        loss = (
            np.log(np.sum(np.exp(shifted)))
            - shifted[target_index]
        )

        # Softmax + cross-entropy derivative: predicted minus actual.
        d_scores = probabilities.copy()
        d_scores[target_index] -= 1.0

        # Restore the output layer's (number_of_moves, 1) shape.
        d_scores = d_scores[:, None]

        hidden = cache["hidden"]

        # Output layer gradients.
        d_W2 = hidden.T @ d_scores
        d_b2 = np.sum(d_scores, axis=0)

        # Pass the gradient back into the hidden layer.
        d_hidden = d_scores @ self.W2.T

        # ReLU derivative is 1 for positive inputs, otherwise 0.
        d_z1 = d_hidden * (cache["z1"] > 0)

        # First layer gradients.
        d_W1 = inputs.T @ d_z1
        d_b1 = np.sum(d_z1, axis=0)

        gradients = {
            "W1": d_W1,
            "b1": d_b1,
            "W2": d_W2,
            "b2": d_b2,
        }

        return float(loss), gradients

    def train_step(self, inputs, target_index, learning_rate=0.01):
        """Take one gradient descent step on one position."""

        loss, gradients = self.loss_and_gradients(
            inputs,
            target_index,
        )

        self.W1 -= learning_rate * gradients["W1"]
        self.b1 -= learning_rate * gradients["b1"]
        self.W2 -= learning_rate * gradients["W2"]
        self.b2 -= learning_rate * gradients["b2"]

        return loss


if __name__ == "__main__":
    import json
    from pathlib import Path

    import chess

    from encoding import encode_candidates

    dataset_path = (
        Path(__file__).resolve().parent
        / "chess_data"
        / "ollervo"
        / "decisions.jsonl"
    )

    with dataset_path.open(encoding="utf-8") as file:
        example = json.loads(next(file))

    board = chess.Board(example["fen"])
    legal_moves, inputs = encode_candidates(example["fen"])

    model = MoveModel(input_size=inputs.shape[1])

    probabilities, _ = model.forward(inputs)

    # Sort move indices from highest to lowest probability.
    ranking = np.argsort(-probabilities)

    print("Randomly initialized model — not trained yet:\n")

    for index in ranking[:10]:
        move_name = board.san(legal_moves[index])
        probability = probabilities[index]

        print(f"{move_name:8s} {probability:7.2%}")

    played_move = chess.Move.from_uci(example["move_uci"])
    target_index = legal_moves.index(played_move)

    print(f"\nYour actual move: {board.san(played_move)}")
    print(f"Assigned probability: {probabilities[target_index]:.2%}")
    print(f"Probability sum: {probabilities.sum():.6f}")

    assert np.isfinite(probabilities).all()
    assert np.all(probabilities >= 0)
    assert np.isclose(probabilities.sum(), 1.0)