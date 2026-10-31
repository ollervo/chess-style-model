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