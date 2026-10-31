import json
from pathlib import Path

import chess
import numpy as np

from encoding import encode_candidates
from model import MoveModel


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

played_move = chess.Move.from_uci(example["move_uci"])
target_index = legal_moves.index(played_move)

model = MoveModel(input_size=inputs.shape[1])

print("Training target:", board.san(played_move))
print("Number of legal moves:", len(legal_moves))


def report(step):
    probabilities, _ = model.forward(inputs)
    loss, _ = model.loss_and_gradients(inputs, target_index)

    print(
        f"Step {step:3d} | "
        f"loss {loss:.4f} | "
        f"your move probability {probabilities[target_index]:.2%}"
    )


report(0)

for step in range(1, 201):
    loss = model.train_step(
        inputs,
        target_index,
        learning_rate=0.01,
    )

    if not np.isfinite(loss):
        raise RuntimeError("Training produced a non-finite loss.")

    if step % 20 == 0:
        report(step)

probabilities, _ = model.forward(inputs)

print("\nFinal top moves:")

for index in np.argsort(-probabilities)[:5]:
    print(
        f"{board.san(legal_moves[index]):8s} "
        f"{probabilities[index]:7.2%}"
    )