import argparse
import json
from pathlib import Path

import chess
import chess.engine


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("engine_path", help="Path to the Stockfish executable")
    args = parser.parse_args()

    dataset_path = (
        Path(__file__).resolve().parent
        / "chess_data"
        / "ollervo"
        / "validation.jsonl"
    )

    with dataset_path.open(encoding="utf-8") as file:
        example = json.loads(next(file))

    board = chess.Board(example["fen"])
    legal_moves = list(board.legal_moves)

    if not legal_moves:
        print("This position has no legal moves.")
        return

    # Evaluate every legal move, including bad ones.
    with chess.engine.SimpleEngine.popen_uci(args.engine_path) as engine:
        engine.configure({
            "Threads": 1,
            "Hash": 128,
        })

        print("Engine:", engine.id.get("name"))
        print("Analysing all legal moves...", flush=True)

        analysis = engine.analyse(
            board,
            chess.engine.Limit(depth=10),
            multipv=len(legal_moves),
        )

    print("\nEvaluations from the moving player's perspective:\n")

    for info in analysis:
        move = info["pv"][0]
        score = info["score"].pov(board.turn)

        if score.is_mate():
            evaluation = f"mate {score.mate():+d}"
        else:
            evaluation = f"{score.score() / 100:+.2f}"

        marker = " <-- your move" if move.uci() == example["move_uci"] else ""

        print(
            f"{board.san(move):8s} "
            f"{evaluation:>12s}"
            f"{marker}"
        )


if __name__ == "__main__":
    main()