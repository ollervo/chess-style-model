import argparse
import json
from pathlib import Path
import sqlite3

import chess
import chess.engine
import numpy as np

from encoding import encode_engine_candidates
from engine_cache import analyse_cached


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("engine_path")
    args = parser.parse_args()

    data_dir = (
        Path(__file__).resolve().parent
        / "chess_data"
        / "ollervo"
    )

    with (data_dir / "validation.jsonl").open(
        encoding="utf-8"
    ) as file:
        example = json.loads(next(file))

    board = chess.Board(example["fen"])

    connection = sqlite3.connect(
        data_dir / "engine_cache.sqlite3"
    )

    try:
        with chess.engine.SimpleEngine.popen_uci(
            args.engine_path
        ) as engine:
            evaluations, cache_hit = analyse_cached(
                engine,
                connection,
                board,
            )
    finally:
        connection.close()

    moves, inputs = encode_engine_candidates(
        example["fen"],
        evaluations,
    )

    print("Loaded evaluations from cache:", cache_hit)
    print("Input shape:", inputs.shape)

    print(
        "\nMove       Quality     Gap   Win mate  Lose mate"
    )

    for move, row in zip(moves, inputs):
        quality, gap, win_mate, lose_mate = row[-4:]

        print(
            f"{board.san(move):8s} "
            f"{quality:9.3f} "
            f"{gap:7.3f} "
            f"{win_mate:10.0f} "
            f"{lose_mate:10.0f}"
        )

    assert inputs.shape == (len(moves), 1353)
    assert np.isfinite(inputs).all()


if __name__ == "__main__":
    main()