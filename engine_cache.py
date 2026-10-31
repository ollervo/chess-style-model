import argparse
import hashlib
import json
from pathlib import Path
import sqlite3

import chess
import chess.engine


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "chess_data" / "ollervo"


def analyse_cached(engine, connection, board, depth=10):
    """Return evaluations indexed by UCI move, using a persistent cache."""

    connection.execute("""
        CREATE TABLE IF NOT EXISTS evaluations (
            cache_key TEXT PRIMARY KEY,
            result_json TEXT NOT NULL
        )
    """)

    # Keep evaluations from different positions/settings separate.
    settings = {
        "schema": 1,
        "engine": engine.id,
        "fen": board.fen(),
        "depth": depth,
        "threads": 1,
        "hash_mb": 128,
        "chess960": board.chess960,
    }

    settings_json = json.dumps(settings, sort_keys=True)
    cache_key = hashlib.sha256(settings_json.encode()).hexdigest()

    row = connection.execute(
        "SELECT result_json FROM evaluations WHERE cache_key = ?",
        (cache_key,),
    ).fetchone()

    if row is not None:
        return json.loads(row[0]), True

    legal_moves = list(board.legal_moves)

    if not legal_moves:
        return {}, False

    engine.configure({
        "Threads": settings["threads"],
        "Hash": settings["hash_mb"],
    })

    # Avoid carrying the previous position's hash-table analysis into
    # this search. Stockfish exposes Clear Hash as a UCI button.
    engine.configure({"Clear Hash": None})

    analysis = engine.analyse(
        board,
        chess.engine.Limit(depth=depth),
        multipv=len(legal_moves),
        game=object(),
    )

    results = {}

    for info in analysis:
        move = info["pv"][0]
        score = info["score"].pov(board.turn)

        results[move.uci()] = {
            "cp": score.score(),
            "mate": score.mate(),
        }

    expected_moves = {move.uci() for move in legal_moves}

    if set(results) != expected_moves:
        raise RuntimeError("Stockfish did not return every legal move.")

    connection.execute(
        """
        INSERT OR REPLACE INTO evaluations (cache_key, result_json)
        VALUES (?, ?)
        """,
        (cache_key, json.dumps(results)),
    )

    connection.commit()

    return results, False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("engine_path")
    args = parser.parse_args()

    with (DATA_DIR / "validation.jsonl").open(encoding="utf-8") as file:
        example = json.loads(next(file))

    board = chess.Board(example["fen"])
    database_path = DATA_DIR / "engine_cache.sqlite3"

    connection = sqlite3.connect(database_path)

    try:
        with chess.engine.SimpleEngine.popen_uci(args.engine_path) as engine:
            evaluations, cache_hit = analyse_cached(
                engine,
                connection,
                board,
            )

        print("Loaded from cache:", cache_hit)

        for move_uci, evaluation in evaluations.items():
            move = chess.Move.from_uci(move_uci)

            print(
                f"{board.san(move):8s} "
                f"cp={str(evaluation['cp']):>6s} "
                f"mate={evaluation['mate']}"
            )
    finally:
        connection.close()


if __name__ == "__main__":
    main()