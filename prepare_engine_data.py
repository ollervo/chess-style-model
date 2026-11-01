import argparse
import json
from pathlib import Path
import sqlite3
import time

import chess
import chess.engine

from engine_cache import analyse_cached
from train import sample_examples


DATA_DIR = (
    Path(__file__).resolve().parent
    / "chess_data"
    / "ollervo"
)


def prepare_split(engine, connection, split, size, seed, depth):
    examples = sample_examples(
        DATA_DIR / f"{split}.jsonl",
        size,
        seed,
    )

    destination = DATA_DIR / f"{split}_engine.jsonl"
    temporary = destination.with_suffix(".jsonl.part")

    started = time.perf_counter()
    cache_hits = 0

    try:
        with temporary.open("w", encoding="utf-8") as output:
            for number, example in enumerate(examples, start=1):
                board = chess.Board(example["fen"])

                evaluations, cache_hit = analyse_cached(
                    engine,
                    connection,
                    board,
                    depth=depth,
                )

                cache_hits += int(cache_hit)

                enriched = {
                    **example,
                    "engine_evaluations": evaluations,
                    "engine_settings": {
                        "engine": engine.id,
                        "depth": depth,
                        "threads": 1,
                        "hash_mb": 128,
                    },
                }

                output.write(json.dumps(enriched) + "\n")

                if number == 1 or number % 25 == 0:
                    elapsed = time.perf_counter() - started

                    print(
                        f"{split}: {number}/{len(examples)} | "
                        f"cache hits {cache_hits} | "
                        f"elapsed {elapsed:.1f}s",
                        flush=True,
                    )

        temporary.replace(destination)

    finally:
        temporary.unlink(missing_ok=True)

    elapsed = time.perf_counter() - started

    print(
        f"{split} finished: {len(examples)} decisions "
        f"in {elapsed:.1f}s\n"
        f"Saved to {destination}",
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("engine_path")
    parser.add_argument("--train-size", type=int, default=10_000)
    parser.add_argument("--validation-size", type=int, default=2_000)
    parser.add_argument("--depth", type=int, default=10)
    args = parser.parse_args()

    if min(args.train_size, args.validation_size, args.depth) < 1:
        parser.error("Sizes and depth must be positive.")

    connection = sqlite3.connect(
        DATA_DIR / "engine_cache.sqlite3"
    )

    try:
        with chess.engine.SimpleEngine.popen_uci(
            args.engine_path
        ) as engine:
            print("Engine:", engine.id.get("name"), flush=True)

            prepare_split(
                engine,
                connection,
                split="train",
                size=args.train_size,
                seed=42,
                depth=args.depth,
            )

            prepare_split(
                engine,
                connection,
                split="validation",
                size=args.validation_size,
                seed=43,
                depth=args.depth,
            )
    finally:
        connection.close()


if __name__ == "__main__":
    main()