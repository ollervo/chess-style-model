from contextlib import ExitStack
import hashlib
import json
from pathlib import Path


DATA_DIR = (
    Path(__file__).resolve().parent
    / "chess_data"
    / "ollervo"
)

INPUT_PATH = DATA_DIR / "decisions.jsonl"

SPLIT_NAMES = ("train", "validation", "test")


def choose_split(game_id):
    """Assign a game reproducibly to one split."""

    # Hashing gives a repeatable assignment without relying on Python's
    # built-in hash(), which can change between interpreter sessions.
    key = f"chess-style-v1:{game_id}".encode("utf-8")
    digest = hashlib.sha256(key).digest()

    value = int.from_bytes(digest[:8], byteorder="big")
    fraction = value / (2 ** 64)

    if fraction < 0.8:
        return "train"

    if fraction < 0.9:
        return "validation"

    return "test"


def main():
    decision_counts = {name: 0 for name in SPLIT_NAMES}
    game_ids = {name: set() for name in SPLIT_NAMES}

    temporary_paths = {
        name: DATA_DIR / f"{name}.jsonl.part"
        for name in SPLIT_NAMES
    }

    try:
        with ExitStack() as stack:
            source = stack.enter_context(
                INPUT_PATH.open(encoding="utf-8")
            )

            outputs = {
                name: stack.enter_context(
                    path.open("w", encoding="utf-8")
                )
                for name, path in temporary_paths.items()
            }

            for line in source:
                if not line.strip():
                    continue

                example = json.loads(line)
                game_id = example["game_id"]

                split = choose_split(game_id)

                outputs[split].write(line.rstrip("\n") + "\n")

                decision_counts[split] += 1
                game_ids[split].add(game_id)

        # Check that no game appears in more than one set.
        assert game_ids["train"].isdisjoint(game_ids["validation"])
        assert game_ids["train"].isdisjoint(game_ids["test"])
        assert game_ids["validation"].isdisjoint(game_ids["test"])

        for name, path in temporary_paths.items():
            path.replace(DATA_DIR / f"{name}.jsonl")

    finally:
        for path in temporary_paths.values():
            path.unlink(missing_ok=True)

    total_decisions = sum(decision_counts.values())

    if total_decisions == 0:
        raise ValueError("The dataset contains no decisions.")

    for name in SPLIT_NAMES:
        count = decision_counts[name]
        percentage = 100 * count / total_decisions

        print(
            f"{name:10s}: "
            f"{len(game_ids[name]):6,d} games | "
            f"{count:9,d} decisions | "
            f"{percentage:5.1f}%"
        )


if __name__ == "__main__":
    main()