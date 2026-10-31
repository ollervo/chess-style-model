import json
from pathlib import Path
import random

import chess
import numpy as np

from encoding import encode_candidates
from model import MoveModel


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "chess_data" / "ollervo"
MODEL_DIR = ROOT / "models"

TRAIN_SIZE = 1000
VALIDATION_SIZE = 200
EPOCHS = 50
LEARNING_RATE = 0.001


def sample_examples(path, sample_size, seed):
    """Uniformly sample decisions without loading the whole file."""

    rng = random.Random(seed)
    sample = []
    count = 0

    with path.open(encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue

            count += 1

            if len(sample) < sample_size:
                sample.append(json.loads(line))
            else:
                index = rng.randrange(count)

                if index < sample_size:
                    sample[index] = json.loads(line)

    if not sample:
        raise ValueError(f"No examples found in {path}")

    return sample


def prepare_example(example):
    moves, inputs = encode_candidates(example["fen"])

    played_move = chess.Move.from_uci(example["move_uci"])
    target_index = moves.index(played_move)

    return inputs, target_index


def evaluate(model, examples):
    """Measure mean loss and top-1 accuracy without updating weights."""

    total_loss = 0.0
    correct = 0

    for example in examples:
        inputs, target_index = prepare_example(example)
        probabilities, cache = model.forward(inputs)

        shifted = cache["shifted_scores"]

        loss = (
            np.log(np.sum(np.exp(shifted)))
            - shifted[target_index]
        )

        total_loss += float(loss)
        correct += int(np.argmax(probabilities) == target_index)

    return {
        "loss": total_loss / len(examples),
        "accuracy": correct / len(examples),
    }


def uniform_baseline(examples):
    """Expected metrics when every legal move is equally likely."""

    move_counts = np.array([
        len(example["legal_moves"])
        for example in examples
    ])

    return {
        "loss": float(np.mean(np.log(move_counts))),
        "accuracy": float(np.mean(1.0 / move_counts)),
    }


def main():
    print("Sampling training decisions...", flush=True)
    training = sample_examples(
        DATA_DIR / "train.jsonl",
        TRAIN_SIZE,
        seed=42,
    )

    print("Sampling validation decisions...", flush=True)
    validation = sample_examples(
        DATA_DIR / "validation.jsonl",
        VALIDATION_SIZE,
        seed=43,
    )

    print(
        f"Using {len(training):,} training decisions and "
        f"{len(validation):,} validation decisions."
    )

    baseline = uniform_baseline(validation)

    print(
        f"\nUniform validation baseline: "
        f"loss {baseline['loss']:.4f} | "
        f"expected accuracy {baseline['accuracy']:.2%}"
    )

    first_inputs, _ = prepare_example(training[0])
    model = MoveModel(input_size=first_inputs.shape[1])

    initial = evaluate(model, validation)

    print(
        f"Initial model: "
        f"validation loss {initial['loss']:.4f} | "
        f"accuracy {initial['accuracy']:.2%}\n"
    )

    MODEL_DIR.mkdir(exist_ok=True)

    def save_model():
        np.savez(
            MODEL_DIR / "best_model.npz",
            W1=model.W1,
            b1=model.b1,
            W2=model.W2,
            b2=model.b2,
        )

    # Keep the initial model if training never improves validation loss.
    best_validation_loss = initial["loss"]
    save_model()

    shuffle_rng = random.Random(44)

    for epoch in range(1, EPOCHS + 1):
        shuffle_rng.shuffle(training)

        for example in training:
            inputs, target_index = prepare_example(example)

            loss = model.train_step(
                inputs,
                target_index,
                learning_rate=LEARNING_RATE,
            )

            if not np.isfinite(loss):
                raise RuntimeError(
                    "Non-finite training loss. Try a lower learning rate."
                )

        # Evaluate both sets using the weights at the end of the epoch.
        train_metrics = evaluate(model, training)
        validation_metrics = evaluate(model, validation)

        improved = validation_metrics["loss"] < best_validation_loss

        if improved:
            best_validation_loss = validation_metrics["loss"]
            save_model()

        marker = " — saved" if improved else ""

        print(
            f"Epoch {epoch:2d} | "
            f"train loss {train_metrics['loss']:.4f}, "
            f"accuracy {train_metrics['accuracy']:.2%} | "
            f"validation loss {validation_metrics['loss']:.4f}, "
            f"accuracy {validation_metrics['accuracy']:.2%}"
            f"{marker}",
            flush=True,
        )

    print(
        f"\nBest validation loss: {best_validation_loss:.4f}"
        f"\nModel saved to: {MODEL_DIR / 'best_model.npz'}"
    )


if __name__ == "__main__":
    main()