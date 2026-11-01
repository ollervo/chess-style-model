"""Train a two-hidden-layer model on the same engine-enriched samples.

Reuses the data preparation/evaluation functions in your existing
train_engine.py. No Stockfish searches or new feature encoding are needed.
"""
import json
from pathlib import Path
import random
import time
import numpy as np

from model_deep import DeepMoveModel
from train_engine import sample_examples, prepare_example, evaluate, uniform_baseline

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / 'chess_data' / 'ollervo'
MODEL_DIR = ROOT / 'models'
TRAIN_SIZE = 10_000
VALIDATION_SIZE = 2_000
EPOCHS = 40
LEARNING_RATE = 0.0003


def main():
    print('Loading the same sampled decisions as the one-hidden-layer experiment...', flush=True)
    training = sample_examples(DATA_DIR / 'train_engine.jsonl', TRAIN_SIZE, seed=42)
    validation = sample_examples(DATA_DIR / 'validation_engine.jsonl', VALIDATION_SIZE, seed=43)
    if len(training) != TRAIN_SIZE or len(validation) != VALIDATION_SIZE:
        raise ValueError('Expected 10,000 training and 2,000 validation decisions. '
                         'Finish the full prepare_engine_data.py run first, or deliberately adjust the sizes.')
    if {item['game_id'] for item in training} & {item['game_id'] for item in validation}:
        raise ValueError('A game occurs in both training and validation.')
    first_inputs, _ = prepare_example(training[0])
    if first_inputs.shape[1] != 1353:
        raise ValueError('Expected 1,353 features. Check prepare_example() in train_engine.py.')
    model = DeepMoveModel(input_size=1353, hidden_size=32, second_hidden_size=32, seed=42)
    parameter_count = sum(getattr(model, name).size for name in model.parameter_names)
    print(f'Architecture: 1353 -> 32 -> 32 -> 1 ({parameter_count:,} parameters)')
    baseline = uniform_baseline(validation)
    print(f"Uniform validation baseline: loss {baseline['loss']:.4f} | expected accuracy {baseline['accuracy']:.2%}")
    initial = evaluate(model, validation)
    print(f"Initial model: validation loss {initial['loss']:.4f} | accuracy {initial['accuracy']:.2%}")
    MODEL_DIR.mkdir(exist_ok=True)
    checkpoint = MODEL_DIR / 'best_engine_deep_model.npz'
    metadata = MODEL_DIR / 'best_engine_deep_model.json'
    best_loss, best_epoch = initial['loss'], 0

    def save(epoch, metrics):
        model.save(checkpoint)
        metadata.write_text(json.dumps({
            'architecture': [1353, 32, 32, 1], 'seed': 42,
            'learning_rate': LEARNING_RATE, 'epoch': epoch,
            'train_size': len(training), 'validation_size': len(validation),
            'validation_loss': float(metrics['loss']),
            'validation_accuracy': float(metrics['accuracy']),
            'train_file': 'train_engine.jsonl', 'validation_file': 'validation_engine.jsonl',
        }, indent=2), encoding='utf-8')

    save(0, initial)
    shuffle_rng = random.Random(44)
    for epoch in range(1, EPOCHS + 1):
        started = time.perf_counter()
        shuffle_rng.shuffle(training)
        for example in training:
            inputs, target = prepare_example(example)
            loss = model.train_step(inputs, target, learning_rate=LEARNING_RATE)
            if not np.isfinite(loss):
                raise RuntimeError('Non-finite loss; try a lower learning rate.')
        train_metrics = evaluate(model, training)
        val_metrics = evaluate(model, validation)
        improved = val_metrics['loss'] < best_loss
        if improved:
            best_loss, best_epoch = val_metrics['loss'], epoch
            save(epoch, val_metrics)
        elapsed = time.perf_counter() - started
        print(
            f"Epoch {epoch:2d} | train loss {train_metrics['loss']:.4f}, "
            f"accuracy {train_metrics['accuracy']:.2%} | "
            f"validation loss {val_metrics['loss']:.4f}, "
            f"accuracy {val_metrics['accuracy']:.2%} | {elapsed:.1f}s"
            + (' -- saved' if improved else ''), flush=True)
    print(f'\nBest validation loss: {best_loss:.6f} (epoch {best_epoch})\nSaved to: {checkpoint}')
    print('Test data was not used. Existing shallow checkpoints were not changed.')


if __name__ == '__main__':
    main()
