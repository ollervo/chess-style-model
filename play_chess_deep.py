"""Launch your existing chess interface with support for both checkpoint types.

Place beside play_chess.py and model_deep.py. Run: py play_chess_deep.py
"""
import numpy as np
import play_chess as interface
from model_deep import DeepMoveModel

shallow_load = interface.load_weights
shallow_predict = interface.predict


def load_checkpoint(path):
    with np.load(path, allow_pickle=False) as archive:
        is_deep = 'W3' in archive.files
    if not is_deep:
        return shallow_load(path)
    model = DeepMoveModel.load(path)
    if model.W1.shape[0] != 1353:
        raise ValueError('Expected 1,353 input features.')
    return model


def predict(inputs, checkpoint):
    if isinstance(checkpoint, DeepMoveModel):
        probabilities, _ = checkpoint.forward(inputs)
        probabilities = probabilities.astype(np.float64)
        return probabilities / probabilities.sum()
    return shallow_predict(inputs, checkpoint)


if __name__ == '__main__':
    interface.load_weights = load_checkpoint
    interface.predict = predict
    interface.main()
