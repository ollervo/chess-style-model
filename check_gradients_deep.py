"""Check every derivative in a tiny two-hidden-layer network."""
import numpy as np
from model_deep import DeepMoveModel


def main():
    model = DeepMoveModel(input_size=4, hidden_size=3, second_hidden_size=3, seed=42)
    for name in model.parameter_names:
        setattr(model, name, getattr(model, name).astype(np.float64))
    # Positive bias helps avoid testing at ReLU's non-differentiable zero.
    model.b1.fill(0.2)
    model.b2.fill(0.2)
    inputs = np.random.default_rng(123).normal(size=(5, 4))
    target = 2
    epsilon = 1e-5
    loss, gradients = model.loss_and_gradients(inputs, target)
    _, base_cache = model.forward(inputs)
    base_masks = {key: base_cache[key] > 0 for key in ('z1', 'z2')}

    def checked_loss():
        _, cache = model.forward(inputs)
        for key, mask in base_masks.items():
            if not np.array_equal(cache[key] > 0, mask):
                raise RuntimeError('Perturbation crossed a ReLU boundary; change the seed or epsilon.')
        shifted = cache['shifted_scores']
        return float(np.log(np.exp(shifted).sum()) - shifted[target])

    print(f'Initial loss: {loss:.6f}')
    for name in model.parameter_names:
        parameter = getattr(model, name)
        numerical = np.zeros_like(parameter)
        for index in np.ndindex(parameter.shape):
            original = parameter[index]
            try:
                parameter[index] = original + epsilon
                plus = checked_loss()
                parameter[index] = original - epsilon
                minus = checked_loss()
            finally:
                parameter[index] = original
            numerical[index] = (plus - minus) / (2 * epsilon)
        error = np.max(np.abs(gradients[name] - numerical))
        print(f'{name}: maximum absolute error = {error:.3e}')
        np.testing.assert_allclose(gradients[name], numerical, rtol=1e-4, atol=1e-7)
    print('All gradient checks passed.')


if __name__ == '__main__':
    main()
