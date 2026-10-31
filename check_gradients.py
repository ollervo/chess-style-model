import numpy as np

from model import MoveModel


# Use a tiny network so the check is quick and easy to inspect.
rng = np.random.default_rng(123)

model = MoveModel(
    input_size=4,
    hidden_size=3,
    seed=42,
)

# Float64 gives more accurate finite-difference calculations.
for name in ("W1", "b1", "W2", "b2"):
    parameter = getattr(model, name)
    setattr(model, name, parameter.astype(np.float64))

# Five hypothetical candidate moves, each with four features.
inputs = rng.normal(size=(5, 4))
target_index = 2

loss, gradients = model.loss_and_gradients(
    inputs,
    target_index,
)

epsilon = 1e-5

# ReLU isn't differentiable at zero. Ensure our perturbations
# won't cross a ReLU boundary in this particular check.
_, cache = model.forward(inputs)

largest_z1_change = epsilon * max(
    1.0,
    float(np.max(np.abs(inputs))),
)

if np.min(np.abs(cache["z1"])) <= largest_z1_change:
    raise RuntimeError("Too close to a ReLU boundary; change the seed.")

print(f"Initial loss: {loss:.6f}\n")

for name in ("W1", "b1", "W2", "b2"):
    parameter = getattr(model, name)
    analytical = gradients[name]
    numerical = np.zeros_like(parameter)

    for index in np.ndindex(parameter.shape):
        original = parameter[index]

        try:
            parameter[index] = original + epsilon
            loss_plus, _ = model.loss_and_gradients(
                inputs,
                target_index,
            )

            parameter[index] = original - epsilon
            loss_minus, _ = model.loss_and_gradients(
                inputs,
                target_index,
            )
        finally:
            parameter[index] = original

        numerical[index] = (
            (loss_plus - loss_minus) / (2 * epsilon)
        )

    max_error = np.max(np.abs(analytical - numerical))

    print(f"{name}: maximum absolute error = {max_error:.3e}")

    np.testing.assert_allclose(
        analytical,
        numerical,
        rtol=1e-4,
        atol=1e-7,
        err_msg=f"Gradient check failed for {name}",
    )

print("\nAll gradient checks passed.")