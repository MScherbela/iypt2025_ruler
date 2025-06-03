# %%
import numpy as np
import matplotlib.pyplot as plt

x_load = 1.0
x_end = 2.0
y_end = 2.0

force = 2.0
E_modulus = 1.0
I_moment_of_inertia = 1.0
load = -force / (E_modulus * I_moment_of_inertia)

equations_and_values = [
    # yL(0) = 0
    ([1, 0, 0, 0, 0, 0, 0, 0], 0),
    # yL'(0) = 0
    ([0, 1, 0, 0, 0, 0, 0, 0], 0),
    # yL(x_load) = xR(x_load)
    ([1, x_load, x_load**2, x_load**3, -1, -x_load, -(x_load**2), -(x_load**3)], 0),
    # yL'(x_load) = xR'(x_load)
    ([0, 1, 2 * x_load, 3 * x_load**2, 0, -1, -2 * x_load, -3 * x_load**2], 0),
    # yL''(x_load) = xR''(x_load)
    ([0, 0, 2, 6 * x_load, 0, 0, -2, -6 * x_load], 0),
    # yR'''(x_load) - xL'''(x_load) = F/EI,
    ([0, 0, 0, -6, 0, 0, 0, 6], load),
    # yR(x_end) = H
    ([0, 0, 0, 0, 1, x_end, x_end**2, x_end**3], y_end),
    # yR''(x_end) = 0
    ([0, 0, 0, 0, 0, 0, 2, 6 * x_end], 0),
]


def evaluate_polynomial(coeffs, x, derivative=0):
    """Evaluate a polynomial defined by its coefficients at a given x."""
    assert len(coeffs) == 4
    if derivative == 0:
        return coeffs[0] + coeffs[1] * x + coeffs[2] * x**2 + coeffs[3] * x**3
    elif derivative == 1:
        return coeffs[1] + 2 * coeffs[2] * x + 3 * coeffs[3] * x**2
    elif derivative == 2:
        return 2 * coeffs[2] + 6 * coeffs[3] * x
    elif derivative == 3:
        return 6 * coeffs[3] * np.ones_like(x)


M = np.array([eq[0] for eq in equations_and_values])
b = np.array([eq[1] for eq in equations_and_values])
coeffs = np.linalg.solve(M, b)
coeffs_left = coeffs[:4]
coeffs_right = coeffs[4:]

x_left = np.linspace(0, x_load, 100)
x_right = np.linspace(x_load, x_end, 100)
y_left = coeffs_left[0] + coeffs_left[1] * x_left + coeffs_left[2] * x_left**2 + coeffs_left[3] * x_left**3
y_right = coeffs_right[0] + coeffs_right[1] * x_right + coeffs_right[2] * x_right**2 + coeffs_right[3] * x_right**3

fig, axes = plt.subplots(2, 2, figsize=(10, 6))

for derivative in range(4):
    ax = axes.flatten()[derivative]
    y_left = evaluate_polynomial(coeffs_left, x_left, derivative)
    y_right = evaluate_polynomial(coeffs_right, x_right, derivative)
    ax.plot(x_left, y_left, f"C{derivative}")
    ax.plot(x_right, y_right, f"C{derivative}")
    ax.set_title(f"y" + "'" * derivative)
    ax.axvline(x_load, color="gray", linestyle="--")
    ax.axhline(0, color="k", linestyle="-", zorder=-1)
    if derivative == 0:
        ax.plot([x_end, x_end], [0, y_end], color="k", lw=2)

force_left = evaluate_polynomial(coeffs_left, x_load, 3) * E_modulus * I_moment_of_inertia
force_right = -evaluate_polynomial(coeffs_right, x_load, 3) * E_modulus * I_moment_of_inertia


fig.tight_layout()
