# %%
import numpy as np
import matplotlib.pyplot as plt
from ruler_equations import get_equations_clamped_end, get_equations_open_end, evaluate_polynomial, get_bending_energy
from scipy.optimize import minimize_scalar


x_load = 0.1
x_ball = 0.2
r_ball = 0.05

ruler_width = 0.05
ruler_height = 2e-3

force = 5.0
E_modulus = 2e9
I_moment_of_inertia = ruler_width * ruler_height**3 / 12
EI = E_modulus * I_moment_of_inertia
load = -force / EI


def get_touch_angle(EI, x_load, force, x_ball, r_ball):
    def potential_energy(touch_angle):
        """Angle of the touch point with the vertical. At 0, the ruler touches at the top, at pi/2 it touches at the left end"""
        x_end = x_ball - np.sin(touch_angle) * r_ball
        y_end = np.cos(touch_angle) * r_ball + r_ball
        slope = np.tan(touch_angle)

        M, b = get_equations_clamped_end(x_load, load, x_end, y_end, slope)
        coeffs = np.linalg.solve(M, b)
        print(coeffs, x_load, EI)
        E_left = get_bending_energy(coeffs[:4], 0, x_load, EI)
        E_right = get_bending_energy(coeffs[4:], x_load, x_end, EI)
        E_load = force * evaluate_polynomial(coeffs[:4], x_load, 0)
        E = E_left + E_right + E_load
        return E

    # Step 1: rough search to approximately find minimum
    n_angles = 100
    angles = np.linspace(0, 0.99 * np.pi / 2, n_angles)
    Epot = np.array([potential_energy(angle) for angle in angles])
    idx_min = np.argmin(Epot)
    # touch_angle = angles[idx_min]

    # Step 2: refine search around the minimum
    assert idx_min > 0 and idx_min < n_angles - 1, f"Minimum should not be at the boundaries; idx={idx_min}"
    result = minimize_scalar(potential_energy, bracket=(angles[idx_min - 1], angles[idx_min + 1]), method="brent")
    touch_angle = result.x
    x_end = x_ball - np.sin(touch_angle) * r_ball
    y_end = np.cos(touch_angle) * r_ball + r_ball
    slope = np.tan(touch_angle)

    return touch_angle, x_end, y_end, slope


touch_angle, x_end, y_end, slope_end = get_touch_angle(EI, x_load, force, x_ball, r_ball)

M, b = get_equations_clamped_end(x_load, load, x_end, y_end, slope_end)

coeffs = np.linalg.solve(M, b)
coeffs_left = coeffs[:4]
coeffs_right = coeffs[4:]

x_left = np.linspace(0, x_load, 100)
x_right = np.linspace(x_load, x_end, 100)

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
        circle = plt.Circle((x_ball, r_ball), r_ball, color="dimgray", fill=True)
        ax.add_patch(circle)
        ax.plot([x_end, x_end], [0, y_end], color="k", lw=2)
        ax.axis("equal")

force_left = evaluate_polynomial(coeffs_left, x_load, 3) * E_modulus * I_moment_of_inertia
force_right = -evaluate_polynomial(coeffs_right, x_load, 3) * E_modulus * I_moment_of_inertia


fig.tight_layout()
