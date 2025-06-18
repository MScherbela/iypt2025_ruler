# %%
import numpy as np
import matplotlib.pyplot as plt
from ruler_equations import (
    get_touch_angle,
    return_coeffs,
    evaluate_polynomial,
    arc_length,
    y_end_optimum,
    get_equations_clamped_end,
)
from scipy.optimize import minimize_scalar

plt.rcParams["font.size"] = 14

x_load = 0.08
x_ball = 0.138  # must be chosen in a way that the projectile is still in contact with the ruler
r_ball = 0.01
L1 = 0.13

ruler_width = 0.03
ruler_height = 0.003

force = 2.5
E_modulus = 2.3e9
I_moment_of_inertia = ruler_width * ruler_height**3 / 12
EI = E_modulus * I_moment_of_inertia
load = -force / EI

arc = False  # boolean - whether theoretical ruler length exceeds actual ruler length

# side of ruler touches projectile
if arc == False:
    touch_angle, x_end, y_end, slope_end = get_touch_angle(EI, x_load, force, x_ball, r_ball, load)
    coeffs_left = return_coeffs(x_end, y_end, load, x_load, slope_end)[:4]
    coeffs_right = return_coeffs(x_end, y_end, load, x_load, slope_end)[4:]
    force_right = np.abs(evaluate_polynomial(coeffs_right, x_end, 3) * E_modulus * I_moment_of_inertia)
    theta = np.abs(touch_angle)

    # veryfying whether theoretical ruler length is to big
    if arc_length(0.0, x_end, EI, x_load, load, x_end, y_end, slope_end) - L1 > 0.0:
        arc = True

# tip of ruler touches projectile
if arc == True:  # = if(arc)
    x_end, y_end, touch_angle, coeffs = y_end_optimum(x_load, load, r_ball, x_ball, EI, L1)
    coeffs_left = coeffs[:4]
    coeffs_right = coeffs[4:]
    force_right = np.abs(evaluate_polynomial(coeffs_right, x_end, 3) * E_modulus * I_moment_of_inertia)
    theta = np.abs(touch_angle)

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
plt.plot()
plt.show()
# plt.close()
