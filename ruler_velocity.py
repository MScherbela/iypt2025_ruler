# %%
import math
import numpy as np
import csv
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from ruler_equations import get_touch_angle, return_coeffs, evaluate_polynomial, arc_length, y_end_optimum
from scipy.integrate import quad
from scipy.optimize import root_scalar


class CriticalDistance:
    def __init__(self):
        self.terminal = True
        self.direction = 0

    def __call__(self, t, y):
        x = float(y[0])
        return x - 0.17  # = length L2


def x_dotdot(x_ball, xdot):  #
    global arc

    # side of ruler touches projectile
    if arc == False:
        touch_angle, x_end, y_end, slope_end = get_touch_angle(EI, x_load, force, x_ball, r_ball, load)
        coeffs_right = return_coeffs(x_end, y_end, load, x_load, slope_end)[4:]
        force_right = np.abs(evaluate_polynomial(coeffs_right, x_end, 3) * E_modulus * I_moment_of_inertia)
        theta = np.abs(touch_angle)

        # veryfying whether just the tip of the ruler touches the projectile
        if arc_length(0.0, x_end, EI, x_load, load, x_end, y_end, slope_end) - L1 > 0.0:
            arc = True

    # tip of ruler touches projectile
    if arc == True:  # = if(arc)
        x_end, y_end, touch_angle, coeffs = y_end_optimum(x_load, load, r_ball, x_ball, EI, L1)
        coeffs_right = coeffs[4:]
        force_right = np.abs(evaluate_polynomial(coeffs_right, x_end, 3) * E_modulus * I_moment_of_inertia)
        theta = np.abs(touch_angle)

        # ruler is out of reach
        if np.isnan(y_end):
            return -(mu_rolling * m * g) / m * (xdot > 0.0) + 0.0

    k = (r_ball - y_end) / (x_ball - x_end)
    phi = np.abs(np.arctan(k))
    gamma = np.pi / 2 - (phi + theta)

    if xdot == 0.0:
        mu = mu_static
    else:
        mu = mu_kinetic

    Fg = m * g
    Fn = force_right * np.cos(theta)
    F = Fn * np.cos(gamma)
    Fx = F * np.cos(phi)
    Fy = F * np.sin(phi)

    FR1 = mu * (Fy) * np.cos(theta)
    FR2 = mu * (Fy + Fg)

    F_net = Fx - FR1 - FR2
    if F_net >= 0.0:
        return F_net / m

    if F_net < 0.0:
        if xdot > 0:
            return F_net / m
        else:
            return 0.0


def dydt(t, y):
    return np.array([y[1], x_dotdot(y[0], y[1])])


param_values = []
x_dot_values = []

param_i = 0.0
param_f = 0.4
n = 100
step_size = (param_f - param_i) / n

for i in np.arange(param_i, param_f, step_size):
    m_o = i
    print(i)

    ##############################################################################################################################

    arc = False

    # properties of the ruler
    L1 = 0.13
    L2 = 0.17
    tilt = 0
    phi = tilt * np.pi / 180
    ruler_width = 0.03
    ruler_height = 0.003
    E_modulus = 2.1 * 10**9
    I_moment_of_inertia = ruler_width * ruler_height**3 / 12
    EI = E_modulus * I_moment_of_inertia

    # properties of the projectile
    x_ball = 0.133
    m = 0.05
    r_ball = 0.01
    w = r_ball * 2
    A = 2 * r_ball * w
    V = math.pow(r_ball, 2) * math.pi * w
    Cd = 1
    rho = 1.293

    # Friction coefficients
    mu_static = 0.15
    mu_kinetic = 0.10
    mu_rolling = 0.01

    # force properties
    """m_o = 0.25"""
    g = 9.81
    x_load = 0.08
    force = m_o * 9.81
    load = force / EI

    ##############################################################################################################################

    y_0 = [x_ball, 0.0]
    t0 = 0.0
    t1 = 0.5
    should_continue = False
    try:
        sol = solve_ivp(dydt, t_span=[t0, t1], y0=y_0, events=CriticalDistance())

        if sol.status != 1:
            print("took too long")

        x_dot_max = np.max(sol.y[1])
        param_values.append(i)
        x_dot_values.append(x_dot_max)
    except:
        continue

plt.plot(param_values, x_dot_values)
plt.show()
# plt.close()
