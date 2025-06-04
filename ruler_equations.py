import numpy as np


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


def get_equations_open_end(x_load, load, x_end, y_end):
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
    M = np.array([eq[0] for eq in equations_and_values])
    b = np.array([eq[1] for eq in equations_and_values])
    return M, b


def get_equations_clamped_end(x_load, load, x_end, y_end, slope_end):
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
        # yR'(x_end) = slope_end
        ([0, 0, 0, 0, 0, 1, 2 * x_end, 3 * x_end**2], slope_end),
    ]
    M = np.array([eq[0] for eq in equations_and_values])
    b = np.array([eq[1] for eq in equations_and_values])
    return M, b


def get_bending_energy(coeffs, x1, x2, EI):
    c2, c3 = coeffs[2], coeffs[3]
    Epot = 2 * c2**2 * (x2 - x1) + 6 * c2 * c3 * (x2**2 - x1**2) + 6 * c3**2 * (x2**3 - x1**3)
    return Epot * EI
