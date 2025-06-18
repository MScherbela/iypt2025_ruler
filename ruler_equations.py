import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar
from scipy.integrate import quad

def evaluate_polynomial(coeffs, x, derivative):#derivative=0
    """Evaluate a polynomial defined by its coefficients at a given x."""
    #assert len(coeffs) == 4
    if(len(coeffs)!=4): return np.nan
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

def get_touch_angle(EI, x_load, force, x_ball, r_ball,load):
    def potential_energy(touch_angle):
        """Angle of the touch point with the vertical. At 0, the ruler touches at the top, at pi/2 it touches at the left end"""
        x_end = x_ball - np.sin(touch_angle) * r_ball
        y_end = np.cos(touch_angle) * r_ball + r_ball
        slope = np.tan(touch_angle)

        M, b = get_equations_clamped_end(x_load, load, x_end, y_end, slope)
        coeffs = np.linalg.solve(M, b)
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

def return_coeffs( x_end, y_end, load, x_load, slope_end):
    M, b = get_equations_clamped_end(x_load, load, x_end, y_end, slope_end)

    coeffs = np.linalg.solve(M, b)
    coeffs_left = coeffs[:4]
    coeffs_right = coeffs[4:]
    return coeffs

def arc_length(x_start, x_target,EI, x_load,load, x_end, y_end, slope_end):
    def arc_length_integrand(x):
        df = evaluate_polynomial(return_coeffs( x_end, y_end, load, x_load, slope_end)[4:],x, 1)
        return np.sqrt(1 + df**2)
    s_actual, _ = quad(arc_length_integrand, x_start, x_target)
    return s_actual

def y_end_optimum(x_load, load, r_ball, x_ball, EI, L1):
    def shape(y_end):
        delta = y_end - r_ball
        radicand = r_ball**2 - delta**2
        if radicand < 0:
            return np.ones(8) * np.nan
        x_end = -np.sqrt(radicand) + x_ball

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

        coeffs = np.linalg.solve(M, b)
        coeffs_left = coeffs[:4]
        coeffs_right = coeffs[4:]

        return coeffs

    def arc_length_dif(y_end):
        coeffs = shape(y_end)
        if np.any(np.isnan(coeffs)):
            return np.inf

        delta = y_end - r_ball
        radicand = r_ball**2 - delta**2
        if radicand < 0:
            return np.ones(8) * np.nan
        x_end = -np.sqrt(radicand) + x_ball

        x_start = 0.0
        x_target = x_end
        def arc_length_integrand_dif(x):
            df = evaluate_polynomial(shape(y_end)[4:],x, 1)
            return np.sqrt(1 + df**2)
        s_actual, _ = quad(arc_length_integrand_dif, x_start, x_target)
        return np.abs(s_actual - L1)

    n_y = 100
    y_values = np.linspace(0, 0.999*2*r_ball, n_y) #should be starting with a bit less than r_ball
    f = np.array([arc_length_dif(y_end) for y_end in y_values])
    idx_min = np.argmin(f)

    assert idx_min < n_y - 1, f"Minimum should not be at the boundaries; idx={idx_min}"
    if(idx_min == 0): return np.nan,np.nan,np.nan,([np.nan,np.nan,np.nan,np.nan])
    result = minimize_scalar(arc_length_dif, bracket=(y_values[idx_min - 1], y_values[idx_min + 1]), method="brent")
    y_end = result.x

    delta = y_end - r_ball
    radicand = r_ball**2 - delta**2
    if radicand < 0:
        return np.ones(8) * np.nan
    x_end = -np.sqrt(radicand) + x_ball

    coeffs = shape(y_end)
    coeffs_right = shape(y_end)[4:]
    touch_angle = np.arctan(np.abs(evaluate_polynomial(coeffs_right,x_end, 1)))

    return x_end, y_end, touch_angle, coeffs
