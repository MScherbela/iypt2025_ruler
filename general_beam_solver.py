# %%
import numpy as np
import matplotlib.pyplot as plt


def get_coeffs_pivot_center(x, y):
    coeffs = np.array(
        [
            [1, x, x**2, x**3, 0, 0, 0, 0],  # value left
            [0, 0, 0, 0, 1, 0, 0, 0],  # value right
            [0, 1, 2 * x, 3 * x**2, 0, -1, 0, 0],  # same slope
            [0, 0, 2, 6 * x, 0, 0, -2, 0],  # same curvature
        ]
    )
    values = np.array([y, y, 0, 0])
    return coeffs, values


def get_coeffs_pivot_end(x, y):
    coeffs = np.array(
        [
            [1, x, x**2, x**3],  # value
            [0, 0, 2, 6 * x],  # no curvature
        ]
    )
    values = np.array([y, 0])
    return coeffs, values


def get_coeffs_clamp_end(x, y, slope):
    coeffs = np.array(
        [
            [1, x, x**2, x**3],  # value
            [0, 1, 2 * x, 3 * x**2],  # slope
        ]
    )
    values = np.array([y, slope])
    return coeffs, values


def get_coeffs_clamp_center(x, y, slope):
    c_left, values_left = get_coeffs_clamp_end(x, y, slope)
    c_right, values_right = get_coeffs_clamp_end(0, y, slope)
    zeros = np.zeros((2, 4))
    coeffs = np.block([[c_left, zeros], [zeros, c_right]])
    values = np.concatenate((values_left, values_right))
    return coeffs, values


class Beam:
    def __init__(self, constraints):
        self.constraints = constraints
        self.x0 = np.array([c[1] for c in constraints])
        A, b = self.get_equations()
        self.coeffs = np.linalg.solve(A, b).reshape([-1, 4])

    @property
    def n_segments(self):
        return len(self.constraints) - 1

    def get_equations(self):
        n_seg = len(self.constraints) - 1
        A = np.zeros([4 * n_seg, 4 * n_seg])
        values = np.zeros([4 * n_seg])
        idx_eq = 0
        idx_segment = 0
        for idx_segment, constraint in enumerate(self.constraints):
            L = self.x0[idx_segment] - self.x0[idx_segment - 1]
            if constraint[0] == "clamp":
                if idx_segment > 0:  # clamp previous segment
                    c, v = get_coeffs_clamp_end(L, constraint[2], constraint[3])
                    A[idx_eq : idx_eq + 2, 4 * idx_segment - 4 : 4 * idx_segment] = c
                    values[idx_eq : idx_eq + 2] = v
                    idx_eq += 2
                if idx_segment < n_seg:  # clamp next segment
                    c, v = get_coeffs_clamp_end(0, constraint[2], constraint[3])
                    A[idx_eq : idx_eq + 2, 4 * idx_segment : 4 * idx_segment + 4] = c
                    values[idx_eq : idx_eq + 2] = v
                    idx_eq += 2
            elif constraint[0] == "pivot":
                if idx_segment == 0:
                    c, v = get_coeffs_pivot_end(0, constraint[2])
                    A[idx_eq : idx_eq + 2, 4 * idx_segment : 4 * idx_segment + 4] = c
                    values[idx_eq : idx_eq + 2] = v
                    idx_eq += 2
                elif idx_segment == n_seg:
                    c, v = get_coeffs_pivot_end(L, constraint[2])
                    A[idx_eq : idx_eq + 2, 4 * idx_segment - 4 : 4 * idx_segment] = c
                    values[idx_eq : idx_eq + 2] = v
                    idx_eq += 2
                else:
                    c, v = get_coeffs_pivot_center(L, constraint[2])
                    A[idx_eq : idx_eq + 4, 4 * idx_segment - 4 : 4 * idx_segment + 4] = c
                    values[idx_eq : idx_eq + 4] = v
                    idx_eq += 4

        return A, values

    def eval(self, x, derivative=0):
        idx_segment = np.clip(np.searchsorted(self.x0, x) - 1, 0, self.n_segments - 1)
        x_rel = x - self.x0[idx_segment]
        coeffs = self.coeffs
        M_deriv = np.array([[0, 0, 0, 0], [1, 0, 0, 0], [0, 2, 0, 0], [0, 0, 3, 0]])
        for _ in range(derivative):
            coeffs = coeffs @ M_deriv
        coeffs = coeffs[idx_segment]
        return coeffs[..., 0] + coeffs[..., 1] * x_rel + coeffs[..., 2] * x_rel**2 + coeffs[..., 3] * x_rel**3


constraints = [
    ("clamp", 0, 0, 0),
    ("pivot", 1.5, 1),
    ("pivot", 2, 2),
]

beam = Beam(constraints)
x_values = np.linspace(min(beam.x0), max(beam.x0), 2_000)
fig, axes = plt.subplots(4, 1, figsize=(6, 10), sharex=True)

for idx_deriv, ax in enumerate(axes):
    y_values = beam.eval(x_values, derivative=idx_deriv)
    # ax.set_title(f"Derivative {idx_deriv}")
    # ax.set_ylabel("y")
    # ax.set_xlabel("x")
    ax.grid()
    ax.plot(x_values, y_values)
    ax.set_ylabel("y" + "'" * idx_deriv)

for c in constraints:
    if c[0] == "clamp":
        dx = 0.05
        _, x, y, slope = c
        axes[0].plot([x - dx, x + dx], [y - slope * dx, y + slope * dx], color="red", alpha=0.5)
    elif c[0] == "pivot":
        _, x, y = c
        axes[0].plot(x, y, "o", color="blue", alpha=0.5)

# y1 = np.diff(y_values) / np.diff(x_values)
# y2 = np.diff(y1) / np.diff(x_values[:-1])
# y3 = np.diff(y2) / np.diff(x_values[:-2])
# axes[1].plot(x_values[:-1], y1)
# axes[2].plot(x_values[:-2], y2)
# axes[3].plot(x_values[:-3], y3)

# for ax in axes:
#     ax.axhline(0, color="black", zorder=0)
