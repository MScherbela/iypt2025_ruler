#%%

#Set all params

# 1) Get touch angle => get initial coefficients, y_end
x_end = get_touch_angle()

# 2) Integrate ODE with euler method => save coefficients at every step
# Use open end equations (y''(right) = 0) for the beam
# Leave x_end constant, only change y_end to be consistent with ball position

x_ball = 
v_ball = 0
dt = 1e-3


all_coeffs = []
ball_positions = []
for i in range(n_steps):
    coeffs = get_coeffs_with_open_end(x_end, y_end)
    Force_end_y = get_force(coeffs)
    Force_ball_x = angle_calculations(touch_angle) * Force_end_y

    v_ball += Force_ball_x * dt / m_ball
    x_ball += v_ball * dt

    all_coeffs.append(coeffs)
    ball_positions.append(x_ball)

all_coeffs = np.array(all_coeffs)
ball_positions = np.array(ball_positions)


# Debugging ideas:
# a) Log energies: bending energy, potential energy of load, kinetic energy of ball



# 3) Animation of the beam using the dumped coefficients

