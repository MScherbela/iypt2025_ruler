#imports
import numpy as np
from ruler_equations import get_touch_angle, return_coeffs, evaluate_polynomial, get_bending_energy
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle

x_dot_values = []
param_values = []

for i in np.arange (0.0,1.5, 0.01):
    print(i)
    m_o = i
    param_values.append(i)

    # properties of the ruler
    L1 = 0.13
    L2 = 0.17
    tilt = 0
    phi = tilt * np.pi / 180
    ruler_width = 0.03
    ruler_height = 0.003
    E_modulus = 2.3 * 10**9
    I_moment_of_inertia = ruler_width * ruler_height**3 / 12
    EI = E_modulus * I_moment_of_inertia

    # properties of the projectile
    x_ball = 0.133
    v_ball = 0.0
    v_ball_max = 0.0
    m_ball = 0.05
    r_ball = 0.01
    w = r_ball * 2
    A = 2 * r_ball * w
    V = r_ball**2 * np.pi * w
    Cd = 1
    rho = 1.293

    # Friction coefficients
    mu_static = 0.10
    mu_kinetic = 0.05
    mu_rolling = 0.0

    # force properties
    #m_o = 0.25
    g = 9.81
    x_load = 0.08
    force = m_o * 9.81
    load = force / EI

    # 1) Get touch angle => get initial coefficients, y_end
    touch_angle, x_end, y_end, slope_end = get_touch_angle(EI, x_load, force, x_ball, r_ball,load)
    coeffs_right = return_coeffs( x_end, y_end, load, x_load, slope_end)[4:]


    # 2) Integrate ODE with euler method => save coefficients at every step
    # Use open end equations (y''(right) = 0) for the beam
    # Leave x_end constant, only change y_end to be consistent with ball position
    n_steps = 1000
    n_current = 0
    dt = 1e-3

    all_coeffs = []
    ball_positions = []
    E_pots = []
    time = []
    velocity = []
    
    for i in range(n_steps):
        if(x_ball <L1+r_ball): y_end = np.sqrt(r_ball**2-(x_end-x_ball)**2)+r_ball
        else: y_end = 0.0

        theta = touch_angle
        phi = np.abs(np.arctan((r_ball-y_end)/(x_ball - x_end)))
        gamma = np.pi/2 - (theta+phi)
        coeffs = return_coeffs( x_end, y_end, load, x_load, slope_end)[4:]

        Force_end_y = evaluate_polynomial(coeffs_right, x_end, 3) * E_modulus * I_moment_of_inertia
        if(x_ball > 0.14): 
            Force_end_y = 0.0
            coeffs = ([0.0,0.0,0.0,0.0])
            theta = 0.0
        
        Force_end_n = Force_end_y * np.cos(touch_angle)
        Force_ball_proj = Force_end_n * np.cos(gamma)
        Force_ball_x = Force_ball_proj * np.cos(phi)
        Force_ball_y = Force_ball_proj * np.sin(phi)

        if(v_ball == 0.0): mu = mu_static
        elif(x_ball < 0.14): mu = mu_kinetic
        else: mu = mu_rolling
        FR1 = -mu *Force_ball_y*np.cos(theta)
        FR2 = -mu *(Force_ball_y+m_ball*g)
        Fd = -0.5 * Cd * A * v_ball**2

        F = Force_ball_x + FR1 + FR2 +Fd
        a_ball = F / m_ball
        if(F>0.0): v_ball += a_ball* dt 
        elif(v_ball > 0.0): v_ball += a_ball* dt 
        else: v_ball += 0.0

        E_left = get_bending_energy(coeffs_right, 0, x_load, EI)
        E_right = get_bending_energy(coeffs_right, x_load, x_end, EI)
        E_load = force * evaluate_polynomial(coeffs_right, x_load, 0)
        E = E_left + E_right + E_load

        if v_ball > v_ball_max: v_ball_max = v_ball
        all_coeffs.append(coeffs)
        ball_positions.append(x_ball)
        E_pots.append(E)
        velocity.append(v_ball)
        time.append(n_current*dt)
        n_current += 1

        if(x_ball > 0.17): break

    all_coeffs = np.array(all_coeffs)
    ball_positions = np.array(ball_positions)
    E_pots = np.array(E_pots)
    time = np.array(time)
    velocity=np.array(velocity)
    x_dot_values.append(v_ball_max)

param_values = np.array(param_values)
x_dot_values = np.array(x_dot_values)

# 3) plotting relevant data
plt.plot(time, velocity)
plt.plot(time, E_pots)
plt.plot(time, ball_positions)
plt.show()

# 4) Animation of the beam using the dumped coefficients
# Plotting  rang
fig, ax = plt.subplots()
ax.set_xlim(0, 0.2)
ax.set_ylim(-0.01, 0.06)
ax.set_aspect('equal')

# Initial conditions
beam_line, = ax.plot([], [], lw=2)
ball_patch = Circle((0, 0), r_ball, color='red')
ax.add_patch(ball_patch)

# x-values for determination y(x)
x_vals = np.linspace(0, x_end, 200)

# Initialisation
def init():
    beam_line.set_data([], [])
    ball_patch.center = (0, 0)
    return beam_line, ball_patch

# Update function
def update(i):
    # Beam modelling
    coeffs = all_coeffs[i]
    y_vals = evaluate_polynomial(coeffs, x_vals, 0)
    beam_line.set_data(x_vals, y_vals)

    # Position of the Ball
    x = ball_positions[i]
    ball_patch.center = (x, r_ball)

    return beam_line, ball_patch

# Animation
ani = animation.FuncAnimation(fig, update, frames=len(time),
                              init_func=init, blit=True, interval=100, repeat = True)
plt.show()

# 5) Parameter plotting

plt.plot(param_values,x_dot_values)
plt.show()