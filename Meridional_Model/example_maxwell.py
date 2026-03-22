import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# -----------------------------
# Physical constants
# -----------------------------
eps0 = 8.854187817e-12
mu0  = 4e-7 * np.pi
c0   = 1.0 / np.sqrt(eps0 * mu0)

# -----------------------------
# Simulation parameters
# -----------------------------
Nx = 400                 # number of spatial cells
dx = 1e-3                # spatial step [m]
S  = 0.99                # Courant number
dt = S * dx / c0         # time step [s]
Nt = 600                 # number of time steps

# Fields
E = np.zeros(Nx)         # E field at integer points
H = np.zeros(Nx - 1)     # H field at half-integer points

# Material parameters (free space everywhere)
eps = eps0 * np.ones(Nx)
mu  = mu0  * np.ones(Nx - 1)

# Precompute update coefficients
ce = dt / (eps * dx)
ch = dt / (mu  * dx)

# -----------------------------
# Source parameters
# -----------------------------
source_position = Nx // 4
t0 = 40
spread = 12

def gaussian_pulse(n):
    return np.exp(-0.5 * ((n - t0) / spread) ** 2)

# -----------------------------
# Simple absorbing boundary storage
# (1st-order Mur-like very rough boundary substitute)
# Here just using old endpoint values as a minimal hack.
# -----------------------------
E_left_old = 0.0
E_right_old = 0.0

# -----------------------------
# Storage for animation
# -----------------------------
frames = []

for n in range(Nt):
    # Update H from E
    H += ch * (E[1:] - E[:-1])

    # Update E from H
    E[1:-1] += ce[1:-1] * (H[1:] - H[:-1])

    # Add soft source to E
    E[source_position] += gaussian_pulse(n)

    # Very simple absorbing boundary approximation
    E[0] = E_left_old
    E[-1] = E_right_old
    E_left_old = E[1]
    E_right_old = E[-2]

    if n % 4 == 0:
        frames.append(E.copy())

# -----------------------------
# Plot animation
# -----------------------------
fig, ax = plt.subplots()
line, = ax.plot(frames[0])
ax.set_ylim(-1.2, 1.2)
ax.set_xlim(0, Nx - 1)
ax.set_xlabel("Grid index")
ax.set_ylabel("E field")
ax.set_title("1D FDTD: Gaussian pulse propagation")

def update(frame):
    line.set_ydata(frames[frame])
    ax.set_title(f"1D FDTD: step {frame * 4}")
    return line,

ani = FuncAnimation(fig, update, frames=len(frames), interval=30, blit=True)
plt.show()