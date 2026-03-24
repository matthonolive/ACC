import numpy as np

def to_rad(degree):
    return degree * np.pi / 180

def to_degree(rad):
    return rad * 180 / np.pi

def thomas_solve(a, b, c, d):
    '''
    Solve a tridiagonal system Ax = d using the Thomas algorithm.
 
    a - lower diagonal (length N, a[0] unused)
    b - main diagonal  (length N)
    c - upper diagonal (length N, c[-1] unused)
    d - right-hand side (length N)
 
    Returns x (length N).
    '''
    N = len(b)

    c_p = np.zeros(N)
    d_p = np.zeros(N)

    ### FORWARD SWEEP ###
    c_p [0] = c[0] / b[0]
    d_p [0] = d[0] / b[0]
    for i in range(1, N):
        m = b[i] - a[i] * c_p[i - 1]
        c_p[i] = c[i] / m
        d_p[i] = (d[i] - a[i] * d_p[i - 1]) / m 

    ### BACK SUBSTITUTION ###
    x = np.zeros(N)
    x[-1] = d_p[-1]
    for i in range(N - 2, -1, -1):
        x[i] = d_p[i] - c_p[i] * x[i + 1]

    return x


def FDTD_CN(t, dt, C, theta, S, K):
    '''
    Crank-Nicolson solver for the meridional diffusion model.
 
    t     - time grid
    dt    - time step
    C     - initial concentration grid (length N)
    theta - latitude grid in degrees (length N)
    S     - source term grid (N x len(t))
    K     - diffusion coefficient grid (length N)
    '''
 
    N = len(theta)
    theta_rad = to_rad(theta)
    dtheta = theta_rad[1] - theta_rad[0]

    ### lambda_{i+1/2} = K_{i+1/2} * cos(theta_{i+1/2}) / (cos(theta_i) * dtheta^2)
    lambda_plus = np.zeros(N)
    lambda_minus = np.zeros(N)

    for i in range(N-1):
        K_plus = (K[i] + K[i+1]) / 2
        K_minus = (K[i] + K[i-1]) / 2
        cos_plus = (np.cos(theta_rad[i]) + np.cos(theta_rad[i+1])) / 2
        cos_minus = (np.cos(theta_rad[i]) + np.cos(theta_rad[i-1]))/ 2
        cos_i = np.cos(theta_rad[i])

        lambda_plus[i] = K_plus * cos_plus / (cos_i * dtheta**2)
        lambda_minus[i] = K_minus * cos_minus / (cos_i * dtheta**2)

    #Build the tridiagonal matrix 
    L_lower = np.zeros(N)
    L_main = np.zeros(N)
    L_upper = np.zeros(N)
    for i in range(1, N-1):
        L_lower[i] = -dt/2 * lambda_minus[i]
        L_main[i]  = 1 + dt/2 * (lambda_plus[i] + lambda_minus[i])
        L_upper[i] = -dt/2 * lambda_plus[i]

        R_lower = dt/2 * lambda_minus[i]
        R_main = 1 - dt/2 * (lambda_plus[i] + lambda_minus[i])
        R_upper = dt/2 * lambda_plus[i]

    # No-flux boundary conditions: C[0] = C[1], C[-1] = C[-2]
    L_main[0] = 1
    L_upper[0] = -1
    L_lower[-1] = -1
    L_main[-1] = 1

    R_main[0] = 0
    R_upper[0] = 0
    R_lower[0] = 0
    R_lower[-1] = 0
    R_main[-1] = 0
    R_upper[-1] = 0

    # Time-stepping loop
    C_t = np.zeros((len(t), N))
    C_t[0, :] = C

    for n in range(1, len(t)):
        C_old = C_t[n-1, :]
        rhs = np.zeros(N)
        for i in range(1, N-1):
            rhs[i] = R_lower * C_old[i-1] + R_main * C_old[i] + R_upper * C_old[i+1] + dt * S[i, n]
    
        rhs[0] = 0
        rhs[-1] = 0

        C_t[n, :] = thomas_solve(L_lower, L_main, L_upper, rhs)
    
    return C_t

def main():
 
    C = np.ones(360) * 315
    theta = np.linspace(-89.5, 89.5, 360)
 
    K = np.ones(360) * 0.8
 
    dt = 1 / 12
    t = np.arange(0, 50, dt)
 
    S = np.zeros((360, len(t)))
    S[180, :] = 1000
 
    result = CN_Meridional(t, dt, C, theta, S, K)
 
    print("Final time step min/max:", result[-1].min(), result[-1].max())
    print("Shape:", result.shape)
 
 
if __name__ == "__main__":
    main()