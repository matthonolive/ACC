'''
Matthew Honorio Oliveira, March 2026

This file contains an finite difference time domain (FDTD) solver for the meridional diffusion model.
'''

import numpy as np

def to_rad(degree):
    return degree * np.pi / 180

def to_degree(rad):
    return rad * 180 / np.pi

def FDTD_Meridional(t, dt, C, theta, S, K):
    '''
    FDTD solver for the meridional diffusion model.

    t - time grid
    dt - time step
    C - concentration grid
    theta - latitude grid
    K - diffusion coefficient grid
    S - source term grid
    '''

    C_t = np.array([C])
    dtheta = to_rad(theta[1]) - to_rad(theta[0])

    for n in range(1, len(t)):
        nu_C = np.zeros_like(C)
        for i in range(1, len(theta)-1):

            F_l = (K[i] + K[i+1])/2 * (np.cos(to_rad(theta[i])) + np.cos(to_rad(theta[i+1])))/2 * (C_t[n-1][i+1] - C_t[n-1][i])/dtheta
            F_r = (K[i] + K[i-1])/2 * (np.cos(to_rad(theta[i])) + np.cos(to_rad(theta[i-1])))/2 * (C_t[n-1][i] - C_t[n-1][i-1])/dtheta
            nu_C[i] = (dt/(np.cos(to_rad(theta[i])) * dtheta)) * (F_l - F_r) + C_t[n-1][i] + dt * S[i, n]
        
        nu_C[0] = nu_C[1]
        nu_C[-1] = nu_C[-2]
        
        C_t = np.append(C_t, [nu_C], axis=0)

    return C_t

def main(): 

    C = np.ones((360))*315
    theta = np.linspace(-89.5, 89.5, 360)

    K = np.ones((360))

    dt = 1/12

    t = np.arange(0, 100, dt)

    S = np.zeros((360, len(t)))

    S[180, :] = 1000

    #print(S)

    print(FDTD_Meridional(t, dt, C, theta, S, K))




if __name__ == "__main__":
    main()
