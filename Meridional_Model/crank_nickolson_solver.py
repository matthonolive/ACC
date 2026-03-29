import numpy as np
import pandas as pd

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

    for i in range(1, N-1):
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
    R_lower = np.zeros(N)
    R_main = np.zeros(N)
    R_upper = np.zeros(N)

    for i in range(1, N-1):
        L_lower[i] = -dt/2 * lambda_minus[i]
        L_main[i]  = 1 + dt/2 * (lambda_plus[i] + lambda_minus[i])
        L_upper[i] = -dt/2 * lambda_plus[i]

        R_lower[i] = dt/2 * lambda_minus[i]
        R_main[i] = 1 - dt/2 * (lambda_plus[i] + lambda_minus[i])
        R_upper[i] = dt/2 * lambda_plus[i]

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
            rhs[i] = R_lower[i] * C_old[i-1] + R_main[i] * C_old[i] + R_upper[i] * C_old[i+1] + dt * S[i, n]
    
        rhs[0] = 0
        rhs[-1] = 0

        C_t[n, :] = thomas_solve(L_lower, L_main, L_upper, rhs)
    
    return C_t

def load_monthly_source(csv_path, value_col="ppm_global_contribution_by_lat"):
    df = pd.read_csv(csv_path)

    # Build a monthly index
    df["date"] = pd.to_datetime(
        {
            "year": df["year"].astype(int),
            "month": df["month"].astype(int),
            "day": 1,
        }
    ).dt.to_period("M")

    # latitude x month table
    table = (
        df.pivot_table(
            index="latitude",
            columns="date",
            values=value_col,
            aggfunc="first"
        )
        .sort_index()
        .sort_index(axis=1)
    )

    if table.isna().any().any():
        raise ValueError("Missing latitude-month entries in source CSV.")

    theta = table.index.to_numpy(dtype=float)          # degrees
    dates = table.columns                              # pandas PeriodIndex
    dC_src = table.to_numpy(dtype=float)               # shape (N_lat, N_months)

    return theta, dates, dC_src

def export_latitude_timeseries(result, theta, dates, target_lat, out_csv):
    idx_lat = np.argmin(np.abs(theta - target_lat))
    actual_lat = theta[idx_lat]

    export_df = pd.DataFrame({
        "year": dates.year.astype(int),
        "month": dates.month.astype(int),
        "date": dates.to_timestamp(),
        "simulated_ppm": result[1:, idx_lat],
        "target_latitude": target_lat,
        "model_latitude": actual_lat,
    })

    export_df.to_csv(out_csv, index=False)
    return idx_lat, actual_lat

def main():
    # Read the monthly source from the CSV
    theta, dates, dC_src = load_monthly_source(
        "zonal_avg.csv",
        value_col="ppm_local_column_equiv"
    )

    N = len(theta)
    N_months = dC_src.shape[1]

    # Initial concentration and diffusion grid must match theta length
    C = 337.5 + 1.5 * np.sin(np.deg2rad(theta))
    K = np.ones(N) * 0.8

    # One timestep = one month
    dt = 1 / 12

    #Need yearly rate per month
    S = np.zeros((N, N_months + 1))
    S[:, 1:] = dC_src / dt * 0.55

    # Time array must match the source length
    t = np.arange(N_months + 1) * dt

    result = FDTD_CN(t, dt, C, theta, S, K)

    # print("Result shape:", result.shape)
    # print("Theta shape:", theta.shape)
    # print("Source shape:", S.shape)
    # print("First month:", dates[0])
    # print("Last month:", dates[-1])

    # # diagnostics
    # idx_ml = np.argmin(np.abs(theta - 19.5))
    # print("Nearest latitude to Mauna Loa:", theta[idx_ml])
    # print("Initial concentration there:", result[0, idx_ml])
    # print("Final concentration there:", result[-1, idx_ml])

    # print("South Pole:", result[-1, 0])
    # print("Final time step min/max:", result[-1].min(), result[-1].max())

    idx_lat, actual_lat = export_latitude_timeseries(
        result, theta, dates,
        target_lat=-14.25,
        out_csv="american_samoa_simulated.csv"
    )

    print("Requested latitude:", -14.25)
    print("Nearest model latitude:", actual_lat)
    print("Initial concentration there:", result[0, idx_lat])
    print("Final concentration there:", result[-1, idx_lat])
 
if __name__ == "__main__":
    main()