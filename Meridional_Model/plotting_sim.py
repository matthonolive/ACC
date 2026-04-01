import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

OBS_UNCERTAINTY_PPM = 0.2

def main():
    sim_df = pd.read_csv("canada_simulated.csv")
    sim_df["date"] = pd.to_datetime(sim_df["date"])

    obs_df = pd.read_csv(
        "canada.csv",
        header=None,
        names=["year", "month", "observed_ppm"]
    )

    obs_df["date"] = pd.to_datetime(
        dict(year=obs_df["year"], month=obs_df["month"], day=1)
    )

    # Merge without bringing in sim_df["date"], so only one date column remains
    df = pd.merge(
        obs_df,
        sim_df[["year", "month", "simulated_ppm"]],
        on=["year", "month"],
        how="inner"
    ).sort_values("date").reset_index(drop=True)

    if df.empty:
        raise ValueError("No overlapping months between observation and simulation CSVs.")

    df["error"] = df["simulated_ppm"] - df["observed_ppm"]
    df["abs_error"] = df["error"].abs()

    df["obs_lower"] = df["observed_ppm"] - OBS_UNCERTAINTY_PPM
    df["obs_upper"] = df["observed_ppm"] + OBS_UNCERTAINTY_PPM

    mae = df["abs_error"].mean()
    rmse = np.sqrt(np.mean(df["error"]**2))
    bias = df["error"].mean()
    max_abs_error = df["abs_error"].max()
    corr = df["simulated_ppm"].corr(df["observed_ppm"])

    print(f"Matched months: {len(df)}")
    print(f"MAE: {mae:.4f} ppm")
    print(f"RMSE: {rmse:.4f} ppm")
    print(f"Bias (sim - obs): {bias:.4f} ppm")
    print(f"Max abs error: {max_abs_error:.4f} ppm")
    print(f"Correlation: {corr:.4f}")

    df.to_csv("alaska_comparison.csv", index=False)

    plt.figure(figsize=(11, 6))
    plt.plot(df["date"], df["observed_ppm"], label="Observed (Alert, Canada)")
    plt.plot(df["date"], df["simulated_ppm"], label="Simulated (diffusion model)")
    plt.xlabel("Date", fontsize=14)
    plt.ylabel("CO$_2$ (ppm)", fontsize=14)
    plt.title("Canada: observed vs simulated CO$_2$", fontsize=14)
    plt.grid(True)
    plt.legend(fontsize=14)
    plt.tight_layout()

    # Dotted uncertainty lines around observations
    plt.plot(
        df["date"], df["obs_upper"],
        linestyle=":", linewidth=1.5,
        label=f"Observed + {OBS_UNCERTAINTY_PPM:.1f} ppm"
    )
    plt.plot(
        df["date"], df["obs_lower"],
        linestyle=":", linewidth=1.5,
        label=f"Observed - {OBS_UNCERTAINTY_PPM:.1f} ppm"
    )

    plt.show()

    plt.figure(figsize=(11, 4))
    plt.plot(df["date"], df["error"])
    plt.axhline(0.0, linestyle="--")
    plt.xlabel("Date", fontsize=14)
    plt.ylabel("Simulated - Observed (ppm)", fontsize=14)
    plt.title("Residuals", fontsize=14)
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()