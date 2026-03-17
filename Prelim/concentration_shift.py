import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ----------------------------
# Load data
# ----------------------------
df_mlo = pd.read_csv("monthly_mlo_dat.csv", header=None, names=["x", "y"])
df_spo = pd.read_csv("monthly_spo_dat.csv", header=None, names=["x", "y"])

x_mlo = df_mlo["x"].to_numpy(dtype=float)
y_mlo = df_mlo["y"].to_numpy(dtype=float)

x_spo = df_spo["x"].to_numpy(dtype=float)
y_spo = df_spo["y"].to_numpy(dtype=float)

# ----------------------------
# Put on common time grid
# ----------------------------
x_min = max(x_mlo.min(), x_spo.min())
x_max = min(x_mlo.max(), x_spo.max())

dx = 1  # monthly spacing in years
x_common = np.arange(x_min, x_max, dx)

y_mlo_i = np.interp(x_common, x_mlo, y_mlo)
y_spo_i = np.interp(x_common, x_spo, y_spo)

# ----------------------------
# Remove polynomial trend
# ----------------------------
coef_mlo = np.polyfit(x_common, y_mlo_i, 2)
coef_spo = np.polyfit(x_common, y_spo_i, 2)

trend_mlo = np.polyval(coef_mlo, x_common)
trend_spo = np.polyval(coef_spo, x_common)

y_mlo_d = y_mlo_i - trend_mlo
y_spo_d = y_spo_i - trend_spo

# ----------------------------
# Low-pass filter in Fourier space
# ----------------------------
def lowpass_fft(y, dx, cutoff):
    Y = np.fft.rfft(y)
    freqs = np.fft.rfftfreq(len(y), d=dx)
    Y[freqs > cutoff] = 0.0
    return np.fft.irfft(Y, n=len(y))

cutoff = 0.07   # cycles/year; adjust as needed
y_mlo_f = lowpass_fft(y_mlo_d, dx, cutoff)
y_spo_f = lowpass_fft(y_spo_d, dx, cutoff)

# Add trend back to get smoothed monotonic curves
y_mlo_smooth = y_mlo_f + trend_mlo
y_spo_smooth = y_spo_f + trend_spo

# ----------------------------
# Equal-concentration matching
# ----------------------------
cmin = max(y_mlo_smooth.min(), y_spo_smooth.min())
cmax = min(y_mlo_smooth.max(), y_spo_smooth.max())

# stay away from edges a little
c_values = np.linspace(cmin + 0.2, cmax - 0.2, 200)

# Sort by concentration in case of tiny non-monotonic wiggles
mlo_sort = np.argsort(y_mlo_smooth)
spo_sort = np.argsort(y_spo_smooth)

t_mlo = np.interp(c_values, y_mlo_smooth[mlo_sort], x_common[mlo_sort])
t_spo = np.interp(c_values, y_spo_smooth[spo_sort], x_common[spo_sort])

lags_months = t_spo - t_mlo
lags_years = 12 * lags_months

print("Mean lag (months):", np.mean(lags_months))
print("Median lag (months):", np.median(lags_months))
print("Std dev (months):", np.std(lags_months))

# ----------------------------
# Plot lag vs concentration
# ----------------------------
plt.figure(figsize=(8,5))
plt.plot(c_values, lags_months)
plt.xlabel("CO2 concentration")
plt.ylabel("Lag (months)")
plt.title("South Pole lag relative to Mauna Loa")
plt.grid(True)
plt.tight_layout()
plt.show()

# ----------------------------
# Plot smoothed curves
# ----------------------------
plt.figure(figsize=(10,5))
plt.plot(x_common, y_mlo_smooth, label="Mauna Loa")
plt.plot(x_common, y_spo_smooth, label="South Pole")
plt.xlabel("Year")
plt.ylabel("CO2")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
