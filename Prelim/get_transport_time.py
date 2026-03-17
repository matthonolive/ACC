import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal


# Read CSV with no header
df = pd.read_csv("monthly_mlo_dat.csv", header=None, names=["x", "y"])
df_sp = pd.read_csv("monthly_spo_dat.csv", header=None, names=["x", "y"])

# Extract columns
x = df["x"].to_numpy(dtype=float)
y = df["y"].to_numpy(dtype=float)

# Compute sample spacing from first column
dx = np.mean(np.diff(x))   
fs = 1.0 / dx             


coeffs = np.polyfit(x, y, deg=2)
trend_quad = np.polyval(coeffs, x)
y_detrended = y - trend_quad

# FFT for real-valued data
Y = np.fft.rfft(y_detrended)
freqs = np.fft.rfftfreq(len(y_detrended), d=dx)

cutoff = 0.07
Y_filtered = Y.copy()
Y_filtered[ freqs > cutoff ] = 0.0

y_detrended_filtered = np.fft.irfft(Y_filtered, n=len(y_detrended))

y_filtered = y_detrended_filtered + trend_quad

amplitude = (2.0 / len(y_detrended)) * np.abs(Y_filtered)

# Plot FFT magnitude

# plt.figure(figsize=(10, 4))
# plt.plot(x, y_filtered, marker='o')
# plt.xlabel("Frequency")
# plt.ylabel("Amplitude")
# plt.title("FFT Magnitude Spectrum")
# plt.grid(True)
# plt.tight_layout()
# plt.show()

# plt.figure(figsize=(10, 4))
# plt.plot(freqs, amplitude, marker='o')
# plt.xlabel("Frequency")
# plt.ylabel("Amplitude")
# plt.title("FFT Magnitude Spectrum")
# plt.grid(True)
# plt.tight_layout()
# plt.show()

# Extract columns
x2 = df_sp["x"].to_numpy(dtype=float)
y2 = df_sp["y"].to_numpy(dtype=float)

# Compute sample spacing from first column
dx2 = np.mean(np.diff(x2))   
fs2 = 1.0 / dx2         


coeffs2 = np.polyfit(x2, y2, deg=2)
trend_quad2 = np.polyval(coeffs2, x2)
y_detrended2 = y2 - trend_quad2

# FFT for real-valued data
Y2 = np.fft.rfft(y_detrended2)
freqs = np.fft.rfftfreq(len(y_detrended2), d=dx2)

cutoff = 0.07
Y_filtered2 = Y2.copy()
Y_filtered2[ freqs > cutoff ] = 0.0

y_detrended_filtered2 = np.fft.irfft(Y_filtered2, n=len(y_detrended2))

y_filtered2 = y_detrended_filtered2 + trend_quad2

# Amplitude spectrum
amplitude2 = (2.0 / len(y_detrended2)) * np.abs(Y_filtered2)


# Plot FFT magnitude
# plt.figure(figsize=(10, 4))
# plt.plot(freqs, amplitude2, marker='o')
# plt.xlabel("Frequency")
# plt.ylabel("Amplitude")
# plt.title("FFT Magnitude Spectrum")
# plt.grid(True)
# plt.tight_layout()
# plt.show()


y_nu = y_filtered - y_filtered2

plt.figure(figsize=(10, 4))
plt.plot(x, y_nu, marker='o')
plt.xlabel("months")
plt.ylabel("CO2 (ppm)")
plt.title("CO2 difference between MLO and SPO")
plt.show()