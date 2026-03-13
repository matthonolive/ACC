import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal


# Read CSV with no header
# Replace "data.csv" with your filename
df = pd.read_csv("monthly.csv", header=None, names=["x", "y"])

# Extract columns
x = df["x"].to_numpy(dtype=float)
y = df["y"].to_numpy(dtype=float)

# Compute sample spacing from first column
dx = np.mean(np.diff(x))   
fs = 1.0 / dx             


### 
# #remove DC offset before FFT
y_detrended = signal.detrend(y, type='linear')
y_trend = y - y_detrended

# quadratic detrend
# coeffs = np.polyfit(x, y, deg=2)
# trend_quad = np.polyval(coeffs, x)
# y_detrended = y - trend_quad

plt.plot(x, y_trend)
print(y_trend[1] - y_trend[0])
plt.show()

# FFT for real-valued data
Y = np.fft.rfft(y_detrended)
freqs = np.fft.rfftfreq(len(y_detrended), d=dx)

# Amplitude spectrum
amplitude = (2.0 / len(y_detrended)) * np.abs(Y)

# Find peaks in amplitude spectrum
peaks, properties = signal.find_peaks(amplitude, height=0.1)   # adjust height as needed

# Print peak frequencies
print("Peak frequencies:")
for i in peaks:
    print(f"f = {freqs[i]:.6f}, amplitude = {amplitude[i]:.6f}")

# Plot original signal
plt.figure(figsize=(10, 4))
plt.plot(x, y, marker='o')
plt.xlabel("Sample / Time")
plt.ylabel("Signal")
plt.title("Input Signal")
plt.grid(True)
plt.tight_layout()
plt.show()

# Plot FFT magnitude
plt.figure(figsize=(10, 4))
plt.plot(freqs, amplitude, marker='o')
plt.xlabel("Frequency")
plt.ylabel("Amplitude")
plt.title("FFT Magnitude Spectrum")
plt.grid(True)
plt.tight_layout()
plt.show()