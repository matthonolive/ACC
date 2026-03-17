import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.odr import ODR, Model, RealData

# Read CSV with no header
df = pd.read_csv("gradient_vs_anthro.csv", header=None)

# Extract columns (0-based indexing)
gradient = pd.to_numeric(df.iloc[:, 2], errors="coerce").to_numpy()       # 3rd column
emissions = pd.to_numeric(df.iloc[:, 3], errors="coerce").to_numpy()      # 4th column
emissions_unc = pd.to_numeric(df.iloc[:, 4], errors="coerce").to_numpy()  # 5th column

# Constant uncertainty in gradient
gradient_unc = np.sqrt(0.5**2 + 0.5**2)

# Remove rows with missing values
mask = np.isfinite(gradient) & np.isfinite(emissions) & np.isfinite(emissions_unc)
gradient = gradient[mask]
emissions = emissions[mask]
emissions_unc = emissions_unc[mask]

# Constant y-uncertainty array
gradient_unc_arr = np.full_like(gradient, gradient_unc, dtype=float)

# Linear model: y = m*x + b
def linear_func(B, x):
    return B[0] * x + B[1]

model = Model(linear_func)
data = RealData(emissions, gradient, sx=emissions_unc, sy=gradient_unc_arr)

# Initial guess from ordinary least squares
m0, b0 = np.polyfit(emissions, gradient, 1)

odr = ODR(data, model, beta0=[m0, b0])
output = odr.run()

m, b = output.beta
m_err, b_err = output.sd_beta

# Fitted values at data points
gradient_fit = linear_func(output.beta, emissions)

# Compute R^2
ss_res = np.sum((gradient - gradient_fit)**2)
ss_tot = np.sum((gradient - np.mean(gradient))**2)
r2 = 1 - ss_res / ss_tot

# Fit line for plotting
xfit = np.linspace(emissions.min(), emissions.max(), 400)
yfit = m * xfit + b

# Plot
plt.figure(figsize=(8, 6))
plt.errorbar(
    emissions,
    gradient,
    xerr=emissions_unc,
    yerr=gradient_unc_arr,
    fmt='o',
    capsize=3,
    markersize=5,
    elinewidth=1,
    label="Data"
)

plt.plot(xfit, yfit, label=f"Linear fit ($R^2$ = {r2:.3f})")

plt.xlabel("Anthropogenic emissions")
plt.ylabel("CO$_2$ gradient")
plt.title("CO$_2$ Gradient vs Anthropogenic Emissions")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

print(f"Slope = {m:.6g} ± {m_err:.2g}")
print(f"Intercept = {b:.6g} ± {b_err:.2g}")
print(f"R^2 = {r2:.6f}")
print(f"Gradient uncertainty used for every point = {gradient_unc:.6f}")
