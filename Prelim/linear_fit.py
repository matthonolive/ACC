import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# -----------------------------
# User settings
# -----------------------------
filename = "monthly.csv"  
sigma_y_value = 0.5   

# -----------------------------
# Define linear model
# -----------------------------
def linear_model(x, m, b):
    return m * x + b

# -----------------------------
# Load data
# -----------------------------

data = np.loadtxt(filename, delimiter=",")
x = data[:, 0]
y = data[:, 1]

# Uncertainty array
sigma_y = np.full_like(y, sigma_y_value, dtype=float)

# -----------------------------
# Fit
# -----------------------------
popt, pcov = curve_fit(
    linear_model,
    x,
    y,
    sigma=sigma_y,
    absolute_sigma=True
)

m_fit, b_fit = popt
m_err, b_err = np.sqrt(np.diag(pcov))

# Fitted values
y_fit = linear_model(x, m_fit, b_fit)

# Goodness of fit
residuals = y - y_fit
chi2 = np.sum((residuals / sigma_y) ** 2)
dof = len(x) - 2
reduced_chi2 = chi2 / dof

# -----------------------------
# Print results
# -----------------------------
print(f"Slope       = {m_fit:.6f} ± {m_err:.6f}")
print(f"Intercept   = {b_fit:.6f} ± {b_err:.6f}")
print(f"Chi^2       = {chi2:.2f}")
print(f"Reduced Chi^2 = {reduced_chi2:.2f}")

# -----------------------------
# Plot
# -----------------------------
plt.figure(figsize=(10, 6))
plt.errorbar(
    x, y,
    yerr=sigma_y,
    fmt='o',
    markersize=3,
    capsize=2,
    label='Data'
)

# Smooth line for fit
x_line = np.linspace(np.min(x), np.max(x), 1000)
y_line = linear_model(x_line, m_fit, b_fit)
plt.plot(x_line, y_line, label='Linear fit')

plt.xlabel("x")
plt.ylabel("y")
plt.title("Linear Fit with Constant Uncertainty")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()