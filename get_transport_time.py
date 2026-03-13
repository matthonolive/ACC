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
