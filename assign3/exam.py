import numpy as np

# Sample precision and recall values
precision = [0.9, 0.85, 0.8, 0.75, 0.7, 0.68, 0.65, 0.6, 0.55, 0.52, 0.5]
recall = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0]

# Interpolate precision values at 11 equally spaced recall levels
interp_recall = np.linspace(0, 1, 11)
interp_precision = np.interp(interp_recall, recall, precision)

# Calculate 11-point Average Precision
ap = np.mean(interp_precision)

print("11-point Average Precision:", ap)