import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# --- [1] Run the Simulation Logic ---
size = 64
x = np.linspace(0, 1, size)
X, Y, Z = np.meshgrid(x, x, x)
# Input: X-axis pattern + Z-axis pattern
signal_3d = np.sin(2 * np.pi * 2 * X) + np.sin(2 * np.pi * 10 * Z)

# FFT Event
f_output = np.fft.fftshift(np.fft.fftn(signal_3d))
magnitude = np.abs(f_output)

# --- [2] Pre-Processing: Thresholding ---
# We only want to visualize the "Ingredients" (The Peaks)
# This mimics the "Compression" of 14GB -> 50,000 points
threshold = magnitude.max() * 0.5
z_peaks, y_peaks, x_peaks = np.where(magnitude > threshold)

# --- [3] Visualizing the 3D Frequency Domain ---
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# Scatter plot of the peaks
# The position represents the FREQUENCY (u, v, w)
scatter = ax.scatter(x_peaks, y_peaks, z_peaks, 
                     c=magnitude[z_peaks, y_peaks, x_peaks], 
                     cmap='viridis', s=50)

ax.set_title("3D Frequency Spectrum (Compressed Peaks)")
ax.set_xlabel("Frequency U (X-axis)")
ax.set_ylabel("Frequency V (Y-axis)")
ax.set_zlabel("Frequency W (Z-axis)")

plt.colorbar(scatter, label='Magnitude Strength')
plt.show()

