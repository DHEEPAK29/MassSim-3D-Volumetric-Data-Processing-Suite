import numpy as np
import matplotlib.pyplot as plt

# 1. SETUP: 100-point signal (Input: 100 Floats)
N = 100
t = np.linspace(0, 1, N)
# Create a signal with two rhythms (5Hz and 20Hz) + random noise
raw_signal = 400 + 300 * np.sin(2 * np.pi * 5 * t) + 200 * np.sin(2 * np.pi * 20 * t)
raw_signal += np.random.normal(0, 30, N)

# 2. THE EVENT: Discrete Fourier Transform (DFT)
f_output = np.fft.fft(raw_signal)
frequencies = np.fft.fftfreq(N, d=t[1]-t[0])
magnitudes = np.abs(f_output) # Convert complex to float magnitude

# 3. VISUALIZATION
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 10))

# Plot 1: Time Domain (What you see)
ax1.plot(t, raw_signal, color='blue')
ax1.set_title("1. Time Domain (Raw 100 Floats)")
ax1.set_ylabel("Amplitude")

# Plot 2: Frequency Domain (What the computer detects)
# Only showing the first half (positive frequencies)
half_n = N // 2
ax2.stem(frequencies[:half_n], magnitudes[:half_n], basefmt=" ")
ax2.set_title("2. Frequency Domain (Full Magnitude Spectrum)")
ax2.set_ylabel("Strength")

# Plot 3: Compression (What we keep)
threshold = np.max(magnitudes) * 0.2
compressed = np.where(magnitudes > threshold, magnitudes, 0)
ax3.stem(frequencies[:half_n], compressed[:half_n], linefmt='red', markerfmt='ro', basefmt=" ")
ax3.set_title("3. Pre-Processing: Thresholding (Compressed Peaks)")
ax3.set_ylabel("Strength")

plt.tight_layout()
plt.show()


