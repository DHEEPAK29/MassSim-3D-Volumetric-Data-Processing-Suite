# MassSim-3D: Volumetric Data Processing Suite

MassSim-3D is a high-performance simulation engine designed to handle **mass-scale volumetric datasets** (e.g., $1000^3$ voxel grids). It utilizes **n-Dimensional Discrete Fourier Transforms (nDFT)** to compress and extract features from raw spatial signals.

### Project Purpose
In large-scale physical simulations, raw data often exceeds 14 GB per snapshot. This module serves as a **Preprocessing Layer** that:
1.  Converts raw 3D Floats into the Frequency Domain.
2.  Filters noise using **Complex Magnitude Thresholding**.
3.  Reduces data footprint from Gigabytes to Megabytes by storing only "Resonant Peaks."

### The "Fourier Event" Logic
For every 3D coordinate $(x,y,z)$, the engine applies a **Complex Wave Correlation**:
$$X[k] = \sum_{n=0}^{N-1} x[n] \cdot e^{-i 2\pi \frac{kn}{N}}$$
*   **Input:** 1D/2D/3D Float arrays (Time/Spatial Domain).
*   **Process:** Multiplies samples by a complex exponential to detect periodic resonance.
*   **Output:** Complex numbers representing **Magnitude** (Strength) and **Phase** (Timing).

### Integration Code (Python)

```python
import numpy as np

def process_mass_volume(data_volume, threshold_percent=0.01):
    """
    Simulates the Fourier 'Event' on a mass 3D dataset.
    Converts 3D Floats -> 3D Complex -> Compressed Sparse Peaks.
    """
    # 1. Employment: 3D Fast Fourier Transform
    # Transforms 14GB of raw floats into frequency space
    f_space = np.fft.fftn(data_volume)
    
    # 2. Thresholding: Extract Magnitude
    # Convert Complex (Real + Imag) to Float (Magnitude)
    magnitude = np.abs(f_space)
    max_val = np.max(magnitude)
    
    # 3. Compression: Remove data below certain threshold
    # Logic: If magnitude < threshold, set to 0 (Hard Thresholding)
    threshold = max_val * threshold_percent
    sparse_indices = np.where(magnitude > threshold)
    
    # 4. Sparse Output: Only store Coordinates and Complex Amplitudes
    compressed_data = {
        "coords": list(zip(*sparse_indices)),
        "values": f_space[sparse_indices]
    }
    
    return compressed_data

# Example: 100-point 1D Time Series Integration
time_series = np.random.uniform(0, 1000, 100)
compressed = process_mass_volume(time_series)
print(f"Original Samples: {len(time_series)} | Retained Peaks: {len(compressed['values'])}")

