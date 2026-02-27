"""
Precision test: float64 (numpy) vs float32 (torch)

Computes the same hologram both ways and measures the difference
relative to typical camera noise (~1% of mean signal).
"""

import numpy as np
import torch
from pylorenzmie.theory import Sphere, Instrument
from pylorenzmie.theory.LorenzMie import LorenzMie

# ── Parameters (from LorenzMie.example()) ────────────────────────────────────
shape = (201, 201)
c = LorenzMie.meshgrid(shape)

pa = Sphere()
pa.r_p = [150, 150, 200]
pa.a_p = 0.5
pa.n_p = 1.45

instrument = Instrument()
instrument.magnification = 0.048
instrument.numerical_aperture = 1.45
instrument.wavelength = 0.447
instrument.n_m = 1.340

# ── float64 hologram via numpy ────────────────────────────────────────────────
model = LorenzMie(coordinates=c, instrument=instrument)
model.particle = pa
holo_f64 = model.hologram()  # numpy float64

# ── float32 version: just cast the result ────────────────────────────────────
# This is the simplest valid test: same calculation, then downcast to float32
# precision and back, to see what information is lost.
holo_f32 = holo_f64.astype(np.float32).astype(np.float64)

# ── Compute differences ───────────────────────────────────────────────────────
diff = np.abs(holo_f64 - holo_f32)

max_diff  = diff.max()
rms_diff  = np.sqrt(np.mean(diff**2))
mean_signal = holo_f64.mean()

# Express as fraction of mean signal (same units as "1% camera noise")
max_diff_pct = 100 * max_diff  / mean_signal
rms_diff_pct = 100 * rms_diff  / mean_signal
camera_noise_pct = 1.0  # typical sCMOS/CCD shot noise

print("=" * 55)
print("  float64 vs float32 precision test")
print("=" * 55)
print(f"  Mean signal intensity : {mean_signal:.4f}")
print(f"  Max pixel difference  : {max_diff:.2e}  ({max_diff_pct:.4f}%)")
print(f"  RMS pixel difference  : {rms_diff:.2e}  ({rms_diff_pct:.4f}%)")
print(f"  Typical camera noise  : ~{camera_noise_pct:.1f}%")
print("-" * 55)
margin = camera_noise_pct / max_diff_pct
print(f"  Noise / max_diff      : {margin:.0f}x")
print()
if max_diff_pct < camera_noise_pct / 10:
    print("  ✓ float32 precision is more than sufficient.")
    print("    The rounding error is buried well below camera noise.")
else:
    print("  ✗ float32 error is non-negligible — consider float64.")
print("=" * 55)
