import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import map_coordinates
from astropy.io import fits
import constants

PAIR_NUM = "0001"

base           = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data_generation", "dataset", f"pair_{PAIR_NUM}")
candidate_path = os.path.join(base, f"candidate_needle_{PAIR_NUM}.fits")
needle_path    = os.path.join(base, f"needle_{PAIR_NUM}.fits")

with fits.open(candidate_path) as f:
    candidate = f[0].data.astype(np.float64)
with fits.open(needle_path) as f:
    needle = f[0].data.astype(np.float64)

N_ANGLES = 2048   # angular bins — higher = finer resolution

candidate -= candidate.mean()
needle    -= needle.mean()

# Hann window to suppress spectral leakage
H, W  = candidate.shape
hann  = np.outer(np.hanning(H), np.hanning(W))


def fft_magnitude(img):
    mag = np.abs(np.fft.fftshift(np.fft.fft2(img * hann)))
    # High-pass: zero out centre (low frequencies) — rotation signal lives at mid/high freqs
    cy, cx  = mag.shape[0] // 2, mag.shape[1] // 2
    r_inner = min(cy, cx) * 0.1   # mask inner 10%
    yy, xx  = np.ogrid[:mag.shape[0], :mag.shape[1]]
    mask    = (xx - cx)**2 + (yy - cy)**2 < r_inner**2
    mag[mask] = 0
    return mag


def to_log_polar(mag, n_angles=N_ANGLES, n_radii=None):
    H, W    = mag.shape
    cy, cx  = H / 2, W / 2
    max_r   = min(cx, cy)
    n_radii = n_radii or int(max_r)

    log_r  = np.linspace(0, np.log(max_r), n_radii)
    angles = np.linspace(0, 2 * np.pi, n_angles, endpoint=False)
    r      = np.exp(log_r)

    rr, aa = np.meshgrid(r, angles, indexing='ij')
    xs     = cx + rr * np.cos(aa)
    ys     = cy + rr * np.sin(aa)

    return map_coordinates(mag, [ys, xs], order=1, mode='constant', cval=0)


# ── Log-polar phase correlation ───────────────────────────────────────────────
mag_candidate = fft_magnitude(candidate)
mag_needle    = fft_magnitude(needle)

lp_candidate  = to_log_polar(mag_candidate)
lp_needle     = to_log_polar(mag_needle)

lp_candidate -= lp_candidate.mean()
lp_needle    -= lp_needle.mean()

F1    = np.fft.fft2(lp_candidate)
F2    = np.fft.fft2(lp_needle)
cross = F1 * np.conj(F2)
cross /= np.abs(cross).clip(1e-10)
corr  = np.fft.fftshift(np.fft.ifft2(cross).real)

# Peak + subpixel parabolic fit
n_radii, n_angles = lp_candidate.shape
yi, xi  = np.unravel_index(np.argmax(corr), corr.shape)

f_neg, f_0, f_pos = corr[yi, xi-1], corr[yi, xi], corr[yi, xi+1]
a  = (f_pos + f_neg - 2*f_0) / 2
b  = (f_pos - f_neg)          / 2
dx = -b / (2 * a) if a != 0 else 0.0

d_angle_idx = (xi + dx) - n_angles // 2
angle_deg   = d_angle_idx * (360.0 / n_angles)
print(f"Detected rotation: {angle_deg:.2f} deg")

if 'NANGLE' in fits.getheader(needle_path):
    true_angle = fits.getheader(needle_path)['NANGLE']
    print(f"True rotation:     {true_angle:.2f} deg")
    print(f"Error:             {abs(angle_deg - true_angle):.2f} deg")

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 4, figsize=(18, 5))

axes[0].imshow(candidate, cmap='viridis', origin='lower')
axes[0].set_title("Candidate needle (from haystack)")
axes[0].axis('off')

axes[1].imshow(needle, cmap='viridis', origin='lower')
axes[1].set_title("Needle (rotated)")
axes[1].axis('off')

axes[2].imshow(lp_candidate, cmap='magma', origin='lower', aspect='auto')
axes[2].set_title("Log-polar FFT magnitude (candidate)")
axes[2].set_xlabel("Angle index")
axes[2].set_ylabel("Log radius index")

im = axes[3].imshow(corr, cmap='magma', origin='lower', aspect='auto')
axes[3].scatter([xi], [yi], c='red', s=40, marker='+', linewidths=1.5)
axes[3].set_title(f"Log-polar correlation  (detected {angle_deg:.2f}°)")
axes[3].set_xlabel("Angle index")
axes[3].set_ylabel("Log radius index")
plt.colorbar(im, ax=axes[3])

plt.tight_layout()
plt.show()
