import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import constants

CROP = 300


def load_data(haystack_path, needle_path, correlation_path):
    with fits.open(haystack_path) as hdul:
        haystack = hdul[0].data.astype(np.float64)

    with fits.open(needle_path) as hdul:
        needle = hdul[0].data.astype(np.float64)

    with fits.open(correlation_path) as hdul:
        header = hdul[0].header
        peak_x = header['PEAK_X']
        peak_y = header['PEAK_Y']
        peak_val = header['PEAK_VAL']

    return haystack, needle, peak_x, peak_y, peak_val


def crop_haystack(haystack, peak_x, peak_y, size):
    H, W = haystack.shape
    half = size // 2

    y0 = int(round(peak_y)) - half
    x0 = int(round(peak_x)) - half
    y1 = y0 + size
    x1 = x0 + size

    # Clamp to image bounds
    y0c, y1c = max(0, y0), min(H, y1)
    x0c, x1c = max(0, x0), min(W, x1)

    crop = np.zeros((size, size), dtype=haystack.dtype)
    cy0 = y0c - y0
    cx0 = x0c - x0
    crop[cy0:cy0 + (y1c - y0c), cx0:cx0 + (x1c - x0c)] = haystack[y0c:y1c, x0c:x1c]
    return crop


def plot(haystack_crop, needle, peak_x, peak_y, peak_val):
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    vmin = min(haystack_crop.min(), needle.min())
    vmax = max(haystack_crop.max(), needle.max())

    axes[0].imshow(haystack_crop, cmap='viridis', vmin=vmin, vmax=vmax, origin='upper')
    axes[0].set_title(f"Haystack crop {CROP}×{CROP}\n"
                      f"centred at ({peak_x:.1f}, {peak_y:.1f})")
    axes[0].axis('off')

    axes[1].imshow(needle, cmap='viridis', vmin=vmin, vmax=vmax, origin='upper')
    axes[1].set_title(f"Needle  {needle.shape[1]}×{needle.shape[0]}\n"
                      f"peak val = {peak_val:.4f}")
    axes[1].axis('off')

    plt.tight_layout()
    plt.show()


def main(haystack_path=None, needle_path=None, correlation_path=None):
    if haystack_path is None:
        haystack_path = os.path.join(constants.FITS_DIR, "haystack_0001.fits")
    if needle_path is None:
        needle_path = os.path.join(constants.FITS_DIR, "needle_0001.fits")
    if correlation_path is None:
        correlation_path = os.path.join(constants.FITS_DIR, "correlation_0001.fits")

    haystack, needle, peak_x, peak_y, peak_val = load_data(
        haystack_path, needle_path, correlation_path
    )

    haystack_crop = crop_haystack(haystack, peak_x, peak_y, CROP)
    plot(haystack_crop, needle, peak_x, peak_y, peak_val)


if __name__ == "__main__":
    main()
