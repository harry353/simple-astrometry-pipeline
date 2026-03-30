import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.wcs import WCS

PAIR_NUM = "0001"

base              = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data_generation", "dataset", f"pair_{PAIR_NUM}")
needle_path       = os.path.join(base, f"needle_{PAIR_NUM}.fits")
corrected_path    = os.path.join(base, f"detranslated_needle_{PAIR_NUM}.fits")
ground_truth_path = os.path.join(base, f"needle_{PAIR_NUM}_ground_truth.fits")

with fits.open(needle_path) as f:
    image        = f[0].data
    wcs_original = WCS(f[0].header)

with fits.open(corrected_path) as f:
    wcs_corrected = WCS(f[0].header)

with fits.open(ground_truth_path) as f:
    wcs_gt = WCS(f[0].header)


def draw_grid(wcs_source, wcs_display, image_shape, ax, color, label, n_lines=6):
    H, W = image_shape
    t    = np.linspace(-50, max(H, W) + 50, 500)
    first = True
    for px in np.linspace(0, W, n_lines):
        ra, dec = wcs_source.pixel_to_world_values(np.full(500, px), t)
        disp_x, disp_y = wcs_display.world_to_pixel_values(ra, dec)
        ax.plot(disp_x, disp_y, color=color, linewidth=1.0, alpha=0.85,
                label=label if first else None)
        first = False
    for py in np.linspace(0, H, n_lines):
        ra, dec = wcs_source.pixel_to_world_values(t, np.full(500, py))
        disp_x, disp_y = wcs_display.world_to_pixel_values(ra, dec)
        ax.plot(disp_x, disp_y, color=color, linewidth=1.0, alpha=0.85)


fig, axes = plt.subplots(1, 2, figsize=(14, 7))

for ax, wcs_shown, title in [
    (axes[0], wcs_original,  "Original needle WCS"),
    (axes[1], wcs_corrected, "Corrected needle WCS"),
]:
    ax.imshow(image, cmap='viridis', origin='lower')
    draw_grid(wcs_shown,  wcs_shown, image.shape, ax, color='red',   label='This WCS')
    draw_grid(wcs_gt,     wcs_shown, image.shape, ax, color='green', label='Ground truth WCS')
    ax.legend(loc='upper right', fontsize=8)
    ax.set_title(f"{title} — pair {PAIR_NUM}")
    ax.set_xlim(0, image.shape[1])
    ax.set_ylim(0, image.shape[0])
    ax.axis('off')

plt.tight_layout()
plt.show()
