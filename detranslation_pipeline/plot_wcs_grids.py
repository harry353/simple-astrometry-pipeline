import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.wcs import WCS

pair_num = "0001"
base = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data_generation", "dataset", f"pair_{pair_num}")

with fits.open(os.path.join(base, f"needle_{pair_num}_ground_truth.fits")) as f:
    gt_image  = f[0].data
    wcs_gt    = WCS(f[0].header)

with fits.open(os.path.join(base, f"needle_{pair_num}.fits")) as f:
    wcs_initial = WCS(f[0].header)

with fits.open(os.path.join(base, f"candidate_needle_{pair_num}.fits")) as f:
    wcs_candidate = WCS(f[0].header)


def draw_grid(wcs_source, wcs_display, image_shape, ax, color, label, n_lines=6):
    """Draw the pixel grid of wcs_source projected into wcs_display pixel space.

    Generates lines of constant x and constant y in wcs_source pixel space,
    converts them to sky coordinates, then projects onto wcs_display pixels.
    This avoids RA wrap-around issues from working in sky space directly.
    """
    H, W = image_shape
    # Extend slightly beyond image bounds so lines reach the edges
    t = np.linspace(-50, max(H, W) + 50, 500)

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


fig, ax = plt.subplots(figsize=(8, 8))
ax.imshow(gt_image, cmap='gray', origin='lower')

draw_grid(wcs_initial,   wcs_gt, gt_image.shape, ax, color='red',    label='Initial WCS (erroneous)')
draw_grid(wcs_candidate, wcs_gt, gt_image.shape, ax, color='orange', label='Candidate WCS (PCC detection)')
draw_grid(wcs_gt,        wcs_gt, gt_image.shape, ax, color='green',  label='Ground truth WCS')

ax.legend(loc='upper right')
ax.set_title(f"WCS grid comparison — pair {pair_num}")
ax.set_xlim(0, gt_image.shape[1])
ax.set_ylim(0, gt_image.shape[0])
ax.axis('off')
plt.tight_layout()
plt.show()
