import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib.pyplot as plt
from skimage.registration import optical_flow_tvl1
from skimage.transform import warp
from utils import load_fits, prepare_images

pair_num = "0002"

base = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data_generation", "dataset", f"pair_{pair_num}")
haystack, header_haystack, needle, header_needle = load_fits(
    os.path.join(base, f"haystack_{pair_num}.fits"),
    os.path.join(base, f"needle_{pair_num}.fits"),
)

w_haystack, w_canvas = prepare_images(haystack, needle)

v, u = optical_flow_tvl1(w_haystack, w_canvas)

nr, nc = w_haystack.shape
row_coords, col_coords = np.meshgrid(np.arange(nr), np.arange(nc), indexing='ij')
canvas_warp = warp(w_canvas, np.array([row_coords + v, col_coords + u]), mode='edge')

norm = np.sqrt(u**2 + v**2)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].imshow(w_haystack, cmap='gray')
axes[0].set_title("Haystack")
axes[0].set_axis_off()

axes[1].imshow(w_canvas, cmap='gray')
axes[1].set_title(f"Needle canvas (pair {pair_num})")
axes[1].set_axis_off()

nvec = 30
step = max(nr // nvec, nc // nvec)
y, x = np.mgrid[:nr:step, :nc:step]
axes[2].imshow(norm, cmap='magma')
axes[2].quiver(x, y, u[::step, ::step], v[::step, ::step],
               color='r', units='dots', angles='xy', scale_units='xy', lw=2)
axes[2].set_title("Optical flow magnitude and vector field")
axes[2].set_axis_off()

fig.tight_layout()
plt.show()
