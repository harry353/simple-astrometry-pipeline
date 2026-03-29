import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib.pyplot as plt
from utils import load_fits, prepare_images

pair_num = "0002"

base = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data_generation", "dataset", f"pair_{pair_num}")
haystack, header_haystack, needle, header_needle = load_fits(
    os.path.join(base, f"haystack_{pair_num}.fits"),
    os.path.join(base, f"needle_{pair_num}.fits"),
)

w_haystack, w_canvas = prepare_images(haystack, needle)

F1 = np.fft.fft2(w_haystack)
F2 = np.fft.fft2(w_canvas)
cross = F1 * np.conj(F2)
denom = np.abs(cross)
denom[denom == 0] = 1e-10
corr_map = np.fft.fftshift(np.fft.ifft2(cross / denom).real)

plt.imshow(corr_map, cmap='magma', vmin=corr_map.min(), vmax=corr_map.max())
plt.colorbar()
plt.title("Phase correlation map")
plt.tight_layout()
plt.show()  
