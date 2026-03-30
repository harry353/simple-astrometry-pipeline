import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from scipy.signal import wiener
import constants
from utils import load_fits, print_localisation_accuracy, save_correlation_map, match_template_subpixel

base = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data_generation", "dataset", "pair_0001")

def main(haystack_path, needle_path):
    haystack, header_haystack, needle, header_needle = load_fits(haystack_path, needle_path)

    with np.errstate(divide='ignore', invalid='ignore'):
        w_haystack = np.nan_to_num(wiener(haystack.astype(np.float64), mysize=7),  nan=0.0)
        w_needle   = np.nan_to_num(wiener(needle.astype(np.float64),   mysize=11), nan=0.0)

    print("Performing template matching...")
    shift_x, shift_y, result = match_template_subpixel(w_haystack, w_needle)

    print_localisation_accuracy(shift_x, shift_y, header_needle, header_haystack)

    if constants.SAVE_CORRELATION_MAP:
        save_correlation_map(result, haystack_path)


if __name__ == "__main__":
    main(
        haystack_path=os.path.join(base, "haystack_0001.fits"),
        needle_path=os.path.join(base, "needle_0001.fits"),
    )
