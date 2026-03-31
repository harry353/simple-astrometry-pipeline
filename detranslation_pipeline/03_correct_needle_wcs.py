import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.wcs import WCS
from utils import load_fits

base = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data_generation", "dataset", "pair_0001")


def main(haystack_path, needle_path, shift_x, shift_y):
    haystack, header_haystack, needle, header_needle = load_fits(haystack_path, needle_path)

    # To correct the needle WCS, we:
    #   1. Find where the needle centre actually is in the haystack (via the shift)
    #   2. Convert that position to sky coordinates using the haystack WCS
    #   3. Reset CRPIX to the geometric centre of the needle
    #   4. Set CRVAL to the sky coordinates we just computed
    H, W = haystack.shape
    det_cx = W / 2 + shift_x   # detected needle centre in haystack pixels (x)
    det_cy = H / 2 + shift_y   # detected needle centre in haystack pixels (y)
    wcs_haystack = WCS(header_haystack)
    corrected_ra, corrected_dec = (float(v) for v in wcs_haystack.pixel_to_world_values(det_cx, det_cy))

    print(f"Corrected CRVAL: RA={corrected_ra:.6f} deg  Dec={corrected_dec:.6f} deg")

    nh, nw = needle.shape
    corrected_header = header_needle.copy()
    corrected_header['CRPIX1'] = nw / 2 + 0.5   # FITS 1-based geometric centre
    corrected_header['CRPIX2'] = nh / 2 + 0.5
    corrected_header['CRVAL1'] = corrected_ra    # sky position of the needle centre
    corrected_header['CRVAL2'] = corrected_dec

    # Save the corrected needle. The pixel data is identical to the original,
    # only the WCS header has changed.
    pair_dir = os.path.dirname(needle_path)
    pair_num = os.path.basename(pair_dir).split("_")[1]
    fits_out = os.path.join(pair_dir, f"detranslated_needle_{pair_num}.fits")
    fits.writeto(fits_out, needle, header=corrected_header, overwrite=True)
    print(f"Saved corrected FITS to {fits_out}")

    png_out = os.path.join(pair_dir, f"detranslated_needle_{pair_num}.png")
    plt.imsave(png_out, needle, cmap='viridis')
    print(f"Saved PNG to {png_out}")


if __name__ == "__main__":
    main(
        haystack_path=os.path.join(base, "haystack_0001.fits"),
        needle_path=os.path.join(base, "needle_0001.fits"),
        shift_x=0.0,
        shift_y=0.0,
    )
