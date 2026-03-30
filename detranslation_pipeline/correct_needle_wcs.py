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

    # Convert detected haystack pixel position to sky coordinates
    H, W = haystack.shape
    det_cx = W / 2 + shift_x
    det_cy = H / 2 + shift_y
    wcs_haystack = WCS(header_haystack)
    corrected_ra, corrected_dec = (float(v) for v in wcs_haystack.pixel_to_world_values(det_cx, det_cy))

    print(f"Corrected CRVAL: RA={corrected_ra:.6f} deg  Dec={corrected_dec:.6f} deg")

    # Build corrected header — update CRVAL only, CRPIX unchanged
    corrected_header = header_needle.copy()
    corrected_header['CRVAL1'] = corrected_ra
    corrected_header['CRVAL2'] = corrected_dec

    # Save corrected FITS (original untouched)
    pair_dir        = os.path.dirname(needle_path)
    fits_out        = os.path.join(pair_dir, "corrected_needle.fits")
    fits.writeto(fits_out, needle, header=corrected_header, overwrite=True)
    print(f"Saved corrected FITS to {fits_out}")

    # Save PNG copy
    png_out = os.path.join(pair_dir, "corrected_needle.png")
    plt.imsave(png_out, needle, cmap='viridis')
    print(f"Saved PNG to {png_out}")


if __name__ == "__main__":
    main(
        haystack_path=os.path.join(base, "haystack_0001.fits"),
        needle_path=os.path.join(base, "needle_0001.fits"),
        shift_x=0.0,
        shift_y=0.0,
    )
