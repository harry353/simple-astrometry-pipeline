import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.wcs import WCS
import constants_astrometry as constants
import constants_datagen
from utils import load_fits

base = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data_generation", "dataset", "pair_0001")


def main(haystack_path, needle_path, shift_x, shift_y):
    haystack, header_haystack, _, _ = load_fits(haystack_path, needle_path)

    # Convert the detected shift into a pixel position in the haystack.
    # shift_x, shift_y are offsets from the haystack centre, so we add them
    # to the centre coordinates to get the detected needle centre position.
    H, W = haystack.shape
    cx = int(np.round(W / 2 + shift_x))
    cy = int(np.round(H / 2 + shift_y))

    # Extract a square cutout of NEEDLE_SIZE around the detected centre.
    # This cutout is what we will compare against the needle in later stages.
    # Clamp to image boundaries and zero-pad if the centre is near an edge.
    half    = constants_datagen.NEEDLE_SIZE // 2
    y0, y1  = cy - half, cy + half
    x0, x1  = cx - half, cx + half

    print(f"Candidate needle centre: ({cx}, {cy})  cutout: x=[{x0},{x1}) y=[{y0},{y1})")

    # Clamped source indices
    src_x0 = max(x0, 0)
    src_y0 = max(y0, 0)
    src_x1 = min(x1, W)
    src_y1 = min(y1, H)

    # Destination indices inside the (NEEDLE_SIZE × NEEDLE_SIZE) canvas
    dst_x0 = src_x0 - x0
    dst_y0 = src_y0 - y0
    dst_x1 = dst_x0 + (src_x1 - src_x0)
    dst_y1 = dst_y0 + (src_y1 - src_y0)

    cutout = np.zeros((constants_datagen.NEEDLE_SIZE, constants_datagen.NEEDLE_SIZE),
                      dtype=haystack.dtype)
    cutout[dst_y0:dst_y1, dst_x0:dst_x1] = haystack[src_y0:src_y1, src_x0:src_x1]

    # Build a WCS header for the cutout by taking the haystack WCS and updating
    # the reference pixel (CRPIX) and reference coordinate (CRVAL) to match the
    # cutout's new centre.
    wcs_h         = WCS(header_haystack)
    cutout_header = header_haystack.copy()
    cutout_header['NAXIS1'] = constants_datagen.NEEDLE_SIZE
    cutout_header['NAXIS2'] = constants_datagen.NEEDLE_SIZE
    cutout_header['CRPIX1'] = constants_datagen.NEEDLE_SIZE / 2 + 0.5   # FITS 1-based geometric centre
    cutout_header['CRPIX2'] = constants_datagen.NEEDLE_SIZE / 2 + 0.5
    center_ra, center_dec   = (float(v) for v in wcs_h.pixel_to_world_values(cx, cy))
    cutout_header['CRVAL1'] = center_ra    # sky coordinates of the detected needle centre
    cutout_header['CRVAL2'] = center_dec

    pair_dir = os.path.dirname(haystack_path)
    pair_num = os.path.basename(pair_dir).split("_")[1]

    fits_out = os.path.join(pair_dir, f"candidate_needle_{pair_num}.fits")
    fits.writeto(fits_out, cutout, header=cutout_header, overwrite=True)
    print(f"Saved candidate FITS to {fits_out}")

    png_out = os.path.join(pair_dir, f"candidate_needle_{pair_num}.png")
    h, w = cutout.shape
    fig = plt.figure(figsize=(w / 100, h / 100), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(cutout, cmap='viridis', aspect='auto')
    ax.axis('off')
    fig.savefig(png_out, dpi=100)
    plt.close(fig)
    print(f"Saved candidate PNG to {png_out}")


if __name__ == "__main__":
    main(
        haystack_path=os.path.join(base, "haystack_0001.fits"),
        needle_path=os.path.join(base, "needle_0001.fits"),
        shift_x=0.0,
        shift_y=0.0,
    )
