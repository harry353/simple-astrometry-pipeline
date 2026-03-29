import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
import constants
from utils import load_fits, prepare_images
from skimage.registration import phase_cross_correlation


def extract_candidate_needle(haystack, header_haystack, center_x, center_y):
    """Extract a 1x1 arcmin cutout from haystack centred at (center_x, center_y).

    Returns (cutout, cutout_header).
    """
    px_scale_deg = abs(header_haystack.get('CDELT2', 0))
    size_px = int(round((1 / 60) / px_scale_deg))  # 1 arcmin in pixels

    H, W = haystack.shape
    half = size_px // 2

    cx = int(round(center_x))
    cy = int(round(center_y))

    x0 = max(0, cx - half)
    x1 = min(W, cx + half)
    y0 = max(0, cy - half)
    y1 = min(H, cy + half)

    cutout = haystack[y0:y1, x0:x1].copy()

    # Update WCS so CRPIX reflects the cutout origin
    wcs = WCS(header_haystack)
    cutout_header = header_haystack.copy()
    cutout_header['NAXIS1'] = cutout.shape[1]
    cutout_header['NAXIS2'] = cutout.shape[0]
    cutout_header['CRPIX1'] = header_haystack['CRPIX1'] - x0
    cutout_header['CRPIX2'] = header_haystack['CRPIX2'] - y0

    return cutout, cutout_header


def main(haystack_path, needle_path, output_path=None):
    haystack, header_haystack, needle, header_needle = load_fits(haystack_path, needle_path)

    w_haystack, w_canvas = prepare_images(haystack, needle)

    print("Performing phase correlation...")
    upsample = getattr(constants, 'UPSAMPLE_FACTOR', 100)
    shift, *_ = phase_cross_correlation(w_haystack, w_canvas, upsample_factor=upsample)

    H, W = haystack.shape
    center_x = W // 2 + shift[1]
    center_y = H // 2 + shift[0]
    print(f"Detected centre (cx, cy): ({center_x:.2f}, {center_y:.2f})")

    cutout, cutout_header = extract_candidate_needle(haystack, header_haystack, center_x, center_y)
    print(f"Extracted candidate needle: {cutout.shape} px")

    if output_path is None:
        output_path = os.path.join(
            os.path.dirname(haystack_path),
            os.path.basename(haystack_path).replace("haystack", "candidate_needle")
        )

    hdu = fits.PrimaryHDU(cutout, header=cutout_header)
    hdu.writeto(output_path, overwrite=True)
    print(f"Saved candidate needle to {output_path}")

    import matplotlib.pyplot as plt
    png_path = os.path.splitext(output_path)[0] + ".png"
    plt.imsave(png_path, cutout, cmap='viridis')
    print(f"Saved candidate needle image to {png_path}")


if __name__ == "__main__":
    base = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data_generation", "dataset", "pair_0001")
    main(
        haystack_path=os.path.join(base, "haystack_0001.fits"),
        needle_path=os.path.join(base, "needle_0001.fits"),
    )
