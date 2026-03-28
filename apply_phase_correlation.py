import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from skimage.registration import phase_cross_correlation
import constants
from utils import load_fits, prepare_images, build_correlation_map, save_correlation, print_localisation_accuracy


def main():
    haystack_path = os.path.join(constants.FITS_DIR, "haystack_0001.fits")
    needle_path = os.path.join(constants.FITS_DIR, "needle_0001.fits")

    haystack, header_haystack, needle, header_needle = load_fits(haystack_path, needle_path)

    w_haystack, w_canvas = prepare_images(haystack, needle)

    print("Performing phase correlation...")
    upsample = getattr(constants, 'UPSAMPLE_FACTOR', 100)
    shift, *_ = phase_cross_correlation(w_haystack, w_canvas, upsample_factor=upsample)

    corr_map = build_correlation_map(w_haystack, w_canvas)

    print_localisation_accuracy(shift[1], shift[0], header_needle, header_haystack)

    save_correlation(corr_map, header_haystack, shift[0], shift[1], haystack_path)


if __name__ == "__main__":
    main()
