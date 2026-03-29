import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from skimage.registration import phase_cross_correlation
import constants
from utils import load_fits, prepare_images, print_localisation_accuracy

base = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data_generation", "dataset", "pair_0001")

def main(haystack_path, needle_path):
    haystack, header_haystack, needle, header_needle = load_fits(haystack_path, needle_path)

    w_haystack, w_canvas = prepare_images(haystack, needle)

    print("Performing phase correlation...")
    upsample = getattr(constants, 'UPSAMPLE_FACTOR', 100)
    shift, *_ = phase_cross_correlation(w_haystack, w_canvas, upsample_factor=upsample)

    print_localisation_accuracy(shift[1], shift[0], header_needle, header_haystack)


if __name__ == "__main__":
    main(
        haystack_path=os.path.join(base, "haystack_0001.fits"),
        needle_path=os.path.join(base, "needle_0001.fits"),
    )
