import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from scipy.ndimage import rotate, zoom, gaussian_filter
from skimage.feature import peak_local_max
import matplotlib.pyplot as plt
import constants_datagen as constants


def process_needle(needle_img, modifications, output_size=300):
    """Applies rotation, stretch, and noise to the needle image based on modifications dict.
    Result is centre-cropped to output_size."""
    angle = modifications.get('angle', 0)
    stretch_y = modifications.get('stretch_y', 1.0)
    stretch_x = modifications.get('stretch_x', 1.0)
    noise_level = modifications.get('noise', 0.0)

    print(f"Processing Needle: Angle={angle:.2f} deg, Stretch Y={stretch_y:.3f}, X={stretch_x:.3f}, Noise={noise_level}, Output={output_size}")
    
    # 1. Rotate
    needle_transformed = rotate(needle_img, angle, reshape=False)
    
    # 2. Zoom
    needle_transformed = zoom(needle_transformed, (stretch_y, stretch_x))
    
    # 3. Center Crop
    h, w = needle_transformed.shape
    start_y = (h - output_size) // 2
    start_x = (w - output_size) // 2
    
    if start_y < 0 or start_x < 0:
        print(f"Warning: Transformed needle ({h}x{w}) smaller than output ({output_size}x{output_size})!")
    
    crop_img = needle_transformed[start_y:start_y+output_size, start_x:start_x+output_size]
    
    # 4. Noise
    np.random.seed(constants.RANDOM_SEED + 2) # Different deterministic seed
    noisy = crop_img + np.random.normal(0, noise_level, crop_img.shape)
    return np.clip(noisy, 0, 1)


def get_galaxy_centroids(patch, threshold=None, size=20, sigma=2):
    """Returns the (y, x) coordinates of distinct galaxies in a patch."""
    if threshold is None:
        noise_level = getattr(constants, 'NEEDLE_NOISE', 0)
        threshold = 0.1 + 2.0 * noise_level

    smoothed = gaussian_filter(patch, sigma=sigma)
    coords = peak_local_max(smoothed, min_distance=size, threshold_abs=threshold)
    return [tuple(c) for c in coords]


def load_haystack(haystack_dir, haystack_prefix):
    """Loads the clean haystack and its header from disk.
    Falls back to the noisy file if no _clean variant exists."""
    h_clean_path = os.path.join(haystack_dir, f"{haystack_prefix}_clean.fits")
    h_noisy_path = os.path.join(haystack_dir, f"{haystack_prefix}.fits")
    for path in (h_clean_path, h_noisy_path):
        try:
            with fits.open(path) as hdul:
                haystack_clean = hdul[0].data
                header_haystack = hdul[0].header
            print(f"Loaded {path} from disk")
            return haystack_clean, header_haystack
        except FileNotFoundError:
            continue
    print(f"Error: [Errno 2] No such file or directory: '{h_clean_path}'. Run 01_create_haystack_galsim.py or 01b_create_haystack_gaussian.py first.")
    return None, None


def find_needle_region(haystack_clean, needle_size, buffer_size):
    """Searches for a suitable needle region in the haystack. Returns (y_idx, x_idx, y_idx_buffer, x_idx_buffer)."""
    max_y = haystack_clean.shape[0] - buffer_size
    max_x = haystack_clean.shape[1] - buffer_size

    if max_y <= 0 or max_x <= 0:
        print("Error: Haystack is smaller than the requested buffer size!")
        exit(1)

    np.random.seed(constants.RANDOM_SEED + 10)

    galaxies_found = 0
    central_galaxies = 0
    attempts = 0
    max_attempts = 1000

    min_galaxies = getattr(constants, 'MIN_GALAXIES', 3)
    min_central = getattr(constants, 'MIN_CENTRAL_GALAXIES', 1)
    central_size = getattr(constants, 'CENTRAL_REGION_SIZE', 100)

    center_start = (needle_size - central_size) // 2
    center_end = center_start + central_size
    offset = (buffer_size - needle_size) // 2

    print(f"Searching for a needle region with >= {min_galaxies} galaxies and >= {min_central} in center {central_size}x{central_size}...")
    while (galaxies_found < min_galaxies or central_galaxies < min_central) and attempts < max_attempts:
        y_idx_buffer = np.random.randint(100, max_y - 100)
        x_idx_buffer = np.random.randint(100, max_x - 100)

        y_idx = y_idx_buffer + offset
        x_idx = x_idx_buffer + offset

        inner_needle = haystack_clean[y_idx:y_idx+needle_size, x_idx:x_idx+needle_size]
        centroids = get_galaxy_centroids(inner_needle, sigma=4)
        galaxies_found = len(centroids)

        central_galaxies = 0
        if galaxies_found >= min_galaxies:
            for cy, cx in centroids:
                if center_start <= cy < center_end and center_start <= cx < center_end:
                    central_galaxies += 1

        attempts += 1

    region_ok = galaxies_found >= min_galaxies and central_galaxies >= min_central
    if not region_ok:
        print(f"Warning: Could not find a matching region after {max_attempts} attempts.")
    else:
        print(f"Found region with {galaxies_found} galaxies ({central_galaxies} central) after {attempts} {'attempt' if attempts == 1 else 'attempts'}.")

    print(f"Needle Buffer: {buffer_size}x{buffer_size} at ({y_idx_buffer}, {x_idx_buffer})")
    print(f"Needle Target: {needle_size}x{needle_size} at ({y_idx}, {x_idx})")
    return y_idx, x_idx, y_idx_buffer, x_idx_buffer, region_ok


def build_needle_header(y_idx, x_idx, needle_size, header_haystack):
    """Builds the FITS header for the needle, including WCS derived from the haystack."""
    hdr = fits.Header()

    half = (needle_size - 1) / 2.0
    true_cx = x_idx + half
    true_cy = y_idx + half

    if header_haystack:
        wcs_haystack = WCS(header_haystack)
        center_ra, center_dec = wcs_haystack.pixel_to_world_values(true_cx, true_cy)

        wcs_error_x = np.random.uniform(-constants.WCS_ERROR_MAX, constants.WCS_ERROR_MAX)
        wcs_error_y = np.random.uniform(-constants.WCS_ERROR_MAX, constants.WCS_ERROR_MAX)

        w_needle = WCS(naxis=2)
        w_needle.wcs.crval = [center_ra, center_dec]
        w_needle.wcs.crpix = [
            half + 1.0 + wcs_error_x,
            half + 1.0 + wcs_error_y
        ]
        w_needle.wcs.ctype = ["RA---TAN", "DEC--TAN"]

        orig_cdelt = wcs_haystack.wcs.cdelt
        if orig_cdelt is None:
            scale_x = -1.0/6.0/3000.0
            scale_y = 1.0/6.0/3000.0
        else:
            scale_x, scale_y = orig_cdelt

        w_needle.wcs.cdelt = [scale_x, scale_y]
        hdr.update(w_needle.to_header())

        px_scale_arcsec = abs(scale_x) * 3600
        hdr['WERR_XPX'] = (wcs_error_x, 'WCS error X in pixels')
        hdr['WERR_YPX'] = (wcs_error_y, 'WCS error Y in pixels')
        hdr['WERR_XAS'] = (wcs_error_x * px_scale_arcsec, 'WCS error X in arcseconds')
        hdr['WERR_YAS'] = (wcs_error_y * px_scale_arcsec, 'WCS error Y in arcseconds')

    return hdr


def save_needles(needle_noisy, hdr, out_path, filename_prefix):
    """Saves the processed needle to disk."""
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    output_path = os.path.join(out_path, f"{filename_prefix}.fits")
    fits.writeto(output_path, needle_noisy, header=hdr, overwrite=True)
    print(f"Saved {output_path} with location info and WCS")

    plt.imsave(os.path.join(out_path, f"{filename_prefix}.png"), needle_noisy, cmap='viridis')
    print(f"Saved {filename_prefix}.png in {out_path}")


def main(output_dir=None, filename_prefix="needle", haystack_dir=None, haystack_prefix="haystack",
         haystack_clean=None, header_haystack=None):
    needle_size = constants.NEEDLE_SIZE
    buffer_size = constants.BUFFER_SIZE
    out_path = output_dir if output_dir else constants.FITS_DIR

    if haystack_clean is None:
        h_dir = haystack_dir if haystack_dir else constants.FITS_DIR
        haystack_clean, header_haystack = load_haystack(h_dir, haystack_prefix)
        if haystack_clean is None:
            return
    else:
        print("Using in-memory haystack data")

    y_idx, x_idx, y_idx_buffer, x_idx_buffer, region_ok = find_needle_region(haystack_clean, needle_size, buffer_size)

    needle_img_buffer = haystack_clean[y_idx_buffer:y_idx_buffer+buffer_size,
                                       x_idx_buffer:x_idx_buffer+buffer_size].copy()

    angle = np.random.uniform(-constants.NEEDLE_ANGLE_MAX, constants.NEEDLE_ANGLE_MAX)
    needle_modifications = {
        'angle': angle,
        'stretch_y': constants.NEEDLE_STRETCH_Y,
        'stretch_x': constants.NEEDLE_STRETCH_X,
        'noise': constants.NEEDLE_NOISE
    }
    needle_noisy = process_needle(needle_img_buffer, needle_modifications, output_size=needle_size)

    hdr = build_needle_header(y_idx, x_idx, needle_size, header_haystack)
    hdr['NANGLE'] = (angle, 'Needle rotation angle in degrees')

    save_needles(needle_noisy, hdr, out_path, filename_prefix)
    return region_ok


if __name__ == "__main__":
    _script_dir = os.path.dirname(os.path.abspath(__file__))
    _pair_dir   = os.path.join(_script_dir, "dataset", "pair_0001")
    main(output_dir=_pair_dir, filename_prefix="needle_0001",
         haystack_dir=_pair_dir, haystack_prefix="haystack_0001")
