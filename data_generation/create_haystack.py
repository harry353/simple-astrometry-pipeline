import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
import matplotlib.pyplot as plt
import constants


def create_galaxy_haystack(size=1500, num_blobs=None):
    if num_blobs is None:
        density = getattr(constants, 'GALAXY_DENSITY', 30)
        num_blobs = int(density * (size * size) / 1_000_000)
        print(f"Calculated {num_blobs} blobs for size {size} (Density: {density})")
    
    img = np.zeros((size, size), dtype=np.float32)
    np.random.seed(constants.RANDOM_SEED)
    
    print(f"Generating {size}x{size} haystack...")
    for _ in range(num_blobs):
        cy, cx = np.random.randint(100, size-100, 2)
        sy, sx = np.random.uniform(5, 25), np.random.uniform(2, 10)
        ang = np.random.uniform(0, 180)
        amp = np.random.uniform(0.5, 1.0)
        
        pad = int(max(sy, sx) * 3)
        y_min, y_max = max(0, cy-pad), min(size, cy+pad)
        x_min, x_max = max(0, cx-pad), min(size, cx+pad)
        
        # Use a centered grid for each blob to avoid 0.5-pixel centering bias
        gy, gx = np.mgrid[0:y_max-y_min, 0:x_max-x_min]
        y_center_in_grid = cy - y_min
        x_center_in_grid = cx - x_min
        y_grid = gy - y_center_in_grid
        x_grid = gx - x_center_in_grid
        
        rad = np.radians(ang)
        x_rot = x_grid * np.cos(rad) + y_grid * np.sin(rad)
        y_rot = -x_grid * np.sin(rad) + y_grid * np.cos(rad)
        
        blob = amp * np.exp(-(x_rot**2 / (2*sx**2) + y_rot**2 / (2*sy**2)))
        img[y_min:y_max, x_min:x_max] += blob

    return img


def create_wcs_header(size, fov_deg):
    """Creates a basic WCS header for the given size and field of view."""
    # Pixel scale in degrees/pixel
    # fov_deg = 5 / 60 = 1/12
    scale = fov_deg / size 
    
    w = WCS(naxis=2)
    
    # Set reference pixel to center
    w.wcs.crpix = [size / 2 + 0.5, size / 2 + 0.5] # 1-based indexing for FITS
    
    # Set reference coordinate (arbitrary, e.g., RA=0, DEC=0)
    w.wcs.crval = [0.0, 0.0]
    
    # intrinsic step size (CDELT)
    w.wcs.cdelt = [-scale, scale] # Standard convention: RA decreases as x increases
    
    # Coordinate system type
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    
    return w.to_header()


def process_haystack(haystack_img, noise_level=constants.HAYSTACK_NOISE):
    """Applies noise to the haystack image."""
    np.random.seed(constants.RANDOM_SEED + 1) # Use a different (but deterministic) seed for noise
    print(f"Processing Haystack: Noise={noise_level}")
    noisy = haystack_img + np.random.normal(0, noise_level, haystack_img.shape)
    return np.clip(noisy, 0, 1)


def save_haystacks(haystack_img, haystack_noisy, header, out_path, filename_prefix, save_clean):
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    clean_name = f"{filename_prefix}_clean.fits"
    noisy_name = f"{filename_prefix}.fits"

    if save_clean:
        fits.writeto(os.path.join(out_path, clean_name), haystack_img, header=header, overwrite=True)
        print(f"Saved {clean_name} in {out_path}")

        plt.imsave(os.path.join(out_path, f"{filename_prefix}_clean.png"), haystack_img, cmap='viridis')
        print(f"Saved {filename_prefix}_clean.png in {out_path}")

    fits.writeto(os.path.join(out_path, noisy_name), haystack_noisy, header=header, overwrite=True)
    print(f"Saved {noisy_name} in {out_path} with WCS info")

    plt.imsave(os.path.join(out_path, f"{filename_prefix}.png"), haystack_noisy, cmap='viridis')
    print(f"Saved {filename_prefix}.png in {out_path}")


def main(output_dir=None, filename_prefix="haystack", save_clean=True):
    size = getattr(constants, 'HAYSTACK_SIZE', 3000)
    out_path = output_dir if output_dir else constants.FITS_DIR

    fov_deg = 5.0 / 60.0
    haystack_img = create_galaxy_haystack(size)
    header = create_wcs_header(size, fov_deg)
    print(f"Created WCS Header for {fov_deg:.4f} deg FOV ({size}px)")
    
    haystack_noisy = process_haystack(haystack_img, noise_level=constants.HAYSTACK_NOISE)

    save_haystacks(haystack_img, haystack_noisy, header, out_path, filename_prefix, save_clean)

    return haystack_img, header


if __name__ == "__main__":
    main()
