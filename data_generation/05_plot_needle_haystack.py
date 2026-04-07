import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from astropy.io import fits
from astropy.wcs import WCS
import constants_datagen as constants
from scipy.ndimage import maximum_filter, gaussian_filter

import random

def plot_all(haystack_path=None, needle_path=None):
    # If no paths provided, try to find a random pair from the dataset directory
    if haystack_path is None and needle_path is None:
        dataset_dir = "dataset"
        if os.path.exists(dataset_dir):
            pairs = [d for d in os.listdir(dataset_dir) if d.startswith("pair_") and os.path.isdir(os.path.join(dataset_dir, d))]
            if pairs:
                selected_pair = random.choice(pairs)
                pair_name = selected_pair.replace("pair_", "")
                pair_path = os.path.join(dataset_dir, selected_pair)
                
                haystack_path = os.path.join(pair_path, f"haystack_{pair_name}.fits")
                needle_path = os.path.join(pair_path, f"needle_{pair_name}.fits")
                print(f"Randomly selected {selected_pair} for plotting.")
    
    # Fallback to default fits_files for local runs
    if haystack_path is None:
        haystack_path = os.path.join(constants.FITS_DIR, "haystack.fits")
    if needle_path is None:
        needle_path = os.path.join(constants.FITS_DIR, "needle.fits")

    # 1. Load Haystack and its WCS
    try:
        with fits.open(haystack_path) as hdul:
            haystack_img = hdul[0].data
            haystack_header = hdul[0].header
        print(f"Loaded {haystack_path}")
    except FileNotFoundError:
        print(f"Error: {haystack_path} not found.")
        return

    # 2. Load Needle and its location info
    try:
        with fits.open(needle_path) as hdul:
            needle_img = hdul[0].data
            needle_header = hdul[0].header
        print(f"Loaded {needle_path}")
    except FileNotFoundError:
        print(f"Error: {needle_path} not found.")
        return

    # Extract coordinates from needle header
    if 'X_IDX' in needle_header and 'Y_IDX' in needle_header:
        x_idx = needle_header['X_IDX']
        y_idx = needle_header['Y_IDX']
    else:
        print("Warning: X_IDX/Y_IDX not in header, falling back to center of haystack.")
        x_idx = haystack_img.shape[1] // 2 - constants.NEEDLE_SIZE // 2
        y_idx = haystack_img.shape[0] // 2 - constants.NEEDLE_SIZE // 2
        
    coords = (y_idx, x_idx)
    needle_size = constants.NEEDLE_SIZE
    
    # Extract extraction params from constants
    angle = constants.NEEDLE_ANGLE_MAX
    print(f"Needle location (from header): {coords} (Size={needle_size}, Angle={angle:.2f} deg)")

    # 3. Plot
    wcs_haystack = WCS(haystack_header)
    wcs_needle = WCS(needle_header)
    
    fig = plt.figure(figsize=(12, 6))
    
    # Left: Haystack with box (using WCS)
    ax1 = fig.add_subplot(1, 2, 1, projection=wcs_haystack)
    ax1.imshow(haystack_img, cmap='viridis', origin='lower')
    
    rect = patches.Rectangle((x_idx, y_idx), needle_size, needle_size, 
                             linewidth=2, edgecolor='red', facecolor='none', 
                             linestyle='--')
    ax1.add_patch(rect)
    ax1.set_title(f"Haystack (WCS)\nNeedle marked")
    
    # Add coordinate grid
    ax1.coords.grid(color='white', alpha=0.5, linestyle='solid')
    ax1.set_xlabel('RA')
    ax1.set_ylabel('Dec')

    # Right: Needle
    ax2 = fig.add_subplot(1, 2, 2, projection=wcs_needle)
    ax2.imshow(needle_img, cmap='viridis', origin='lower')
    ax2.set_title(f"Needle (Extracted, WCS)\nAngle: {angle:.2f} deg")
    
    # Add coordinate grid for needle
    ax2.coords.grid(color='white', alpha=0.5, linestyle='solid')
    ax2.set_xlabel('RA')
    ax2.set_ylabel('Dec')
    
    # Plot central region on needle for visual verification
    central_size = getattr(constants, 'CENTRAL_REGION_SIZE', 100)
    center_start = (needle_size - central_size) // 2
    center_end = center_start + central_size
    
    # Use noise level to set a robust threshold
    noise_level = getattr(constants, 'NEEDLE_NOISE', 0)
    threshold = 0.1 + 2.0 * noise_level

    # We will compute the central galaxies from the full needle image 
    # to avoid falsely detecting boundary slopes as local maxima
    # 1. Smooth to reduce noise spikes
    smoothed = gaussian_filter(needle_img, sigma=2)
    
    # 2. Find local peaks on smoothed image
    local_max = maximum_filter(smoothed, size=20) == smoothed
    is_galaxy_peak = local_max & (smoothed > threshold)
    
    y_peaks, x_peaks = np.where(is_galaxy_peak)
    
    central_peaks = []
    for py, px in zip(y_peaks, x_peaks):
        if center_start <= py < center_end and center_start <= px < center_end:
            central_peaks.append((py, px))
            
    print(f"Found {len(central_peaks)} galaxies within the central {central_size}x{central_size} region in the final needle image.")
    for py, px in central_peaks:
        circ = patches.Circle((px, py), radius=5, color='cyan', fill=False)
        ax2.add_patch(circ)
        print(f"  - Galaxy at needle ({px}, {py})")

    rect2 = patches.Rectangle((center_start, center_start), central_size, central_size,
                              linewidth=2, edgecolor='red', facecolor='none',
                              linestyle='--')
    ax2.add_patch(rect2)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    plot_all()
