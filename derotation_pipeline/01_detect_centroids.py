import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from photutils.segmentation import detect_sources, SourceCatalog

PAIR_NUM = "0004"
SIGMA    = 5    # detection threshold in units of background sigma
NPIXELS  = 40   # minimum connected pixels to be counted as a source

base = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data_generation", "dataset", f"pair_{PAIR_NUM}")


def detect_centroids(image):
    # Estimate the background noise level
    _, median, std = sigma_clipped_stats(image, sigma=3.0)

    # Anything above median + SIGMA*std is considered a source.
    # Higher SIGMA = fewer but more confident detections.
    threshold = median + SIGMA * std

    # Label connected regions of pixels above the threshold. Each labelled
    # region is a candidate source. NPIXELS sets the minimum region size,
    # filtering out noise spikes that would otherwise appear as sources.
    segmap = detect_sources(image, threshold, npixels=NPIXELS)

    # Compute flux-weighted centroids for each segmented region.
    # Subtracting the median first removes the background contribution from the flux.
    catalog = SourceCatalog(image - median, segmap)
    return catalog, median


def main(pair_dir, plot=False):
    pair_num = os.path.basename(pair_dir).split("_")[1]
    candidate_path    = os.path.join(pair_dir, f"candidate_needle_{pair_num}.fits")
    detranslated_path = os.path.join(pair_dir, f"detranslated_needle_{pair_num}.fits")

    # Load both images
    with fits.open(candidate_path) as f:
        img_candidate = f[0].data.astype(np.float64)
    with fits.open(detranslated_path) as f:
        img_detranslated = f[0].data.astype(np.float64)

    # Run centroid detection on both images independently
    cat_candidate,    _ = detect_centroids(img_candidate)
    cat_detranslated, _ = detect_centroids(img_detranslated)

    # Save centroids to CSV, one file per image. The row index in each CSV
    # does NOT yet correspond between the two files; that matching happens
    # in match_centroids.py.
    for label, cat, filename in [
        ("Candidate",    cat_candidate,    f"centroids_candidate_{pair_num}.csv"),
        ("Detranslated", cat_detranslated, f"centroids_detranslated_{pair_num}.csv"),
    ]:
        n = len(cat) if cat is not None else 0
        print(f"{label}: {n} sources detected")
        csv_path = os.path.join(pair_dir, filename)
        if cat is not None:
            # Extract only the columns needed later: pixel position and integrated
            # flux. Flux is kept so match_centroids.py can optionally weight or filter
            # by brightness when pairing sources across the two images.
            df = pd.DataFrame({'x': np.array(cat.xcentroid), 'y': np.array(cat.ycentroid),
                               'flux': np.array(cat.segment_flux)})
            df.to_csv(csv_path, index=False)
            print(f"Saved {csv_path}")
        else:
            # Write an empty CSV with the correct column headers so downstream
            # scripts can load the file without special-casing the no-sources case.
            pd.DataFrame(columns=['x', 'y', 'flux']).to_csv(csv_path, index=False)
            print(f"Saved empty {csv_path}")

    if plot:
        fig, axes = plt.subplots(1, 2, figsize=(14, 7))

        for ax, image, catalog, title in [
            (axes[0], img_candidate,    cat_candidate,    "Candidate needle"),
            (axes[1], img_detranslated, cat_detranslated, "Detranslated needle"),
        ]:
            ax.imshow(image, cmap='viridis', origin='lower',
                      vmin=np.percentile(image, 1), vmax=np.percentile(image, 99))
            # Overlay the detected centroids as open
            if catalog is not None:
                ax.scatter(catalog.xcentroid, catalog.ycentroid,
                           s=60, facecolors='none', edgecolors='red', linewidths=1.2)
            n = len(catalog) if catalog is not None else 0
            ax.set_title(f"{title} — pair {pair_num}  ({n} sources)")
            ax.axis('off')

        plt.tight_layout()
        plt.show()

    return cat_candidate, cat_detranslated


if __name__ == "__main__":
    main(pair_dir=base, plot=True)
