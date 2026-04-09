import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from astropy.io import fits
from astropy.wcs import WCS
import constants_datagen as constants

N_SAMPLES = 5

DATA_GENERATION_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR         = os.path.join(DATA_GENERATION_DIR, "dataset")


def iter_pairs(root_dir):
    for name in sorted(os.listdir(root_dir)):
        path = os.path.join(root_dir, name)
        if os.path.isdir(path) and name.startswith("pair_"):
            yield name, path


def load_wcs_errors(root_dir):
    ex, ey, angles = [], [], []
    for _, pair_path in iter_pairs(root_dir):
        for f in os.listdir(pair_path):
            if f.startswith("needle_") and f.endswith(".fits"):
                hdr = fits.getheader(os.path.join(pair_path, f))
                if "WERR_XPX" in hdr:
                    ex.append(hdr["WERR_XPX"])
                    ey.append(hdr["WERR_YPX"])
                if "NANGLE" in hdr:
                    angles.append(hdr["NANGLE"])
    return np.array(ex), np.array(ey), np.array(angles)


def load_samples(root_dir, n):
    samples = []
    for _, pair_path in iter_pairs(root_dir):
        if len(samples) >= n:
            break
        haystack = needle = None
        haystack_hdr = needle_hdr = None
        for f in os.listdir(pair_path):
            fp = os.path.join(pair_path, f)
            if f.startswith("haystack_") and f.endswith(".fits"):
                with fits.open(fp) as hdul:
                    haystack = hdul[0].data.astype(float)
                    haystack_hdr = hdul[0].header
            elif f.startswith("needle_") and f.endswith(".fits"):
                with fits.open(fp) as hdul:
                    needle = hdul[0].data.astype(float)
                    needle_hdr = hdul[0].header

        if haystack is None or needle is None:
            continue

        # Recover needle centre in haystack pixel coords via CRVAL + haystack WCS
        needle_cx = needle_cy = None
        if haystack_hdr is not None and needle_hdr is not None:
            try:
                wcs_h = WCS(haystack_hdr)
                ra, dec = needle_hdr['CRVAL1'], needle_hdr['CRVAL2']
                needle_cx, needle_cy = wcs_h.world_to_pixel_values(ra, dec)
            except Exception:
                pass

        samples.append((haystack, needle, needle_cx, needle_cy))
    return samples


def main(root_dir=DATASET_DIR, output_path=os.path.join(DATA_GENERATION_DIR, "diagnostics.png")):
    print("Loading samples and WCS errors...")
    samples = load_samples(root_dir, N_SAMPLES)
    ex, ey, angles = load_wcs_errors(root_dir)

    n = len(samples)
    fig = plt.figure(figsize=(4 * max(n, 5), 10))
    fig.suptitle(f"Data Generation Diagnostics  ({len(ex)} pairs)", fontsize=14, y=1.01)

    cols = max(n, 5)

    # Row 1: haystacks with needle location marked
    half = constants.NEEDLE_SIZE / 2
    for i, (haystack, _, needle_cx, needle_cy) in enumerate(samples):
        ax = fig.add_subplot(3, cols, i + 1)
        ax.imshow(haystack, cmap="viridis", origin="lower")
        if needle_cx is not None:
            rect = patches.Rectangle(
                (needle_cx - half, needle_cy - half),
                constants.NEEDLE_SIZE, constants.NEEDLE_SIZE,
                linewidth=1, edgecolor="red", facecolor="none", linestyle="--"
            )
            ax.add_patch(rect)
        ax.set_title(f"Haystack {i+1}", fontsize=9)
        ax.axis("off")

    # Row 2: noisy needles
    for i, (_, needle, _, _) in enumerate(samples):
        ax = fig.add_subplot(3, cols, cols + i + 1)
        ax.imshow(needle, cmap="viridis", origin="lower")
        ax.set_title(f"Needle {i+1}", fontsize=9)
        ax.axis("off")

    # Row 3: WCS error statistics + rotation histogram
    offset = 2 * cols + 1

    ax_sc = fig.add_subplot(3, cols, offset)
    ax_hx = fig.add_subplot(3, cols, offset + 1)
    ax_hy = fig.add_subplot(3, cols, offset + 2)
    ax_mg = fig.add_subplot(3, cols, offset + 3)
    ax_rot = fig.add_subplot(3, cols, offset + 4)

    if len(ex) == 0:
        for ax, title in [(ax_sc, "WCS error scatter"),
                          (ax_hx, "WCS error X"),
                          (ax_hy, "WCS error Y"),
                          (ax_mg, "WCS error magnitude")]:
            ax.text(0.5, 0.5, "No WCS error data\n(regenerate dataset)",
                    ha="center", va="center", transform=ax.transAxes)
            ax.set_title(title)
    else:
        mag = np.sqrt(ex**2 + ey**2)

        ax_sc.scatter(ex, ey, s=4, alpha=0.5)
        ax_sc.axhline(0, color="k", linewidth=0.5)
        ax_sc.axvline(0, color="k", linewidth=0.5)
        ax_sc.set_xlabel("Error X (px)")
        ax_sc.set_ylabel("Error Y (px)")
        ax_sc.set_title("WCS error scatter")

        ax_hx.hist(ex, bins=30, color="steelblue", edgecolor="none")
        ax_hx.axvline(0, color="k", linewidth=0.5)
        ax_hx.set_xlabel("Error X (px)")
        ax_hx.set_title(f"WCS error X  (μ={ex.mean():.1f}, σ={ex.std():.1f})")

        ax_hy.hist(ey, bins=30, color="coral", edgecolor="none")
        ax_hy.axvline(0, color="k", linewidth=0.5)
        ax_hy.set_xlabel("Error Y (px)")
        ax_hy.set_title(f"WCS error Y  (μ={ey.mean():.1f}, σ={ey.std():.1f})")

        ax_mg.hist(mag, bins=30, color="mediumseagreen", edgecolor="none")
        ax_mg.set_xlabel("Error magnitude (px)")
        ax_mg.set_title(f"WCS error magnitude  (μ={mag.mean():.1f})")

    if len(angles) == 0:
        ax_rot.text(0.5, 0.5, "No rotation data\n(regenerate dataset)",
                    ha="center", va="center", transform=ax_rot.transAxes)
        ax_rot.set_title("Rotation angle")
    else:
        ax_rot.hist(angles, bins=30, color="mediumpurple", edgecolor="none")
        ax_rot.axvline(0, color="k", linewidth=0.5)
        ax_rot.set_xlabel("Rotation angle (deg)")
        ax_rot.set_title(f"Rotation  (μ={angles.mean():.2f}°, σ={angles.std():.2f}°)")

    plt.tight_layout()
    plt.savefig(output_path, dpi=120, bbox_inches="tight")
    print(f"Saved diagnostics to {output_path}")
    plt.close()


if __name__ == "__main__":
    main()
