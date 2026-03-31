import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.wcs import WCS

PAIR_NUM = "0001"

base = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data_generation", "dataset", f"pair_{PAIR_NUM}")


def main(pair_dir, voted_angle, plot=False):
    pair_num = os.path.basename(pair_dir).split("_")[1]
    detranslated_path = os.path.join(pair_dir, f"detranslated_needle_{pair_num}.fits")
    output_path       = os.path.join(pair_dir, f"corrected_needle_{pair_num}.fits")

    with fits.open(detranslated_path) as f:
        image  = f[0].data.astype(np.float64)
        header = f[0].header.copy()

    # Build a 2x2 rotation matrix for -voted_angle. We rotate in the opposite
    # direction to bring the WCS back into alignment.
    theta = np.radians(-voted_angle)
    R = np.array([[np.cos(theta), -np.sin(theta)],   # rotation matrix R(-theta)
                  [np.sin(theta),  np.cos(theta)]])

    # Read the existing CD matrix from the header. The CD matrix encodes both the
    # pixel scale and the orientation of the image on the sky. Each column maps one
    # pixel axis to sky coordinates (RA, Dec).
    # If no CD matrix is present fall back to building one from CDELT + PC matrix.
    wcs = WCS(header)
    if wcs.wcs.has_cd():
        CD = np.array(wcs.wcs.cd)                      # already a full CD matrix
    else:
        CD = np.diag(wcs.wcs.get_cdelt()) @ wcs.wcs.get_pc()  # CD = CDELT * PC

    # Apply the rotation: CD_new = R * CD_old.
    # This rotates the sky axes without touching the pixel data, no resampling,
    # no interpolation, no blurring.
    CD_new = R @ CD

    # Write the updated CD matrix back into the header. Using CD1_1 ... CD2_2
    # directly avoids ambiguity with the older CDELT/CROTA convention.
    header['CD1_1'] = CD_new[0, 0]   # dRA  / dx
    header['CD1_2'] = CD_new[0, 1]   # dRA  / dy
    header['CD2_1'] = CD_new[1, 0]   # dDec / dx
    header['CD2_2'] = CD_new[1, 1]   # dDec / dy

    # Remove CDELT/CROTA/PC keywords so there is no conflict with the CD matrix
    for kw in ('CDELT1', 'CDELT2', 'CROTA1', 'CROTA2', 'PC1_1', 'PC1_2', 'PC2_1', 'PC2_2'):
        header.remove(kw, ignore_missing=True)

    # Record the applied correction in the header for traceability
    header['DROT_ANG'] = (voted_angle, 'derotation angle applied (deg)')

    # Pixel data is written unchanged -- only the WCS has been updated
    fits.writeto(output_path, image.astype(np.float32), header=header, overwrite=True)
    print(f"Voted angle:   {voted_angle:.4f} deg")
    print(f"Saved {output_path}")

    if plot:
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))

        # Both panels show the same pixel data. The correction is in the WCS,
        # not the image, so overlaying WCS grid lines would reveal the difference.
        for ax, img, title in [
            (axes[0], image, "Detranslated needle (original WCS)"),
            (axes[1], image, "Corrected needle (updated WCS)"),
        ]:
            ax.imshow(img, cmap='viridis', origin='lower',
                      vmin=np.percentile(img, 1), vmax=np.percentile(img, 99))
            ax.set_title(f"{title} -- pair {pair_num}")
            ax.axis('off')

        plt.suptitle(f"WCS derotation  ({voted_angle:.4f} deg)", y=1.01)
        plt.tight_layout()
        plt.show()

    return image   # pixel data is unchanged; the correction lives in the saved header


if __name__ == "__main__":
    import importlib
    solve_rotation = importlib.import_module("04_solve_rotation")
    voted_angle, _ = solve_rotation.main(pair_dir=base)
    main(pair_dir=base, voted_angle=voted_angle, plot=True)
