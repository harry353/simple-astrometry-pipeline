import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS

DATASET_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data_generation", "dataset")


def collect_statistics():
    records = []

    pair_dirs = sorted(
        d for d in os.listdir(DATASET_DIR)
        if d.startswith("pair_") and os.path.isdir(os.path.join(DATASET_DIR, d))
    )

    for pair_name in pair_dirs:
        pair_num = pair_name.split("_")[1]
        pair_dir = os.path.join(DATASET_DIR, pair_name)

        haystack_path       = os.path.join(pair_dir, f"haystack_{pair_num}.fits")
        needle_path         = os.path.join(pair_dir, f"needle_{pair_num}.fits")
        detranslated_needle_path = os.path.join(pair_dir, f"detranslated_needle_{pair_num}.fits")

        if not all(os.path.exists(p) for p in [haystack_path, needle_path, detranslated_needle_path]):
            continue

        with fits.open(haystack_path) as hdul:
            header_haystack = hdul[0].header
        with fits.open(needle_path) as hdul:
            header_needle = hdul[0].header
        with fits.open(detranslated_needle_path) as hdul:
            header_corrected = hdul[0].header

        wcs_h = WCS(header_haystack)
        px_scale = abs(header_haystack.get('CDELT2', 0)) * 3600  # arcsec/px

        # True needle centre in haystack pixels
        true_cx, true_cy = wcs_h.world_to_pixel_values(header_needle['CRVAL1'], header_needle['CRVAL2'])

        # Detected centre (from corrected CRVAL)
        det_cx, det_cy = wcs_h.world_to_pixel_values(header_corrected['CRVAL1'], header_corrected['CRVAL2'])

        # Initial WCS error (pre-correction)
        wcs_err_x = header_needle.get('WERR_XPX', 0)
        wcs_err_y = header_needle.get('WERR_YPX', 0)

        residual_dx   = float(det_cx - true_cx)
        residual_dy   = float(det_cy - true_cy)
        residual_dist = float(np.sqrt(residual_dx**2 + residual_dy**2))
        initial_dist  = float(np.sqrt(wcs_err_x**2 + wcs_err_y**2))

        records.append({
            'pair':          pair_num,
            'initial_dx':    float(wcs_err_x),
            'initial_dy':    float(wcs_err_y),
            'initial_dist':  initial_dist,
            'residual_dx':   residual_dx,
            'residual_dy':   residual_dy,
            'residual_dist': residual_dist,
            'px_scale':      px_scale,
        })

    return records


def print_statistics(records):
    if not records:
        print("No processed pairs found. Run run_detranslation_pipeline.py first.")
        return

    def col(vals):
        a = np.array(vals)
        return a.mean(), a.std(), a.min(), a.max()

    initial_dists  = [r['initial_dist']  for r in records]
    residual_dists = [r['residual_dist'] for r in records]
    residual_dxs   = [r['residual_dx']   for r in records]
    residual_dys   = [r['residual_dy']   for r in records]
    px_scale       = records[0]['px_scale']

    def fmt(val_px):
        return f"{val_px:+.4f} px  ({val_px * px_scale:+.4f} arcsec)"

    print(f"\n{'='*62}")
    print(f"  Detranslation statistics — {len(records)} pairs")
    print(f"{'='*62}")

    print(f"  {'Pair':<8} {'init err (px)':>14}  {'res dx (px)':>12}  {'res dy (px)':>12}  {'res dist (px)':>14}")
    print(f"  {'-'*66}")
    for r in records:
        print(f"  {r['pair']:<8} {r['initial_dist']:>14.4f}  {r['residual_dx']:>+12.4f}  {r['residual_dy']:>+12.4f}  {r['residual_dist']:>14.4f}")

    print(f"\n  {'Metric':<32} {'mean':>10}  {'std':>8}  {'min':>8}  {'max':>8}")
    print(f"  {'-'*60}")

    for label, vals, signed in [
        ("Initial WCS error (px)",  initial_dists,  False),
        ("Residual error (px)",     residual_dists, False),
        ("Residual dx (px)",        residual_dxs,   True),
        ("Residual dy (px)",        residual_dys,   True),
    ]:
        mean, std, mn, mx = col(vals)
        fmt_str = f"{mean:>+10.4f}" if signed else f"{mean:>10.4f}"
        print(f"  {label:<32} {fmt_str}  {std:>8.4f}  {mn:>+8.4f}  {mx:>+8.4f}")

    mean_initial  = np.mean(initial_dists)
    mean_residual = np.mean(residual_dists)
    improvement   = (1 - mean_residual / mean_initial) * 100 if mean_initial > 0 else 0

    print(f"  {'-'*60}")
    print(f"  {'Mean improvement':<32} {improvement:>10.1f}%")
    print(f"  {'Pixel scale':<32} {px_scale:>10.4f} arcsec/px")
    print(f"  Residual error in arcsec:  mean={mean_residual*px_scale:.4f}  "
          f"max={max(residual_dists)*px_scale:.4f}")
    print(f"{'='*62}\n")


if __name__ == "__main__":
    records = collect_statistics()
    print_statistics(records)
