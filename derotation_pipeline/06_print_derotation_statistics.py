import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from astropy.io import fits

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

        needle_path    = os.path.join(pair_dir, f"needle_{pair_num}.fits")
        corrected_path = os.path.join(pair_dir, f"corrected_needle_{pair_num}.fits")

        if not all(os.path.exists(p) for p in [needle_path, corrected_path]):
            continue

        with fits.open(needle_path) as f:
            header_needle = f[0].header
        with fits.open(corrected_path) as f:
            header_corrected = f[0].header

        # True rotation applied to the needle during data generation
        true_angle = header_needle.get('NANGLE', None)
        if true_angle is None:
            continue

        # Rotation correction we estimated and applied
        applied_angle = header_corrected.get('DROT_ANG', None)
        if applied_angle is None:
            continue

        # The pipeline estimates the rotation from candidate → detranslated, which
        # equals the needle's rotation error (NANGLE). A perfect correction would
        # give applied_angle == true_angle, so the residual is their difference.
        residual = applied_angle - true_angle

        records.append({
            'pair':          pair_num,
            'true_angle':    float(true_angle),
            'applied_angle': float(applied_angle),
            'residual':      float(residual),
        })

    return records


def print_statistics(records):
    if not records:
        print("No processed pairs found. Run run_derotation_pipeline.py first.")
        return

    true_angles    = [r['true_angle']    for r in records]
    applied_angles = [r['applied_angle'] for r in records]
    residuals      = [r['residual']      for r in records]

    print(f"\n{'='*62}")
    print(f"  Derotation statistics — {len(records)} pairs")
    print(f"{'='*62}")

    # Per-pair table
    print(f"  {'Pair':<8} {'true (deg)':>12}  {'applied (deg)':>14}  {'residual (deg)':>15}")
    print(f"  {'-'*54}")
    for r in records:
        print(f"  {r['pair']:<8} {r['true_angle']:>+12.4f}  {r['applied_angle']:>+14.4f}  {r['residual']:>+15.4f}")

    # Aggregate statistics
    def col(vals):
        a = np.array(vals)
        return a.mean(), a.std(), a.min(), a.max()

    print(f"\n  {'Metric':<32} {'mean':>10}  {'std':>8}  {'min':>8}  {'max':>8}")
    print(f"  {'-'*60}")

    for label, vals in [
        ("True angle (deg)",     true_angles),
        ("Applied angle (deg)",  applied_angles),
        ("Residual error (deg)", residuals),
    ]:
        mean, std, mn, mx = col(vals)
        print(f"  {label:<32} {mean:>+10.4f}  {std:>8.4f}  {mn:>+8.4f}  {mx:>+8.4f}")

    mean_abs_residual = np.mean(np.abs(residuals))
    mean_abs_true     = np.mean(np.abs(true_angles))
    improvement       = (1 - mean_abs_residual / mean_abs_true) * 100 if mean_abs_true > 0 else 0

    print(f"  {'-'*60}")
    print(f"  {'Mean |residual|':<32} {mean_abs_residual:>10.4f} deg")
    print(f"  {'Mean improvement':<32} {improvement:>10.1f}%")
    print(f"{'='*62}\n")


if __name__ == "__main__":
    records = collect_statistics()
    print_statistics(records)
