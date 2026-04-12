import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import datetime
import numpy as np


def generate_diagnostics(results, failures, dt_start, dt_end, total, dataset_dir):
    """
    Write a diagnostics file and print the run summary to stdout.

    Args:
        results   : list of (pair_num, refit)  — refit is None for failed pairs
        failures  : list of (pair_num, reason)
        dt_start  : datetime of pipeline start
        dt_end    : datetime of pipeline end
        total     : total elapsed seconds (float)
        dataset_dir : directory where the diagnostics file is written
    """
    # ── Diagnostics file ──────────────────────────────────────────────────────
    diag_path = os.path.join(
        dataset_dir,
        f"quads_diagnostics_{dt_end.strftime('%Y%m%d_%H%M%S')}.txt",
    )
    with open(diag_path, "w") as f:
        f.write(f"Quads pipeline diagnostics — {dt_end.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Started         : {dt_start.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Ended           : {dt_end.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Pairs processed : {len(results)}\n")
        f.write(f"Solved          : {len(results) - len(failures)}\n")
        f.write(f"Failed          : {len(failures)}\n\n")
        if failures:
            f.write(f"{'pair':<12}  reason\n")
            f.write(f"{'-'*60}\n")
            for p, reason in failures:
                f.write(f"pair_{p}      {reason}\n")
        else:
            f.write("All pairs solved.\n")
    print(f"  Diagnostics written → {diag_path}")

    # ── Summary ───────────────────────────────────────────────────────────────
    solved   = [(p, r) for p, r in results if r is not None]
    unsolved = [p for p, r in results if r is None]

    print(f"{'='*60}")
    print(f"  Quads pipeline summary")
    print(f"{'='*60}")
    print(f"  Pairs processed   : {len(results)}")
    print(f"  Solved            : {len(solved)}")
    print(f"  Failed            : {len(unsolved)}")
    if unsolved:
        print(f"  Failed pairs      : {', '.join(unsolved)}")

    if solved:
        params = [
            ("err scale",  [r['err_scale'] for _, r in solved], "{:.4f}"),
            ("err angle°", [r['err_angle'] for _, r in solved], "{:.4f}"),
            ("err tx (px)",[r['err_tx']    for _, r in solved], "{:.2f}"),
            ("err ty (px)",[r['err_ty']    for _, r in solved], "{:.2f}"),
            ("inliers",    [r['n_inliers'] for _, r in solved], "{:.1f}"),
        ]
        col_w = 12
        print(f"\n  Re-fit statistics ({len(solved)} solved pairs)")
        print(f"  {'parameter':<14}  {'mean':>{col_w}}  {'std':>{col_w}}  {'min':>{col_w}}  {'max':>{col_w}}")
        print(f"  {'-'*14}  {'-'*col_w}  {'-'*col_w}  {'-'*col_w}  {'-'*col_w}")
        for label, vals, fmt in params:
            arr = np.array(vals)
            print(f"  {label:<14}  "
                  f"{fmt.format(np.mean(arr)):>{col_w}}  "
                  f"{fmt.format(np.std(arr)):>{col_w}}  "
                  f"{fmt.format(np.min(arr)):>{col_w}}  "
                  f"{fmt.format(np.max(arr)):>{col_w}}")

    print(f"\n  Total time        : {total:.2f}s")
    if results:
        print(f"  Avg per pair      : {total / len(results):.2f}s")
    print(f"{'='*60}\n")
