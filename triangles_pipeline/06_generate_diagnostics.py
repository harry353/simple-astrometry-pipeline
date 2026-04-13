import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import datetime
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


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
        f"triangles_diagnostics_{dt_end.strftime('%Y%m%d_%H%M%S')}.txt",
    )
    with open(diag_path, "w") as f:
        f.write(f"Triangles pipeline diagnostics — {dt_end.strftime('%Y-%m-%d %H:%M:%S')}\n")
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
    print(f"  Triangles pipeline summary")
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

    # ── Residual frequency collage ────────────────────────────────────────────
    if solved:
        project_dir = os.path.dirname(os.path.dirname(dataset_dir))
        plot_path   = os.path.join(
            project_dir,
            f"triangles_residuals_{dt_end.strftime('%Y%m%d_%H%M%S')}.png",
        )

        ARCSEC_PER_PX = 0.2

        panels = [
            ("err_scale",   "Scale error",          "Δscale",         "steelblue",   False),
            ("err_angle",   "Angle error (°)",       "Δangle (°)",     "darkorange",  False),
            ("err_tx",      "Translation X (px)",    "Δtx (px)",       "seagreen",    True),
            ("err_ty",      "Translation Y (px)",    "Δty (px)",       "crimson",     True),
        ]

        fig, axes = plt.subplots(2, 2, figsize=(10, 8))
        fig.suptitle(
            f"Residual error distributions — {len(solved)} solved pairs\n"
            f"{dt_end.strftime('%Y-%m-%d %H:%M:%S')}",
            fontsize=13,
        )

        for ax, (key, title, xlabel, color, has_arcsec) in zip(axes.flat, panels):
            vals = np.array([r[key] for _, r in solved])

            auto_edges = np.histogram_bin_edges(vals, bins="auto")
            n_bins     = max(1, (len(auto_edges) - 1) * 2)
            ax.hist(vals, bins=n_bins, range=(auto_edges[0], auto_edges[-1]),
                    color=color, edgecolor="white", linewidth=0.5, alpha=0.7)

            mean_val = np.mean(vals)
            ax.axvline(mean_val, color="black", linestyle="--", linewidth=1,
                       label=f"mean={mean_val:.4g}")
            ax.set_title(title)
            ax.set_xlabel(xlabel)
            ax.set_ylabel("count")
            ax.legend(fontsize=8)

            if has_arcsec:
                ax_top = ax.twiny()
                lo, hi = ax.get_xlim()
                ax_top.set_xlim(lo * ARCSEC_PER_PX, hi * ARCSEC_PER_PX)
                ax_top.set_xlabel('arcsec', fontsize=8)

        fig.tight_layout()
        fig.savefig(plot_path, dpi=150)
        plt.close(fig)
        print(f"  Residual plot   → {plot_path}")
