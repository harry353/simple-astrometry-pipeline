import sys
import os

QUADS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "quads_pipeline")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, QUADS_DIR)

import time
import constants
import importlib
hash_quads    = importlib.import_module("01_hash_quads")
build_kdtree  = importlib.import_module("02_build_kdtree")
match_quads   = importlib.import_module("03_match_quads")
fit_transform = importlib.import_module("04_fit_transform")

DATASET_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data_generation", "dataset")


def get_pair_dirs():
    dirs = sorted(
        d for d in os.listdir(DATASET_DIR)
        if d.startswith("pair_") and os.path.isdir(os.path.join(DATASET_DIR, d))
    )
    n = constants.TRANSLATION_PIPELINE_PAIRS
    return dirs if n == 0 else dirs[:n]


def process_pair(pair_name):
    pair_num      = pair_name.split("_")[1]
    pair_dir      = os.path.join(DATASET_DIR, pair_name)
    haystack_path = os.path.join(pair_dir, f"haystack_{pair_num}.fits")
    needle_path   = os.path.join(pair_dir, f"needle_{pair_num}.fits")

    # Step 1 — hash quads; skip CSV writes (haystack can be millions of rows)
    (_, haystack_df) = hash_quads.hash_image(haystack_path, label="haystack", save=False)
    (_, needle_df)   = hash_quads.hash_image(needle_path,   label="needle",   save=False)

    # Step 2 — build k-d tree in-memory; skip pkl write
    tree, haystack_df = build_kdtree.build_kdtree(df=haystack_df, save=False)

    # Step 3 — match needle quads; pass everything in-memory
    candidates = match_quads.match_quads(
        needle_df=needle_df, tree=tree, haystack_df=haystack_df, save=False,
    )

    if candidates.empty:
        print(f"  No candidates found for pair {pair_num} — skipping fit.")
        return pair_num, None

    # Step 4 — fit similarity transform
    _, gt, refit = fit_transform.fit_transforms(
        candidates_df=candidates,
        needle_fits_path=needle_path,
        haystack_fits_path=haystack_path,
        save=False,
    )

    return pair_num, refit


def main():
    pair_dirs = get_pair_dirs()
    print(f"Running quads pipeline on {len(pair_dirs)} pair(s)...\n")

    t_start  = time.perf_counter()
    results  = []

    for i, pair_name in enumerate(pair_dirs, 1):
        pair_num = pair_name.split("_")[1]
        print(f"{'='*60}")
        print(f"  Pair {pair_num}  ({i}/{len(pair_dirs)})")
        print(f"{'='*60}")

        pair_num, refit = process_pair(pair_name)
        results.append((pair_num, refit))
        print()

    total = time.perf_counter() - t_start

    # ── Summary ───────────────────────────────────────────────────────────────
    solved   = [(p, r) for p, r in results if r is not None]
    unsolved = [p for p, r in results if r is None]

    print(f"{'='*60}")
    print(f"  Quads pipeline summary")
    print(f"{'='*60}")
    print(f"  Pairs processed   : {len(results)}")
    print(f"  Solved            : {len(solved)}")
    print(f"  No candidates     : {len(unsolved)}")
    if unsolved:
        print(f"  Unsolved pairs    : {', '.join(unsolved)}")

    if solved:
        import numpy as np
        inliers  = [r['n_inliers'] for _, r in solved]
        residuals = [r['residual'] for _, r in solved]
        print(f"\n  Re-fit statistics (over {len(solved)} solved pairs)")
        print(f"  {'-'*48}")
        print(f"  {'Inliers   mean/min/max':<32}  "
              f"{np.mean(inliers):.1f} / {np.min(inliers)} / {np.max(inliers)}")
        print(f"  {'Residual  mean/min/max':<32}  "
              f"{np.mean(residuals):.3f} / {np.min(residuals):.3f} / {np.max(residuals):.3f}")

    print(f"\n  Total time        : {total:.2f}s")
    if results:
        print(f"  Avg per pair      : {total / len(results):.2f}s")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
