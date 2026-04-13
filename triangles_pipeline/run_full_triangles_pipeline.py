import sys
import os

TRIANGLES_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR      = os.path.dirname(TRIANGLES_DIR)
sys.path.insert(0, ROOT_DIR)
sys.path.insert(0, TRIANGLES_DIR)

import time
import datetime
import constants_astrometry
import importlib
hash_triangles       = importlib.import_module("01_hash_triangles")
build_kdtree         = importlib.import_module("02_build_kdtree")
match_triangles      = importlib.import_module("03_match_triangles")
fit_candidates_mod   = importlib.import_module("04a_fit_candidates")
fit_ransac_mod       = importlib.import_module("04b_fit_ransac")
generate_diagnostics = importlib.import_module("05_generate_diagnostics")

DATASET_DIR = os.path.join(ROOT_DIR, "data_generation", "dataset")


def get_pair_dirs():
    dirs = sorted(
        d for d in os.listdir(DATASET_DIR)
        if d.startswith("pair_") and os.path.isdir(os.path.join(DATASET_DIR, d))
    )
    n = constants_astrometry.TRANSLATION_PIPELINE_PAIRS
    return dirs if n == 0 else dirs[:n]


def process_pair(pair_name):
    """Returns (pair_num, refit, failure_reason).  failure_reason is None on success."""
    pair_num      = pair_name.split("_")[1]
    pair_dir      = os.path.join(DATASET_DIR, pair_name)
    haystack_path = os.path.join(pair_dir, f"haystack_{pair_num}.fits")
    needle_path   = os.path.join(pair_dir, f"needle_{pair_num}.fits")

    # Step 1 — hash triangles; skip CSV writes (haystack can be millions of rows)
    (haystack_df, haystack_centroids) = hash_triangles.hash_image(haystack_path, label="haystack", save=False)
    (needle_df,   needle_centroids)   = hash_triangles.hash_image(needle_path,   label="needle",   save=False)

    if needle_df.empty:
        n = len(needle_centroids)
        reason = f"only {n}/3 sources detected in needle"
        print(f"  No triangles formed for needle {pair_num} — skipping pair.")
        return pair_num, None, reason

    # Step 2 — build k-d tree in-memory; skip pkl write
    tree, haystack_df = build_kdtree.build_kdtree(df=haystack_df, save=False)

    # Step 3 — match needle triangles; pass everything in-memory
    candidates = match_triangles.match_triangles(
        needle_df=needle_df, tree=tree, haystack_df=haystack_df, save=False,
    )

    if candidates.empty:
        print(f"  No candidates found for pair {pair_num} — skipping fit.")
        return pair_num, None, "no matches found in haystack"

    # Step 4a — vectorized similarity fit for all candidate pairs
    fitted_df, gt = fit_candidates_mod.fit_candidates(
        candidates_df=candidates,
        needle_fits_path=needle_path,
        haystack_fits_path=haystack_path,
        save=False,
    )

    # Step 4b — RANSAC hypothesis selection + iterative inlier refit
    _, gt, refit = fit_ransac_mod.fit_ransac(
        candidates_df=fitted_df,
        needle_fits_path=needle_path,
        haystack_fits_path=haystack_path,
        needle_centroids=needle_centroids,
        haystack_centroids=haystack_centroids,
        gt=gt,
        save=False,
    )

    if refit is None:
        return pair_num, None, "insufficient inliers for transform refit"

    refit['err_scale'] = abs(refit['scale']     - gt['true_scale'])
    refit['err_angle'] = abs(refit['angle_deg'] - gt['true_angle_deg'])
    refit['err_tx']    = abs(refit['tx']        - gt['true_tx'])
    refit['err_ty']    = abs(refit['ty']        - gt['true_ty'])

    return pair_num, refit, None


def main():
    pair_dirs = get_pair_dirs()
    print(f"Running triangles pipeline on {len(pair_dirs)} pair(s)...\n")

    t_start   = time.perf_counter()
    dt_start  = datetime.datetime.now()
    results   = []
    failures  = []   # list of (pair_num, reason)

    for i, pair_name in enumerate(pair_dirs, 1):
        pair_num = pair_name.split("_")[1]
        print(f"{'='*60}")
        print(f"  Pair {pair_num}  ({i}/{len(pair_dirs)})")
        print(f"{'='*60}")

        t_pair = time.perf_counter()
        pair_num, refit, failure_reason = process_pair(pair_name)
        t_pair = time.perf_counter() - t_pair
        results.append((pair_num, refit))
        if failure_reason is not None:
            failures.append((pair_num, failure_reason))
        iters_str = f", {refit['n_iters']} iterations" if refit is not None else ""
        print(f"  Pair {pair_num} done  ({t_pair:.2f}s{iters_str})\n")

    total  = time.perf_counter() - t_start
    dt_end = datetime.datetime.now()

    generate_diagnostics.generate_diagnostics(
        results=results,
        failures=failures,
        dt_start=dt_start,
        dt_end=dt_end,
        total=total,
        dataset_dir=DATASET_DIR,
    )


if __name__ == "__main__":
    main()
