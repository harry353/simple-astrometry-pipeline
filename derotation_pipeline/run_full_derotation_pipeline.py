import sys
import os

DEROTATION_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "derotation_pipeline")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, DEROTATION_DIR)

import time
import constants_astrometry as constants
import importlib
detect_centroids            = importlib.import_module("01_detect_centroids")
match_centroids             = importlib.import_module("02_match_centroids")
build_triangles             = importlib.import_module("03_build_triangles")
solve_rotation              = importlib.import_module("04_solve_rotation")
apply_derotation            = importlib.import_module("05_apply_derotation")
print_derotation_statistics = importlib.import_module("06_print_derotation_statistics")

DATASET_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data_generation", "dataset")


def get_pair_dirs():
    dirs = sorted(
        d for d in os.listdir(DATASET_DIR)
        if d.startswith("pair_") and os.path.isdir(os.path.join(DATASET_DIR, d))
    )
    n = constants.TRANSLATION_PIPELINE_PAIRS
    return dirs if n == 0 else dirs[:n]


def main():
    pair_dirs = get_pair_dirs()
    print(f"Running derotation pipeline on {len(pair_dirs)} pair(s)...\n")

    t_start = time.perf_counter()
    for pair_name in pair_dirs:
        pair_dir = os.path.join(DATASET_DIR, pair_name)
        pair_num = pair_name.split("_")[1]

        print(f"{'='*60}")
        print(f"  Pair {pair_num}")
        print(f"{'='*60}")

        detect_centroids.main(pair_dir=pair_dir)
        match_centroids.main(pair_dir=pair_dir)
        if not build_triangles.main(pair_dir=pair_dir):
            print(f"Skipping solve/apply for pair {pair_num} (insufficient matched points).")
            continue
        voted_angle, refined_angle = solve_rotation.main(pair_dir=pair_dir)
        apply_derotation.main(pair_dir=pair_dir, voted_angle=voted_angle)
        print()
    total = time.perf_counter() - t_start

    print(f"Total time: {total:.2f}s | Average per pair: {total / len(pair_dirs):.2f}s\n")

    records = print_derotation_statistics.collect_statistics()
    print_derotation_statistics.print_statistics(records)


if __name__ == "__main__":
    main()
