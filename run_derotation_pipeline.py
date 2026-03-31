import sys
import os

DEROTATION_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "derotation_pipeline")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, DEROTATION_DIR)

import constants
import detect_centroids
import match_centroids
import build_triangles
import solve_rotation
import apply_derotation

DATASET_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data_generation", "dataset")


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

    for pair_name in pair_dirs:
        pair_dir = os.path.join(DATASET_DIR, pair_name)
        pair_num = pair_name.split("_")[1]

        print(f"{'='*60}")
        print(f"  Pair {pair_num}")
        print(f"{'='*60}")

        detect_centroids.main(pair_dir=pair_dir)
        match_centroids.main(pair_dir=pair_dir)
        build_triangles.main(pair_dir=pair_dir)
        voted_angle, refined_angle = solve_rotation.main(pair_dir=pair_dir)
        apply_derotation.main(pair_dir=pair_dir, voted_angle=voted_angle)
        print()


if __name__ == "__main__":
    main()
