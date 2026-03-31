import sys
import os

DETRANSLATION_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "detranslation_pipeline")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, DETRANSLATION_DIR)

import constants
import importlib
apply_matched_filtering     = importlib.import_module("01_apply_matched_filtering")
export_candidate_needle     = importlib.import_module("02_export_candidate_needle")
correct_needle_wcs          = importlib.import_module("03_correct_needle_wcs")
print_detranslation_statistics = importlib.import_module("04_print_detranslation_statistics")

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
    print(f"Running detranslation pipeline on {len(pair_dirs)} pair(s)...\n")

    for pair_name in pair_dirs:
        pair_num      = pair_name.split("_")[1]
        pair_dir      = os.path.join(DATASET_DIR, pair_name)
        haystack_path = os.path.join(pair_dir, f"haystack_{pair_num}.fits")
        needle_path   = os.path.join(pair_dir, f"needle_{pair_num}.fits")

        print(f"{'='*60}")
        print(f"  Pair {pair_num}")
        print(f"{'='*60}")

        shift_x, shift_y = apply_matched_filtering.main(haystack_path, needle_path)
        export_candidate_needle.main(haystack_path, needle_path, shift_x, shift_y)
        correct_needle_wcs.main(haystack_path, needle_path, shift_x, shift_y)
        print()

    records = print_detranslation_statistics.collect_statistics()
    print_detranslation_statistics.print_statistics(records)


if __name__ == "__main__":
    main()
