import sys
import os
import shutil

DATA_GENERATION_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data_generation")
sys.path.insert(0, DATA_GENERATION_DIR)

import random
import time
import gc
import functools
import constants
import create_haystack
import create_needle
import create_data_generation_diagnostics
from concurrent.futures import ProcessPoolExecutor, as_completed


def timed(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        num_pairs = kwargs.get('num_pairs', args[0] if args else 1)
        print(f"\n=== Dataset generation complete! ===")
        print(f"Total time: {elapsed:.1f}s ({elapsed/num_pairs:.2f}s per pair)")
        return result
    return wrapper


def generate_single_pair(i, root_dir, save_clean):
    """Worker function to generate a single haystack/needle pair."""
    # Ensure a unique random seed for each process
    pair_seed = random.randint(1, 1000000) + i
    constants.RANDOM_SEED = pair_seed

    pair_name = f"{i:04d}"
    pair_dir = os.path.join(root_dir, f"pair_{pair_name}")

    if not os.path.exists(pair_dir):
        os.makedirs(pair_dir)

    # 1. Create Haystack
    haystack_prefix = f"haystack_{pair_name}"
    h_clean, h_header = create_haystack.main(output_dir=pair_dir,
                                             filename_prefix=haystack_prefix,
                                             save_clean=save_clean)

    # 2. Create Needle
    needle_prefix = f"needle_{pair_name}"
    create_needle.main(output_dir=pair_dir, filename_prefix=needle_prefix,
                       haystack_dir=pair_dir, haystack_prefix=haystack_prefix,
                       haystack_clean=h_clean, header_haystack=h_header)

    # Explicitly free memory
    del h_clean
    del h_header
    gc.collect()
    return i


def setup_output_dir(root_dir):
    if os.path.exists(root_dir):
        shutil.rmtree(root_dir)
        print(f"Cleared existing data in {root_dir}")
    os.makedirs(root_dir)
    print(f"Created root directory: {root_dir}")


def run_parallel_generation(num_pairs, root_dir, save_clean, num_cores):
    print(f"Starting parallel generation of {num_pairs} pairs using {num_cores} cores...")
    total_start = time.time()
    completed = 0

    with ProcessPoolExecutor(max_workers=num_cores) as executor:
        futures = {executor.submit(generate_single_pair, i, root_dir, save_clean): i for i in range(1, num_pairs + 1)}

        for future in as_completed(futures):
            completed += 1
            if completed % 10 == 0:
                elapsed = time.time() - total_start
                avg_time = elapsed / completed
                remaining = avg_time * (num_pairs - completed)
                print(f"Progress: {completed}/{num_pairs} pairs done. Elapsed: {elapsed:.1f}s, Est. remaining: {remaining:.1f}s")


DATA_GENERATION_DIR     = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data_generation")
DATA_GENERATION_DATASET = os.path.join(DATA_GENERATION_DIR, "dataset")


@timed
def main(num_pairs=constants.NUM_PAIRS, root_dir=DATA_GENERATION_DATASET, save_clean=False):
    num_cores = getattr(constants, 'NUM_CORES', 4)

    setup_output_dir(root_dir)
    run_parallel_generation(num_pairs, root_dir, save_clean, num_cores)
    create_data_generation_diagnostics.main(
        root_dir=root_dir,
        output_path=os.path.join(DATA_GENERATION_DIR, "diagnostics.png")
    )


if __name__ == "__main__":
    main(save_clean=False)
