import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import pickle
import numpy as np
import pandas as pd
from scipy.spatial import KDTree

PAIR_NUM = "0001"

base = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data_generation", "dataset", f"pair_{PAIR_NUM}")


def build_kdtree(haystack_quads_path):
    t_start = time.perf_counter()

    df = pd.read_csv(haystack_quads_path)
    print(f"Loaded {len(df)} haystack quads from {haystack_quads_path}")

    # The 4D hash space: (Cx, Cy, Dx, Dy)
    hashes = df[['Cx', 'Cy', 'Dx', 'Dy']].to_numpy()

    t_build = time.perf_counter()
    tree = KDTree(hashes)
    t_build = time.perf_counter() - t_build

    t_total = time.perf_counter() - t_start

    # ── Print results ──────────────────────────────────────────────────────────
    print(f"\n{'='*72}")
    print(f"  k-d tree — haystack pair {PAIR_NUM}")
    print(f"{'='*72}")

    print(f"\n  Index")
    print(f"  {'-'*68}")
    print(f"  {'Dimensions':<32}  4  (Cx, Cy, Dx, Dy)")
    print(f"  {'Points indexed':<32}  {len(df)}")
    print(f"  {'Build time':<32}  {t_build*1e3:.2f} ms")
    print(f"  {'Total time':<32}  {t_total*1e3:.2f} ms")

    print(f"\n  Hash space ranges")
    print(f"  {'-'*68}")
    print(f"  {'coord':<10} {'min':>10}  {'max':>10}  {'std':>10}")
    print(f"  {'-'*44}")
    for col in ['Cx', 'Cy', 'Dx', 'Dy']:
        v = df[col]
        print(f"  {col:<10} {v.min():>+10.4f}  {v.max():>+10.4f}  {v.std():>10.4f}")

    print(f"\n{'='*72}\n")

    # ── Save ──────────────────────────────────────────────────────────────────
    pair_dir  = os.path.dirname(haystack_quads_path)
    pair_num  = os.path.basename(pair_dir).split("_")[1]
    tree_path = os.path.join(pair_dir, f"kdtree_haystack_{pair_num}.pkl")

    with open(tree_path, "wb") as f:
        pickle.dump({'tree': tree, 'df': df}, f)

    print(f"Saved k-d tree → {tree_path}")

    return tree, df


if __name__ == "__main__":
    build_kdtree(
        haystack_quads_path=os.path.join(base, f"quads_haystack_{PAIR_NUM}.csv"),
    )
