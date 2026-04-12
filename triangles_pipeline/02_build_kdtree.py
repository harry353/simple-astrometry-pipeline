import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import pickle
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

PAIR_NUM = "0001"

base = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data_generation", "dataset", f"pair_{PAIR_NUM}")


def build_kdtree(haystack_triangles_path=None, df=None, save=True):
    t_start = time.perf_counter()

    if df is None:
        df = pd.read_csv(haystack_triangles_path)
        print(f"Loaded {len(df)} haystack triangles from {haystack_triangles_path}")
    else:
        print(f"Using {len(df)} haystack triangles (in-memory)")

    # The 2D hash space: (Cx, Cy)
    hashes = df[['Cx', 'Cy']].to_numpy()

    t_build = time.perf_counter()
    tree = cKDTree(hashes)
    t_build = time.perf_counter() - t_build

    t_total = time.perf_counter() - t_start

    # ── Print results ──────────────────────────────────────────────────────────
    import constants_astrometry
    if constants_astrometry.VERBOSE:
        print(f"\n{'='*72}")
        print(f"  k-d tree — haystack pair {PAIR_NUM}")
        print(f"{'='*72}")

        print(f"\n  Index")
        print(f"  {'-'*68}")
        print(f"  {'Dimensions':<32}  2  (Cx, Cy)")
        print(f"  {'Points indexed':<32}  {len(df)}")
        print(f"  {'Build time':<32}  {t_build*1e3:.2f} ms")
        print(f"  {'Total time':<32}  {t_total*1e3:.2f} ms")

        print(f"\n  Hash space ranges")
        print(f"  {'-'*68}")
        print(f"  {'coord':<10} {'min':>10}  {'max':>10}  {'std':>10}")
        print(f"  {'-'*44}")
        for col in ['Cx', 'Cy']:
            v = df[col]
            print(f"  {col:<10} {v.min():>+10.4f}  {v.max():>+10.4f}  {v.std():>10.4f}")

        print(f"\n{'='*72}\n")
    else:
        stds = {col: df[col].std() for col in ['Cx', 'Cy']}
        print(f"  [kdtree] {len(df)} points  ({t_build*1e3:.0f} ms)  "
              f"stds: Cx={stds['Cx']:.4f}  Cy={stds['Cy']:.4f}")

    # ── Save ──────────────────────────────────────────────────────────────────
    if save and haystack_triangles_path is not None:
        pair_dir  = os.path.dirname(haystack_triangles_path)
        pair_num  = os.path.basename(pair_dir).split("_")[1]
        tree_path = os.path.join(pair_dir, f"kdtree_haystack_{pair_num}.pkl")
        with open(tree_path, "wb") as f:
            pickle.dump({'tree': tree, 'df': df}, f)
        print(f"Saved k-d tree → {tree_path}")

    return tree, df


if __name__ == "__main__":
    build_kdtree(
        haystack_triangles_path=os.path.join(base, f"triangles_haystack_{PAIR_NUM}.csv"),
    )
