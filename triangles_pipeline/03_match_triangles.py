import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import pickle
import numpy as np
import pandas as pd

PAIR_NUM  = "0001"

base = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data_generation", "dataset", f"pair_{PAIR_NUM}")


def match_triangles(needle_triangles_path=None, haystack_tree_path=None,
                    needle_df=None, tree=None, haystack_df=None, save=True):
    t_start = time.perf_counter()

    # ── Load inputs ───────────────────────────────────────────────────────────
    if needle_df is None:
        needle_df = pd.read_csv(needle_triangles_path)
    if tree is None or haystack_df is None:
        with open(haystack_tree_path, "rb") as f:
            payload = pickle.load(f)
        tree        = payload['tree']
        haystack_df = payload['df']

    import constants_astrometry
    if constants_astrometry.VERBOSE:
        print(f"Needle triangles  : {len(needle_df)}")
        print(f"Haystack triangles: {len(haystack_df)}")
        print(f"Tolerance         : {constants_astrometry.TRIANGLE_MATCH_TOLERANCE}  (2D hash-space radius)")

    # ── Query k-d tree ────────────────────────────────────────────────────────
    needle_hashes   = needle_df[['Cx', 'Cy']].to_numpy()
    haystack_hashes = haystack_df[['Cx', 'Cy']].to_numpy()

    t_query = time.perf_counter()
    hit_indices = tree.query_ball_point(needle_hashes, r=constants_astrometry.TRIANGLE_MATCH_TOLERANCE)
    t_query = time.perf_counter() - t_query

    # ── Collect candidate pairs (vectorized) ──────────────────────────────────
    # Flatten hit_indices into two parallel index arrays, then slice DataFrames once.
    ni_flat = np.array([ni for ni, hits in enumerate(hit_indices) for _ in hits], dtype=np.int64)
    hi_flat = np.array([hi for hits in hit_indices for hi in hits],               dtype=np.int64)

    rows = None
    if len(ni_flat) > 0:
        n_cols = needle_df.iloc[ni_flat].reset_index(drop=True)
        h_cols = haystack_df.iloc[hi_flat].reset_index(drop=True)

        hash_diffs  = needle_hashes[ni_flat] - haystack_hashes[hi_flat]
        hash_dists  = np.linalg.norm(hash_diffs, axis=1)

        row_data = {
            'needle_tri_idx':   ni_flat,
            'haystack_tri_idx': hi_flat,
            'n_A_px_x': n_cols['A_px_x'].to_numpy(), 'n_A_px_y': n_cols['A_px_y'].to_numpy(),
            'n_B_px_x': n_cols['B_px_x'].to_numpy(), 'n_B_px_y': n_cols['B_px_y'].to_numpy(),
            'n_Cx': n_cols['Cx'].to_numpy(), 'n_Cy': n_cols['Cy'].to_numpy(),
            'h_A_px_x': h_cols['A_px_x'].to_numpy(), 'h_A_px_y': h_cols['A_px_y'].to_numpy(),
            'h_B_px_x': h_cols['B_px_x'].to_numpy(), 'h_B_px_y': h_cols['B_px_y'].to_numpy(),
            'h_Cx': h_cols['Cx'].to_numpy(), 'h_Cy': h_cols['Cy'].to_numpy(),
            'hash_dist': hash_dists,
        }
        # src_indices is only present when the DataFrame was loaded from CSV (save=True path)
        if 'src_indices' in needle_df.columns:
            row_data['needle_src']   = n_cols['src_indices'].to_numpy()
            row_data['haystack_src'] = h_cols['src_indices'].to_numpy()
        rows = pd.DataFrame(row_data)

    t_total = time.perf_counter() - t_start

    n_needle_matched = sum(1 for h in hit_indices if len(h) > 0)
    n_candidates     = len(rows) if rows is not None else 0

    # ── Print results ─────────────────────────────────────────────────────────
    if constants_astrometry.VERBOSE:
        print(f"\n{'='*72}")
        print(f"  Triangle matching — pair {PAIR_NUM}")
        print(f"{'='*72}")

        print(f"\n  Summary")
        print(f"  {'-'*68}")
        print(f"  {'Needle triangles queried':<36}  {len(needle_df)}")
        print(f"  {'Needle triangles with ≥1 hit':<36}  {n_needle_matched}")
        print(f"  {'Total candidate pairs':<36}  {n_candidates}")
        print(f"  {'k-d tree query time':<36}  {t_query*1e3:.2f} ms")
        print(f"  {'Total time':<36}  {t_total*1e3:.2f} ms")

        if rows is not None:
            print(f"\n  Hash distance statistics")
            print(f"  {'-'*68}")
            d = rows['hash_dist']
            print(f"  {'mean':<16} {d.mean():>10.6f}")
            print(f"  {'std':<16} {d.std():>10.6f}")
            print(f"  {'min':<16} {d.min():>10.6f}")
            print(f"  {'max':<16} {d.max():>10.6f}")

            n_show = min(10, n_candidates)
            print(f"\n  First {n_show} candidate pairs")
            print(f"  {'-'*68}")
            print(f"  {'needle_t':>8}  {'hay_t':>6}  {'hash_dist':>10}")
            print(f"  {'-'*30}")
            for _, row in rows.head(n_show).iterrows():
                print(f"  {int(row['needle_tri_idx']):>8}  {int(row['haystack_tri_idx']):>6}  "
                      f"{row['hash_dist']:>10.6f}")

        print(f"\n{'='*72}\n")
    else:
        print(f"  [match] {n_needle_matched}/{len(needle_df)} needle triangles matched → "
              f"{n_candidates} candidates  ({t_query*1e3:.0f} ms)")

    # ── Save ──────────────────────────────────────────────────────────────────
    if rows is not None:
        if save and needle_triangles_path is not None:
            pair_dir = os.path.dirname(needle_triangles_path)
            pair_num = os.path.basename(pair_dir).split("_")[1]
            csv_path = os.path.join(pair_dir, f"candidates_{pair_num}.csv")
            rows.to_csv(csv_path, index=False)
            print(f"Saved {n_candidates} candidates → {csv_path}")
        return rows

    return pd.DataFrame()


if __name__ == "__main__":
    match_triangles(
        needle_triangles_path=os.path.join(base, f"triangles_needle_{PAIR_NUM}.csv"),
        haystack_tree_path   =os.path.join(base, f"kdtree_haystack_{PAIR_NUM}.pkl"),
    )
