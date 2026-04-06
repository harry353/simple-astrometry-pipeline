import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import pickle
import numpy as np
import pandas as pd

PAIR_NUM  = "0001"
TOLERANCE = 0.02   # radius in 4D hash space (Cx, Cy, Dx, Dy) to accept a match

base = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data_generation", "dataset", f"pair_{PAIR_NUM}")


def match_quads(needle_quads_path, haystack_tree_path):
    t_start = time.perf_counter()

    # ── Load inputs ───────────────────────────────────────────────────────────
    needle_df = pd.read_csv(needle_quads_path)
    with open(haystack_tree_path, "rb") as f:
        payload = pickle.load(f)
    tree        = payload['tree']
    haystack_df = payload['df']

    print(f"Needle quads  : {len(needle_df)}")
    print(f"Haystack quads: {len(haystack_df)}")
    print(f"Tolerance     : {TOLERANCE}  (4D hash-space radius)")

    # ── Query k-d tree ────────────────────────────────────────────────────────
    needle_hashes = needle_df[['Cx', 'Cy', 'Dx', 'Dy']].to_numpy()

    t_query = time.perf_counter()
    hit_indices = tree.query_ball_point(needle_hashes, r=TOLERANCE)
    t_query = time.perf_counter() - t_query

    # ── Collect candidate pairs ───────────────────────────────────────────────
    # Each row carries full pixel-coord info from both quads so the next
    # script can fit the similarity transform without reloading the CSVs.
    rows = []
    for ni, haystack_hits in enumerate(hit_indices):
        for hi in haystack_hits:
            n = needle_df.iloc[ni]
            h = haystack_df.iloc[hi]
            rows.append({
                'needle_quad_idx':   ni,
                'haystack_quad_idx': hi,
                'needle_src':        n['src_indices'],
                'haystack_src':      h['src_indices'],
                # needle pixel coords
                'n_A_px_x': n['A_px_x'], 'n_A_px_y': n['A_px_y'],
                'n_B_px_x': n['B_px_x'], 'n_B_px_y': n['B_px_y'],
                'n_Cx': n['Cx'], 'n_Cy': n['Cy'],
                'n_Dx': n['Dx'], 'n_Dy': n['Dy'],
                # haystack pixel coords
                'h_A_px_x': h['A_px_x'], 'h_A_px_y': h['A_px_y'],
                'h_B_px_x': h['B_px_x'], 'h_B_px_y': h['B_px_y'],
                'h_Cx': h['Cx'], 'h_Cy': h['Cy'],
                'h_Dx': h['Dx'], 'h_Dy': h['Dy'],
                # hash distance
                'hash_dist': float(np.linalg.norm(
                    needle_hashes[ni] - haystack_df.iloc[hi][['Cx','Cy','Dx','Dy']].to_numpy()
                )),
            })

    t_total = time.perf_counter() - t_start

    n_needle_matched = sum(1 for h in hit_indices if len(h) > 0)
    n_candidates     = len(rows)

    # ── Print results ─────────────────────────────────────────────────────────
    print(f"\n{'='*72}")
    print(f"  Quad matching — pair {PAIR_NUM}")
    print(f"{'='*72}")

    print(f"\n  Summary")
    print(f"  {'-'*68}")
    print(f"  {'Needle quads queried':<36}  {len(needle_df)}")
    print(f"  {'Needle quads with ≥1 hit':<36}  {n_needle_matched}")
    print(f"  {'Total candidate pairs':<36}  {n_candidates}")
    print(f"  {'k-d tree query time':<36}  {t_query*1e3:.2f} ms")
    print(f"  {'Total time':<36}  {t_total*1e3:.2f} ms")

    if rows:
        cdf = pd.DataFrame(rows)

        print(f"\n  Hash distance statistics")
        print(f"  {'-'*68}")
        d = cdf['hash_dist']
        print(f"  {'mean':<16} {d.mean():>10.6f}")
        print(f"  {'std':<16} {d.std():>10.6f}")
        print(f"  {'min':<16} {d.min():>10.6f}")
        print(f"  {'max':<16} {d.max():>10.6f}")

        n_show = min(10, len(rows))
        print(f"\n  First {n_show} candidate pairs")
        print(f"  {'-'*68}")
        print(f"  {'needle_q':>8}  {'hay_q':>6}  {'hash_dist':>10}")
        print(f"  {'-'*30}")
        for _, row in cdf.head(n_show).iterrows():
            print(f"  {int(row['needle_quad_idx']):>8}  {int(row['haystack_quad_idx']):>6}  "
                  f"{row['hash_dist']:>10.6f}")

    print(f"\n{'='*72}\n")

    # ── Save ──────────────────────────────────────────────────────────────────
    if rows:
        pair_dir = os.path.dirname(needle_quads_path)
        pair_num = os.path.basename(pair_dir).split("_")[1]
        csv_path = os.path.join(pair_dir, f"candidates_{pair_num}.csv")
        cdf.to_csv(csv_path, index=False)
        print(f"Saved {n_candidates} candidates → {csv_path}")
        return cdf

    return pd.DataFrame()


if __name__ == "__main__":
    match_quads(
        needle_quads_path =os.path.join(base, f"quads_needle_{PAIR_NUM}.csv"),
        haystack_tree_path=os.path.join(base, f"kdtree_haystack_{PAIR_NUM}.pkl"),
    )
