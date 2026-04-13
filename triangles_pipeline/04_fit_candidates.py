import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import numpy as np
import pandas as pd
import constants_astrometry
from utils import fit_similarity_batch, load_ground_truth

PAIR_NUM = "0001"

base = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data_generation", "dataset", f"pair_{PAIR_NUM}")


def fit_candidates(candidates_path=None, needle_fits_path=None, haystack_fits_path=None,
                   candidates_df=None, save=True):
    """Vectorized similarity fit for all candidate triangle pairs.

    Reconstructs the pixel coordinates of point C from the stored hash
    coordinates, then fits a similarity transform to each (needle, haystack)
    triangle pair in one batched call.

    Returns (cdf, gt) where cdf has scale/angle_deg/tx/ty/residual and
    err_* columns added, and gt is the ground-truth dict from the FITS header.
    """
    t_start = time.perf_counter()

    if candidates_df is not None:
        cdf = candidates_df.copy()
        if constants_astrometry.VERBOSE:
            print(f"Using {len(cdf)} candidate pairs (in-memory)")
    else:
        cdf = pd.read_csv(candidates_path)
        if constants_astrometry.VERBOSE:
            print(f"Loaded {len(cdf)} candidate pairs from {candidates_path}")

    gt = load_ground_truth(needle_fits_path, haystack_fits_path)

    # ── Vectorized similarity fit for all candidate pairs ─────────────────────
    n_A = cdf[['n_A_px_x', 'n_A_px_y']].to_numpy()   # (N, 2)
    n_B = cdf[['n_B_px_x', 'n_B_px_y']].to_numpy()
    h_A = cdf[['h_A_px_x', 'h_A_px_y']].to_numpy()
    h_B = cdf[['h_B_px_x', 'h_B_px_y']].to_numpy()

    n_AB   = n_B - n_A                                 # (N, 2)
    h_AB   = h_B - h_A
    n_perp = np.stack([-n_AB[:, 1], n_AB[:, 0]], axis=1)
    h_perp = np.stack([-h_AB[:, 1], h_AB[:, 0]], axis=1)

    n_Cx = cdf['n_Cx'].to_numpy()[:, None]
    n_Cy = cdf['n_Cy'].to_numpy()[:, None]
    h_Cx = cdf['h_Cx'].to_numpy()[:, None]
    h_Cy = cdf['h_Cy'].to_numpy()[:, None]

    # Reconstruct pixel coords of C from the (Cx, Cy) hash coordinates.
    # In the triangle hash, A→(0,0) and B→(1,1) via a 45° rotation of the AB
    # frame, so the stored coords are Cx = pu−pv, Cy = pu+pv (where pu, pv are
    # the normalised along-AB and perpendicular components).  Inverting:
    #   pu = (Cx + Cy) / 2,   pv = (Cy − Cx) / 2
    # C_pixel = A + pu*(B−A) + pv*perp(B−A)
    n_C = n_A + ((n_Cx + n_Cy) / 2) * n_AB + ((n_Cy - n_Cx) / 2) * n_perp   # (N, 2)
    h_C = h_A + ((h_Cx + h_Cy) / 2) * h_AB + ((h_Cy - h_Cx) / 2) * h_perp

    needle_pts   = np.stack([n_A, n_B, n_C], axis=1)   # (N, 3, 2)
    haystack_pts = np.stack([h_A, h_B, h_C], axis=1)

    scales, angles, txs, tys, residuals = fit_similarity_batch(needle_pts, haystack_pts)

    cdf['scale']     = scales
    cdf['angle_deg'] = angles
    cdf['tx']        = txs
    cdf['ty']        = tys
    cdf['residual']  = residuals

    # Error relative to ground truth
    cdf['err_scale'] = np.abs(cdf['scale']     - gt['true_scale'])
    cdf['err_angle'] = np.abs(cdf['angle_deg'] - gt['true_angle_deg'])
    cdf['err_tx']    = np.abs(cdf['tx']        - gt['true_tx'])
    cdf['err_ty']    = np.abs(cdf['ty']        - gt['true_ty'])

    t_total = time.perf_counter() - t_start

    # ── Print results ─────────────────────────────────────────────────────────
    if constants_astrometry.VERBOSE:
        print(f"\n{'='*72}")
        print(f"  Candidate fits — pair {PAIR_NUM}")
        print(f"{'='*72}")

        print(f"\n  Ground truth  (from needle header)")
        print(f"  {'-'*68}")
        print(f"  {'NANGLE applied':<36}  {gt['nangle']:>+10.4f} °")
        print(f"  {'Expected scale':<36}  {gt['true_scale']:>+10.4f}")
        print(f"  {'Expected angle (= NANGLE)':<36}  {gt['true_angle_deg']:>+10.4f} °")
        print(f"  {'Expected tx':<36}  {gt['true_tx']:>+10.2f} px")
        print(f"  {'Expected ty':<36}  {gt['true_ty']:>+10.2f} px")
        print(f"  {'WCS error X':<36}  {gt['werr_x_px']:>+10.4f} px")
        print(f"  {'WCS error Y':<36}  {gt['werr_y_px']:>+10.4f} px")

        print(f"\n  All candidates — recovered transform statistics")
        print(f"  {'-'*68}")
        print(f"  {'param':<14} {'mean':>10}  {'std':>10}  {'min':>10}  {'max':>10}")
        print(f"  {'-'*58}")
        for col in ['scale', 'angle_deg', 'tx', 'ty', 'residual']:
            v = cdf[col]
            print(f"  {col:<14} {v.mean():>+10.4f}  {v.std():>10.4f}  "
                  f"{v.min():>+10.4f}  {v.max():>+10.4f}")

        print(f"\n  All candidates — error vs ground truth")
        print(f"  {'-'*68}")
        print(f"  {'param':<14} {'mean err':>10}  {'std err':>10}  {'min err':>10}  {'max err':>10}")
        print(f"  {'-'*58}")
        for col in ['err_scale', 'err_angle', 'err_tx', 'err_ty']:
            v = cdf[col]
            print(f"  {col:<14} {v.mean():>10.4f}  {v.std():>10.4f}  "
                  f"{v.min():>10.4f}  {v.max():>10.4f}")

        best = cdf.loc[cdf['residual'].idxmin()]
        print(f"\n  Best candidate  (lowest residual)")
        print(f"  {'-'*68}")
        print(f"  {'':30}  {'recovered':>10}  {'expected':>10}  {'error':>10}")
        print(f"  {'-'*68}")
        for label, rec, exp, err in [
            ('scale',     best['scale'],     gt['true_scale'],     best['err_scale']),
            ('angle (°)', best['angle_deg'], gt['true_angle_deg'], best['err_angle']),
            ('tx (px)',   best['tx'],        gt['true_tx'],        best['err_tx']),
            ('ty (px)',   best['ty'],        gt['true_ty'],        best['err_ty']),
            ('residual',  best['residual'],  0.0,                  best['residual']),
        ]:
            print(f"  {label:<30}  {rec:>+10.4f}  {exp:>+10.4f}  {err:>10.4f}")

        print(f"\n  {'Candidates fitted':<36}  {len(cdf)}")
        print(f"  {'Total time':<36}  {t_total*1e3:.2f} ms")
        print(f"\n{'='*72}\n")

    # ── Save ──────────────────────────────────────────────────────────────────
    if save and candidates_path is not None:
        pair_dir = os.path.dirname(candidates_path)
        pair_num = os.path.basename(pair_dir).split("_")[1]
        csv_path = os.path.join(pair_dir, f"transforms_{pair_num}.csv")
        cdf.to_csv(csv_path, index=False)
        print(f"Saved {len(cdf)} fitted transforms → {csv_path}")

    return cdf, gt


if __name__ == "__main__":
    fit_candidates(
        candidates_path    =os.path.join(base, f"candidates_{PAIR_NUM}.csv"),
        needle_fits_path   =os.path.join(base, f"needle_{PAIR_NUM}.fits"),
        haystack_fits_path =os.path.join(base, f"haystack_{PAIR_NUM}.fits"),
    )
