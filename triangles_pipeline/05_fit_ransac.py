import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import numpy as np
import pandas as pd
from astropy.io import fits
from scipy.spatial import KDTree
import importlib
import constants_astrometry
from utils import detect_centroids, fit_similarity, load_ground_truth

fit_candidates_mod = importlib.import_module("04a_fit_candidates")

PAIR_NUM = "0001"

base = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data_generation", "dataset", f"pair_{PAIR_NUM}")


def ransac_best_transform(needle_centroids, haystack_centroids, cdf):
    """
    Two-stage inlier approach:
      1. Score every candidate transform using the wide INLIER_SEARCH_RADIUS to
         pick the hypothesis with the most inliers (tolerant of a rough seed).
      2. Iteratively refit using the tight INLIER_RADIUS until the inlier set
         stabilises or INLIER_MAX_ITERS is reached (removes noisy pairs to
         improve angle accuracy).

    Returns (refit_dict | None, n_inliers, best_iloc).
    """
    search_radius = constants_astrometry.INLIER_SEARCH_RADIUS
    fit_radius    = constants_astrometry.INLIER_RADIUS
    tree          = KDTree(haystack_centroids)

    scales     = cdf['scale'].to_numpy().copy()
    angles_rad = np.radians(cdf['angle_deg'].to_numpy())
    txs        = cdf['tx'].to_numpy()
    tys        = cdf['ty'].to_numpy()

    if constants_astrometry.FORCE_SCALE_ONE:
        scales = np.ones_like(scales)

    a = scales * np.cos(angles_rad)   # (N,)
    b = scales * np.sin(angles_rad)   # (N,)

    # ── Stage 1: score all hypotheses with the wide search radius ─────────────
    R = np.stack([np.stack([a, -b], axis=1),
                  np.stack([b,  a], axis=1)], axis=1)          # (N, 2, 2)
    pred  = np.einsum('nij,mj->nmi', R, needle_centroids)      # (N, M, 2)
    pred += np.stack([txs, tys], axis=1)[:, None, :]

    N, M          = pred.shape[:2]
    dists         = tree.query(pred.reshape(-1, 2), k=1)[0].reshape(N, M)
    inlier_counts = (dists < search_radius).sum(axis=1)
    best_iloc     = int(inlier_counts.argmax())

    # Seed transform from the winning hypothesis
    cur_a  = float(a[best_iloc])
    cur_b  = float(b[best_iloc])
    cur_tx = float(txs[best_iloc])
    cur_ty = float(tys[best_iloc])

    # ── Stage 2: iterative refit with the tight fit radius ────────────────────
    prev_inlier_set = None
    refit           = None
    n_iters         = 0

    for _ in range(constants_astrometry.INLIER_MAX_ITERS):
        R_cur    = np.array([[cur_a, -cur_b], [cur_b, cur_a]])
        pred_cur = (R_cur @ needle_centroids.T).T + np.array([cur_tx, cur_ty])
        dists_cur, idxs_cur = tree.query(pred_cur, k=1)
        mask      = dists_cur < fit_radius
        n_inliers = int(mask.sum())

        if n_inliers < 4:
            break

        inlier_set = frozenset(np.where(mask)[0])
        if inlier_set == prev_inlier_set:
            break
        prev_inlier_set = inlier_set

        weights = 1.0 / (dists_cur[mask] + 1e-6)
        scale, angle, tx_new, ty_new, res = fit_similarity(
            needle_centroids[mask],
            haystack_centroids[idxs_cur[mask]],
            weights=weights,
        )
        if constants_astrometry.FORCE_SCALE_ONE:
            scale = 1.0

        n_iters += 1
        refit = {
            'scale': scale, 'angle_deg': angle,
            'tx': tx_new, 'ty': ty_new,
            'residual': res, 'n_inliers': n_inliers,
            'n_iters': n_iters,
        }

        angle_rad_new = np.radians(angle)
        cur_a  = scale * np.cos(angle_rad_new)
        cur_b  = scale * np.sin(angle_rad_new)
        cur_tx = tx_new
        cur_ty = ty_new

    if refit is None:
        return None, 0, best_iloc

    return refit, refit['n_inliers'], best_iloc


def fit_ransac(candidates_path=None, needle_fits_path=None, haystack_fits_path=None,
               candidates_df=None, needle_centroids=None, haystack_centroids=None,
               gt=None, save=True):
    """RANSAC hypothesis selection followed by iterative inlier refit.

    Expects candidates_df to already have scale/angle_deg/tx/ty columns from
    fit_candidates.  Detects centroids from the FITS images if not supplied.

    Returns (cdf, gt, refit) where refit is the final inlier-fit dict or None.
    """
    t_start = time.perf_counter()

    if candidates_df is not None:
        cdf = candidates_df.copy()
    else:
        cdf = pd.read_csv(candidates_path)
        if constants_astrometry.VERBOSE:
            print(f"Loaded {len(cdf)} fitted candidates from {candidates_path}")

    if gt is None:
        gt = load_ground_truth(needle_fits_path, haystack_fits_path)

    if needle_centroids is None or haystack_centroids is None:
        with fits.open(needle_fits_path) as f:
            needle_img = f[0].data.astype(np.float64)
        with fits.open(haystack_fits_path) as f:
            haystack_img = f[0].data.astype(np.float64)
        needle_centroids, _, _   = detect_centroids(
            needle_img,
            sigma=constants_astrometry.REFIT_DETECTION_SIGMA,
            npixels=constants_astrometry.REFIT_DETECTION_NPIXELS,
        )
        haystack_centroids, _, _ = detect_centroids(
            haystack_img,
            sigma=constants_astrometry.REFIT_DETECTION_SIGMA,
            npixels=constants_astrometry.REFIT_DETECTION_NPIXELS,
        )

    refit, n_inliers, best_iloc = ransac_best_transform(
        needle_centroids, haystack_centroids, cdf
    )

    cdf['in_consensus'] = False
    cdf.iloc[best_iloc, cdf.columns.get_loc('in_consensus')] = True

    t_total = time.perf_counter() - t_start

    best_ransac = cdf.iloc[best_iloc]

    # ── Print results ─────────────────────────────────────────────────────────
    if constants_astrometry.VERBOSE:
        print(f"\n{'='*72}")
        print(f"  RANSAC + inlier refit — pair {PAIR_NUM}")
        print(f"{'='*72}")

        print(f"\n  RANSAC best hypothesis  ({n_inliers} inliers, iloc={best_iloc})")
        print(f"  {'-'*68}")
        print(f"  {'':30}  {'recovered':>10}  {'expected':>10}  {'error':>10}")
        print(f"  {'-'*68}")
        for label, rec, exp in [
            ('scale',     best_ransac['scale'],     gt['true_scale']),
            ('angle (°)', best_ransac['angle_deg'], gt['true_angle_deg']),
            ('tx (px)',   best_ransac['tx'],         gt['true_tx']),
            ('ty (px)',   best_ransac['ty'],         gt['true_ty']),
        ]:
            err = abs(rec - exp)
            print(f"  {label:<30}  {rec:>+10.4f}  {exp:>+10.4f}  {err:>10.4f}")

        print(f"\n  Inlier re-fit  (radius={constants_astrometry.INLIER_RADIUS:.0f} px, {n_inliers} inliers)")
        print(f"  {'-'*68}")
        if refit is not None:
            print(f"  {'':30}  {'recovered':>10}  {'expected':>10}  {'error':>10}")
            print(f"  {'-'*68}")
            for label, rec, exp in [
                ('scale',     refit['scale'],     gt['true_scale']),
                ('angle (°)', refit['angle_deg'], gt['true_angle_deg']),
                ('tx (px)',   refit['tx'],        gt['true_tx']),
                ('ty (px)',   refit['ty'],        gt['true_ty']),
            ]:
                err = abs(rec - exp)
                print(f"  {label:<30}  {rec:>+10.4f}  {exp:>+10.4f}  {err:>10.4f}")
            print(f"  {'residual':<30}  {refit['residual']:>+10.4f}")
        else:
            print(f"  Insufficient inliers ({n_inliers} < 4) — re-fit skipped.")

        print(f"\n  {'Needle centroids':<36}  {len(needle_centroids)}")
        print(f"  {'Haystack centroids':<36}  {len(haystack_centroids)}")
        print(f"  {'Total time':<36}  {t_total*1e3:.2f} ms")
        print(f"\n{'='*72}\n")
    else:
        if refit is not None:
            print(f"  [refit] scale={refit['scale']:+.4f} (err {abs(refit['scale']-gt['true_scale']):.4f})  "
                  f"angle={refit['angle_deg']:+.4f}° (err {abs(refit['angle_deg']-gt['true_angle_deg']):.4f})  "
                  f"tx={refit['tx']:+.2f} (err {abs(refit['tx']-gt['true_tx']):.2f})  "
                  f"ty={refit['ty']:+.2f} (err {abs(refit['ty']-gt['true_ty']):.2f})  "
                  f"inliers={n_inliers}")
        else:
            print(f"  [refit] insufficient inliers ({n_inliers} < 4) — skipped")

    # ── Save ──────────────────────────────────────────────────────────────────
    if save and candidates_path is not None:
        pair_dir = os.path.dirname(candidates_path)
        pair_num = os.path.basename(pair_dir).split("_")[1]
        csv_path = os.path.join(pair_dir, f"transforms_{pair_num}.csv")
        cdf.to_csv(csv_path, index=False)
        print(f"Saved {len(cdf)} transforms with consensus column → {csv_path}")

    return cdf, gt, refit


if __name__ == "__main__":
    cdf, gt = fit_candidates_mod.fit_candidates(
        candidates_path    =os.path.join(base, f"candidates_{PAIR_NUM}.csv"),
        needle_fits_path   =os.path.join(base, f"needle_{PAIR_NUM}.fits"),
        haystack_fits_path =os.path.join(base, f"haystack_{PAIR_NUM}.fits"),
        save=False,
    )
    fit_ransac(
        candidates_df      =cdf,
        candidates_path    =os.path.join(base, f"candidates_{PAIR_NUM}.csv"),
        needle_fits_path   =os.path.join(base, f"needle_{PAIR_NUM}.fits"),
        haystack_fits_path =os.path.join(base, f"haystack_{PAIR_NUM}.fits"),
        gt=gt,
    )
