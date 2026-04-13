import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import numpy as np
import pandas as pd
from astropy.io import fits
from scipy.spatial import KDTree
import constants_astrometry
from utils import detect_centroids, fit_similarity, fit_similarity_batch, load_ground_truth

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


def fit_transforms(candidates_path=None, needle_fits_path=None, haystack_fits_path=None,
                   candidates_df=None, needle_centroids=None, haystack_centroids=None,
                   save=True):
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

    # ── RANSAC: pick best hypothesis by inlier count, then refit ─────────────
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

    # ── Best candidate (lowest residual) ──────────────────────────────────────
    best = cdf.loc[cdf['residual'].idxmin()]
    best_ransac = cdf.iloc[best_iloc]

    # ── Print results ─────────────────────────────────────────────────────────
    if constants_astrometry.VERBOSE:
        print(f"\n{'='*72}")
        print(f"  Similarity transform fits — pair {PAIR_NUM}")
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

        print(f"\n  {'Candidates fitted':<36}  {len(cdf)}")
        print(f"  {'Needle centroids':<36}  {len(needle_centroids)}")
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
        print(f"Saved {len(cdf)} fitted transforms → {csv_path}")

    return cdf, gt, refit


if __name__ == "__main__":
    fit_transforms(
        candidates_path    =os.path.join(base, f"candidates_{PAIR_NUM}.csv"),
        needle_fits_path   =os.path.join(base, f"needle_{PAIR_NUM}.fits"),
        haystack_fits_path =os.path.join(base, f"haystack_{PAIR_NUM}.fits"),
    )
