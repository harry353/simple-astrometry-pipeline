import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
from collections import Counter
import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.wcs import WCS
from astropy.stats import sigma_clipped_stats
from photutils.segmentation import detect_sources, SourceCatalog
from scipy.spatial import KDTree

PAIR_NUM      = "0001"
SIGMA         = 5    # must match 01_hash_quads.py
NPIXELS       = 40   # must match 01_hash_quads.py
INLIER_RADIUS = 3.0  # px: centroid match radius for inlier re-fit
BIN_SIZE      = 3.0  # px: grid cell size for (tx, ty) consensus voting

base = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data_generation", "dataset", f"pair_{PAIR_NUM}")


def _quad_to_px(A, B, cx, cy):
    """Convert a point in quad space back to pixel coordinates."""
    AB   = B - A
    perp = np.array([-AB[1], AB[0]])
    return A + cx * AB + cy * perp


def detect_centroids(image):
    _, median, std = sigma_clipped_stats(image, sigma=3.0)
    threshold = median + SIGMA * std
    segmap = detect_sources(image, threshold, npixels=NPIXELS)
    if segmap is None:
        return np.empty((0, 2)), median
    catalog = SourceCatalog(image - median, segmap)
    centroids = np.column_stack([np.array(catalog.xcentroid),
                                 np.array(catalog.ycentroid)])
    return centroids, median


def fit_similarity(needle_pts, haystack_pts):
    """
    Fit a similarity transform (rotation, uniform scale, translation) that maps
    needle pixel coords to haystack pixel coords via least squares.

    Model:  q = s * R(θ) * p + t
    Linear form per point pair (px, py) → (qx, qy):

        [px  -py  1  0] [a ]   [qx]
        [py   px  0  1] [b ] = [qy]
                         [tx]
                         [ty]

    where  a = s·cos θ,  b = s·sin θ.
    """
    n   = len(needle_pts)
    A   = np.zeros((2 * n, 4))
    rhs = np.zeros(2 * n)

    for i, ((px, py), (qx, qy)) in enumerate(zip(needle_pts, haystack_pts)):
        A[2*i]     = [px, -py, 1, 0]
        A[2*i + 1] = [py,  px, 0, 1]
        rhs[2*i]   = qx
        rhs[2*i+1] = qy

    params, residuals, _, _ = np.linalg.lstsq(A, rhs, rcond=None)
    a, b, tx, ty = params

    scale     = float(np.sqrt(a**2 + b**2))
    angle_deg = float(np.degrees(np.arctan2(b, a)))
    residual  = float(np.sum((A @ params - rhs) ** 2))

    return scale, angle_deg, float(tx), float(ty), residual


def load_ground_truth(needle_fits_path, haystack_fits_path):
    """
    Extract ground-truth transform parameters from the needle header.

    The needle was:
      1. Cut from the haystack at some pixel position (x_idx, y_idx).
      2. Rotated by NANGLE degrees CCW.
      3. Given a WCS with CRPIX shifted by (WERR_XPX, WERR_YPX).

    True needle→haystack pixel transform:
      - scale    ≈ 1.0  (same pixel scale)
      - angle    ≈ -NANGLE  (inverse of the rotation applied)
      - tx, ty   ≈ position of the needle origin in the haystack,
                   which we derive from CRVAL + corrected CRPIX via the haystack WCS.
    """
    with fits.open(needle_fits_path) as f:
        needle_hdr = f[0].header
    with fits.open(haystack_fits_path) as f:
        haystack_hdr = f[0].header

    nangle    = float(needle_hdr['NANGLE'])
    werr_x    = float(needle_hdr['WERR_XPX'])
    werr_y    = float(needle_hdr['WERR_YPX'])
    needle_sz = int(needle_hdr['NAXIS1'])

    # The true center of the needle in the sky (CRVAL is correct — only CRPIX was shifted)
    crval_ra  = float(needle_hdr['CRVAL1'])
    crval_dec = float(needle_hdr['CRVAL2'])

    # True CRPIX (before WCS error was applied) in 0-indexed pixel coords
    true_crpix_x = (needle_sz - 1) / 2.0   # = half in create_needle.py (0-indexed)
    true_crpix_y = (needle_sz - 1) / 2.0

    # Map the needle's sky center to haystack pixel coords
    wcs_haystack = WCS(haystack_hdr)
    cx_hay, cy_hay = wcs_haystack.all_world2pix([[crval_ra, crval_dec]], 0)[0]

    # fit_similarity model: qx = a*px - b*py + tx,  qy = b*px + a*py + ty
    # scipy.ndimage.rotate(buf, NANGLE) maps needle pixel (nx, ny) to haystack via:
    #   hay_col = cos(NANGLE)*nx - sin(NANGLE)*ny + tx
    #   hay_row = sin(NANGLE)*nx + cos(NANGLE)*ny + ty
    # so  a = cos(NANGLE),  b = sin(NANGLE),  angle = arctan2(b,a) = NANGLE.
    # Centre constraint — needle centre (crpix_x, crpix_y) maps to (cx_hay, cy_hay):
    #   tx = cx_hay - a*crpix_x + b*crpix_y
    #   ty = cy_hay - b*crpix_x - a*crpix_y
    nangle_rad = np.radians(nangle)
    cos_n      = np.cos(nangle_rad)
    sin_n      = np.sin(nangle_rad)
    true_tx    = float(cx_hay - cos_n * true_crpix_x + sin_n * true_crpix_y)
    true_ty    = float(cy_hay - sin_n * true_crpix_x - cos_n * true_crpix_y)

    return {
        'true_scale':     1.0,
        'true_angle_deg': float(nangle),   # = arctan2(sin_n, cos_n) = NANGLE
        'true_tx':        true_tx,
        'true_ty':        true_ty,
        'werr_x_px':      werr_x,
        'werr_y_px':      werr_y,
        'nangle':         nangle,
    }


def find_consensus(cdf, bin_size=BIN_SIZE):
    """
    Vote on (tx, ty) using a 2D grid. Returns the subset of candidates
    within ±1 bin of the peak cell and the vote count at the peak.
    """
    tx_bins = np.round(cdf['tx'].to_numpy() / bin_size).astype(int)
    ty_bins = np.round(cdf['ty'].to_numpy() / bin_size).astype(int)
    votes = Counter(zip(tx_bins.tolist(), ty_bins.tolist()))
    best_bin, best_count = votes.most_common(1)[0]
    mask = (np.abs(tx_bins - best_bin[0]) <= 1) & (np.abs(ty_bins - best_bin[1]) <= 1)
    return cdf[mask].copy(), int(best_count)


def refit_with_inliers(needle_centroids, haystack_centroids, tx, ty, a, b):
    """
    Given an initial transform (a, b, tx, ty), map all needle centroids to
    haystack space, find matches within INLIER_RADIUS, and re-fit with all
    inlier pairs.
    """
    R    = np.array([[a, -b], [b, a]])
    pred = (R @ needle_centroids.T).T + np.array([tx, ty])
    tree = KDTree(haystack_centroids)
    dists, idxs = tree.query(pred, k=1)
    mask      = dists < INLIER_RADIUS
    n_inliers = int(mask.sum())
    if n_inliers < 4:
        return None, n_inliers
    scale, angle, tx_new, ty_new, res = fit_similarity(
        needle_centroids[mask],
        haystack_centroids[idxs[mask]]
    )
    return {
        'scale': scale, 'angle_deg': angle,
        'tx': tx_new, 'ty': ty_new,
        'residual': res, 'n_inliers': n_inliers,
    }, n_inliers


def fit_transforms(candidates_path=None, needle_fits_path=None, haystack_fits_path=None,
                   candidates_df=None, save=True):
    t_start = time.perf_counter()

    if candidates_df is not None:
        cdf = candidates_df.copy()
        print(f"Using {len(cdf)} candidate pairs (in-memory)")
    else:
        cdf = pd.read_csv(candidates_path)
        print(f"Loaded {len(cdf)} candidate pairs from {candidates_path}")

    gt = load_ground_truth(needle_fits_path, haystack_fits_path)

    # ── Fit similarity transform for every candidate pair ─────────────────────
    scales, angles, txs, tys, residuals = [], [], [], [], []

    for _, row in cdf.iterrows():
        n_A = np.array([row['n_A_px_x'], row['n_A_px_y']])
        n_B = np.array([row['n_B_px_x'], row['n_B_px_y']])
        h_A = np.array([row['h_A_px_x'], row['h_A_px_y']])
        h_B = np.array([row['h_B_px_x'], row['h_B_px_y']])

        needle_pts   = np.array([n_A, n_B,
                                 _quad_to_px(n_A, n_B, row['n_Cx'], row['n_Cy']),
                                 _quad_to_px(n_A, n_B, row['n_Dx'], row['n_Dy'])])
        haystack_pts = np.array([h_A, h_B,
                                 _quad_to_px(h_A, h_B, row['h_Cx'], row['h_Cy']),
                                 _quad_to_px(h_A, h_B, row['h_Dx'], row['h_Dy'])])

        s, a, tx, ty, res = fit_similarity(needle_pts, haystack_pts)
        scales.append(s);  angles.append(a)
        txs.append(tx);    tys.append(ty);  residuals.append(res)

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

    # ── Consensus voting on (tx, ty) ──────────────────────────────────────────
    consensus_cdf, n_votes = find_consensus(cdf)
    cons_tx    = float(consensus_cdf['tx'].median())
    cons_ty    = float(consensus_cdf['ty'].median())
    cons_angle = float(consensus_cdf['angle_deg'].median())
    cons_scale = float(consensus_cdf['scale'].median())

    cdf['in_consensus'] = False
    cdf.loc[consensus_cdf.index, 'in_consensus'] = True

    # ── Inlier re-fit ─────────────────────────────────────────────────────────
    with fits.open(needle_fits_path) as f:
        needle_img = f[0].data.astype(np.float64)
    with fits.open(haystack_fits_path) as f:
        haystack_img = f[0].data.astype(np.float64)

    needle_centroids, _   = detect_centroids(needle_img)
    haystack_centroids, _ = detect_centroids(haystack_img)

    angle_rad = np.radians(cons_angle)
    cons_a    = cons_scale * np.cos(angle_rad)
    cons_b    = cons_scale * np.sin(angle_rad)

    refit, n_inliers = refit_with_inliers(
        needle_centroids, haystack_centroids,
        cons_tx, cons_ty, cons_a, cons_b,
    )

    t_total = time.perf_counter() - t_start

    # ── Best candidate (lowest residual) ──────────────────────────────────────
    best = cdf.loc[cdf['residual'].idxmin()]

    # ── Print results ─────────────────────────────────────────────────────────
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

    print(f"\n  Consensus  ({n_votes} peak-bin votes, {len(consensus_cdf)} candidates within ±1 bin)")
    print(f"  {'-'*68}")
    print(f"  {'':30}  {'recovered':>10}  {'expected':>10}  {'error':>10}")
    print(f"  {'-'*68}")
    for label, rec, exp in [
        ('scale',     cons_scale, gt['true_scale']),
        ('angle (°)', cons_angle, gt['true_angle_deg']),
        ('tx (px)',   cons_tx,    gt['true_tx']),
        ('ty (px)',   cons_ty,    gt['true_ty']),
    ]:
        err = abs(rec - exp)
        print(f"  {label:<30}  {rec:>+10.4f}  {exp:>+10.4f}  {err:>10.4f}")

    print(f"\n  Inlier re-fit  (radius={INLIER_RADIUS:.0f} px, {n_inliers} inliers)")
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
