import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from astropy.stats import sigma_clipped_stats
from photutils.segmentation import detect_sources, SourceCatalog


def detect_centroids(image, sigma, npixels):
    """
    Detect sources in an image and return their centroids, background median,
    and per-source fluxes.

    Args:
        image    : 2D numpy array
        sigma    : detection threshold in units of background sigma
        npixels  : minimum connected pixels to count as a source

    Returns:
        centroids : (N, 2) array of (x, y) centroid positions
        median    : background median
        fluxes    : (N,) array of segment fluxes
    """
    _, median, std = sigma_clipped_stats(image, sigma=3.0)
    threshold = median + sigma * std
    segmap = detect_sources(image, threshold, npixels=npixels)
    if segmap is None:
        return np.empty((0, 2)), median, np.empty(0)
    catalog = SourceCatalog(image - median, segmap)
    centroids = np.column_stack([np.array(catalog.xcentroid),
                                 np.array(catalog.ycentroid)])
    fluxes = np.array(catalog.segment_flux)
    return centroids, median, fluxes


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

    params, _, _, _ = np.linalg.lstsq(A, rhs, rcond=None)
    a, b, tx, ty = params

    scale     = float(np.sqrt(a**2 + b**2))
    angle_deg = float(np.degrees(np.arctan2(b, a)))
    residual  = float(np.sum((A @ params - rhs) ** 2))

    return scale, angle_deg, float(tx), float(ty), residual


def fit_similarity_batch(needle_pts, haystack_pts):
    """
    Vectorized similarity fit for N quads at once.

    needle_pts, haystack_pts: shape (N, 4, 2)
    Returns arrays of shape (N,): scales, angles_deg, txs, tys, residuals.
    """
    N = needle_pts.shape[0]
    A   = np.zeros((N, 8, 4))
    rhs = np.zeros((N, 8))
    for i in range(4):
        px = needle_pts[:, i, 0]
        py = needle_pts[:, i, 1]
        qx = haystack_pts[:, i, 0]
        qy = haystack_pts[:, i, 1]
        r0, r1 = 2 * i, 2 * i + 1
        A[:, r0, 0] =  px;  A[:, r0, 1] = -py;  A[:, r0, 2] = 1;  A[:, r0, 3] = 0
        A[:, r1, 0] =  py;  A[:, r1, 1] =  px;  A[:, r1, 2] = 0;  A[:, r1, 3] = 1
        rhs[:, r0]  =  qx
        rhs[:, r1]  =  qy

    params     = np.array([np.linalg.lstsq(A[k], rhs[k], rcond=None)[0] for k in range(N)])
    a, b, tx, ty = params[:, 0], params[:, 1], params[:, 2], params[:, 3]
    scales     = np.sqrt(a**2 + b**2)
    angles_deg = np.degrees(np.arctan2(b, a))
    pred       = np.einsum('nij,nj->ni', A.reshape(N, 8, 4), params)
    residuals  = np.sum((pred - rhs) ** 2, axis=1)
    return scales, angles_deg, tx, ty, residuals


def load_ground_truth(needle_fits_path, haystack_fits_path):
    """
    Extract ground-truth transform parameters from the needle FITS header.

    The needle was:
      1. Cut from the haystack at some pixel position (x_idx, y_idx).
      2. Rotated by NANGLE degrees CCW.
      3. Given a WCS with CRPIX shifted by (WERR_XPX, WERR_YPX).

    True needle→haystack pixel transform:
      - scale    ≈ 1.0  (same pixel scale)
      - angle    ≈ -NANGLE  (inverse of the rotation applied)
      - tx, ty   ≈ position of the needle origin in the haystack,
                   derived from CRVAL + corrected CRPIX via the haystack WCS.
    """
    with fits.open(needle_fits_path) as f:
        needle_hdr = f[0].header
    with fits.open(haystack_fits_path) as f:
        haystack_hdr = f[0].header

    nangle    = float(needle_hdr['NANGLE'])
    werr_x    = float(needle_hdr['WERR_XPX'])
    werr_y    = float(needle_hdr['WERR_YPX'])
    needle_sz = int(needle_hdr['NAXIS1'])

    crval_ra  = float(needle_hdr['CRVAL1'])
    crval_dec = float(needle_hdr['CRVAL2'])

    true_crpix_x = (needle_sz - 1) / 2.0
    true_crpix_y = (needle_sz - 1) / 2.0

    wcs_haystack   = WCS(haystack_hdr)
    cx_hay, cy_hay = wcs_haystack.all_world2pix([[crval_ra, crval_dec]], 0)[0]

    nangle_rad = np.radians(nangle)
    cos_n      = np.cos(nangle_rad)
    sin_n      = np.sin(nangle_rad)
    true_tx    = float(cx_hay - cos_n * true_crpix_x + sin_n * true_crpix_y)
    true_ty    = float(cy_hay - sin_n * true_crpix_x - cos_n * true_crpix_y)

    return {
        'true_scale':     1.0,
        'true_angle_deg': float(nangle),
        'true_tx':        true_tx,
        'true_ty':        true_ty,
        'werr_x_px':      werr_x,
        'werr_y_px':      werr_y,
        'nangle':         nangle,
    }
