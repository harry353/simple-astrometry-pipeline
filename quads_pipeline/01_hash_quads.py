import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import numpy as np
from itertools import combinations
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from photutils.segmentation import detect_sources, SourceCatalog

PAIR_NUM = "0001"
SIGMA    = 5    # detection threshold in units of background sigma
NPIXELS  = 40   # minimum connected pixels to count as a source

base = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data_generation", "dataset", f"pair_{PAIR_NUM}")


def detect_centroids(image):
    # Estimate the background noise level
    _, median, std = sigma_clipped_stats(image, sigma=3.0)

    # Anything above median + SIGMA*std is considered a source.
    threshold = median + SIGMA * std

    # Label connected regions of pixels above the threshold into a segmentation map.
    segmap = detect_sources(image, threshold, npixels=NPIXELS)
    if segmap is None:
        return np.empty((0, 2)), median

    # Compute flux-weighted centroids, subtracting median to remove background bias.
    catalog = SourceCatalog(image - median, segmap)
    centroids = np.column_stack([np.array(catalog.xcentroid),
                                 np.array(catalog.ycentroid)])
    return centroids, median


def quad_coordinate_system(sources):
    """
    Given 4 source positions (shape 4x2), finds the pair furthest apart
    (A and B) and builds an orthonormal coordinate system rooted at A with
    the AB vector as the x-axis, scaled so that B lies at (1, 0).

    Returns:
        A, B         — pixel coords of the two anchor sources
        C_q, D_q     — (x, y) coords of the other two sources in quad space
        baseline_px  — pixel distance |AB|
        idx_a, idx_b — indices into `sources` of the anchor pair
        idx_c, idx_d — indices into `sources` of the inner pair
    """
    # Find the pair of sources with the greatest pixel separation
    max_dist = -1.0
    idx_a, idx_b = 0, 1
    for i, j in combinations(range(4), 2):
        d = np.linalg.norm(sources[i] - sources[j])
        if d > max_dist:
            max_dist = d
            idx_a, idx_b = i, j

    A = sources[idx_a]
    B = sources[idx_b]

    # u is the unit vector from A to B — this becomes the x-axis of quad space.
    # v is u rotated 90° counterclockwise — this becomes the y-axis of quad space.
    # Together they form a right-handed orthonormal basis rooted at A, scaled so
    # that B maps exactly to (1, 0). The two anchor sources therefore always map
    # to (0, 0) and (1, 0), and the remaining pair's coordinates are the hash.
    baseline = max_dist
    u = (B - A) / baseline
    v = np.array([-u[1], u[0]])

    inner_indices = [i for i in range(4) if i not in (idx_a, idx_b)]
    idx_c, idx_d  = inner_indices

    def to_quad(P):
        rel = P - A
        return np.array([np.dot(rel, u) / baseline,
                         np.dot(rel, v) / baseline])

    C_q = to_quad(sources[idx_c])
    D_q = to_quad(sources[idx_d])

    return A, B, C_q, D_q, baseline, idx_a, idx_b, idx_c, idx_d


def main(haystack_path):
    t_start = time.perf_counter()

    with fits.open(haystack_path) as hdul:
        image  = hdul[0].data.astype(np.float64)
        header = hdul[0].header

    print(f"Loaded haystack: shape={image.shape}  path={haystack_path}")

    t_detect = time.perf_counter()
    centroids, _ = detect_centroids(image)
    t_detect = time.perf_counter() - t_detect

    n_sources = len(centroids)
    n_quads   = 0

    print(f"Detected {n_sources} sources  ({t_detect*1e3:.1f} ms)")

    if n_sources < 4:
        print("Fewer than 4 sources detected — cannot form any quads. Exiting.")
        return []

    # C(n, 4) quads total
    n_expected = len(list(combinations(range(n_sources), 4)))
    print(f"Building quads: C({n_sources}, 4) = {n_expected} quads...")

    t_hash = time.perf_counter()

    quads = []
    for combo in combinations(range(n_sources), 4):
        sources = centroids[list(combo)]
        A, B, C_q, D_q, baseline, ia, ib, ic, id_ = quad_coordinate_system(sources)
        quads.append({
            'src_indices': combo,                # global source indices (i, j, k, l)
            'idx_a':       combo[ia],            # anchor source A
            'idx_b':       combo[ib],            # anchor source B
            'idx_c':       combo[ic],            # inner source C
            'idx_d':       combo[id_],           # inner source D
            'A_px':        A,                    # pixel coords of A
            'B_px':        B,                    # pixel coords of B
            'baseline_px': baseline,             # pixel distance |AB|
            'C_q':         C_q,                  # C in quad space: (x, y)
            'D_q':         D_q,                  # D in quad space: (x, y)
        })
        n_quads += 1

    t_hash = time.perf_counter() - t_hash

    t_total = time.perf_counter() - t_start

    # ── Print results ──────────────────────────────────────────────────────────
    print(f"\n{'='*72}")
    print(f"  Quad hashing — pair {PAIR_NUM}")
    print(f"{'='*72}")

    print(f"\n  Coordinate system")
    print(f"  {'-'*68}")
    print(f"  {'Origin (0, 0)':<32}  anchor source A  (furthest-apart pair)")
    print(f"  {'Unit point (1, 0)':<32}  anchor source B  (furthest-apart pair)")
    print(f"  {'x-axis':<32}  unit vector A → B")
    print(f"  {'y-axis':<32}  x-axis rotated 90° CCW")
    print(f"  {'Scale':<32}  1 unit = |AB| pixels")
    print(f"  {'Invariances':<32}  translation, rotation, scale")
    print(f"  {'Hash':<32}  (Cx, Cy, Dx, Dy)  — inner source coords")

    print(f"\n  Summary")
    print(f"  {'-'*68}")
    print(f"  {'Sources detected':<32}  {n_sources}")
    print(f"  {'Quads built':<32}  {n_quads}")
    print(f"  {'Source detection time':<32}  {t_detect*1e3:.2f} ms")
    print(f"  {'Quad hashing time':<32}  {t_hash*1e3:.2f} ms")
    print(f"  {'Total time':<32}  {t_total*1e3:.2f} ms")

    # Collect baseline and hash stats across all quads
    baselines = np.array([q['baseline_px'] for q in quads])
    Cxs = np.array([q['C_q'][0] for q in quads])
    Cys = np.array([q['C_q'][1] for q in quads])
    Dxs = np.array([q['D_q'][0] for q in quads])
    Dys = np.array([q['D_q'][1] for q in quads])

    print(f"\n  Baseline statistics (|AB| in pixels)")
    print(f"  {'-'*68}")
    print(f"  {'mean':<16} {baselines.mean():>10.2f} px")
    print(f"  {'std':<16} {baselines.std():>10.2f} px")
    print(f"  {'min':<16} {baselines.min():>10.2f} px")
    print(f"  {'max':<16} {baselines.max():>10.2f} px")

    print(f"\n  Hash coordinate statistics  (all quads)")
    print(f"  {'-'*68}")
    print(f"  {'coord':<10} {'mean':>10}  {'std':>10}  {'min':>10}  {'max':>10}")
    print(f"  {'-'*54}")
    for label, vals in [("Cx", Cxs), ("Cy", Cys), ("Dx", Dxs), ("Dy", Dys)]:
        print(f"  {label:<10} {vals.mean():>+10.4f}  {vals.std():>10.4f}  "
              f"{vals.min():>+10.4f}  {vals.max():>+10.4f}")

    # Print the first 10 quads in detail to illustrate the hashing
    n_show = min(10, len(quads))
    print(f"\n  First {n_show} quads (detailed)")
    print(f"  {'-'*68}")
    hdr = (f"  {'Quad':<6} {'A':>5} {'B':>5} {'C':>5} {'D':>5}  "
           f"{'baseline(px)':>14}  {'Cx':>8} {'Cy':>8}  {'Dx':>8} {'Dy':>8}")
    print(hdr)
    print(f"  {'-'*68}")
    for q in quads[:n_show]:
        ia, ib, ic, id_ = q['idx_a'], q['idx_b'], q['idx_c'], q['idx_d']
        Cx, Cy = q['C_q']
        Dx, Dy = q['D_q']
        print(f"  {str(q['src_indices']):<6}  "
              f"{ia:>4} {ib:>4} {ic:>4} {id_:>4}  "
              f"{q['baseline_px']:>14.2f}  "
              f"{Cx:>+8.4f} {Cy:>+8.4f}  "
              f"{Dx:>+8.4f} {Dy:>+8.4f}")

    print(f"\n{'='*72}\n")

    return quads


if __name__ == "__main__":
    main(haystack_path=os.path.join(base, f"haystack_{PAIR_NUM}.fits"))
