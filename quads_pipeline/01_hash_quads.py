import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp
from itertools import combinations
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from photutils.segmentation import detect_sources, SourceCatalog

PAIR_NUM = "0001"
SIGMA    = 5    # detection threshold in units of background sigma
NPIXELS  = 40   # minimum connected pixels to count as a source
USE_GPU  = True  # True = prefer GPU, fall back to CPU; False = always CPU

base = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data_generation", "dataset", f"pair_{PAIR_NUM}")

if USE_GPU:
    try:
        jax.devices("gpu")
        print("JAX backend: GPU")
    except RuntimeError:
        jax.config.update("jax_platform_name", "cpu")
        print("JAX backend: CPU (no GPU found)")
else:
    jax.config.update("jax_platform_name", "cpu")
    print("JAX backend: CPU (USE_GPU=False)")

# All 6 index pairs from 4 sources, and the corresponding inner pairs.
# PAIR_INDICES[k] = (i, j) means the k-th candidate anchor pair.
# INNER_PAIRS[k]  = (p, q) are the two remaining source indices when anchors are PAIR_INDICES[k].
_PAIR_INDICES = jnp.array([[0,1],[0,2],[0,3],[1,2],[1,3],[2,3]])
_INNER_PAIRS  = jnp.array([[2,3],[1,3],[1,2],[0,3],[0,2],[0,1]])


def _hash_quad(sources):
    """
    Compute the quad hash for a single group of 4 sources (shape 4x2).

    Finds the furthest-apart pair (A, B), builds an orthonormal coordinate
    system rooted at A scaled so B → (1, 0), and returns the projected
    coordinates of the two inner sources as the hash.

    Returns local indices (into the 4-element group), pixel coords, baseline,
    and the hash coordinates (Cx, Cy, Dx, Dy).
    """
    diffs    = sources[_PAIR_INDICES[:, 0]] - sources[_PAIR_INDICES[:, 1]]
    dists_sq = jnp.sum(diffs ** 2, axis=1)
    k        = jnp.argmax(dists_sq)

    ia  = _PAIR_INDICES[k, 0]
    ib  = _PAIR_INDICES[k, 1]
    ic  = _INNER_PAIRS[k, 0]
    id_ = _INNER_PAIRS[k, 1]

    A        = sources[ia]
    B        = sources[ib]
    baseline = jnp.sqrt(dists_sq[k])
    u        = (B - A) / baseline
    v        = jnp.stack([-u[1], u[0]])

    def proj(P):
        rel = P - A
        return jnp.stack([jnp.dot(rel, u) / baseline,
                          jnp.dot(rel, v) / baseline])

    return ia, ib, ic, id_, A, B, baseline, proj(sources[ic]), proj(sources[id_])


_hash_quads_batch = jax.jit(jax.vmap(_hash_quad))


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


def hash_image(fits_path, label):
    """
    Detect sources in a FITS image, hash all C(n,4) quads, print a summary,
    save results to CSV, and return the quads list.

    Args:
        fits_path : path to the FITS file
        label     : display name used in printed headings (e.g. "haystack", "needle")
    """
    t_start = time.perf_counter()

    with fits.open(fits_path) as hdul:
        image = hdul[0].data.astype(np.float64)

    print(f"Loaded {label}: shape={image.shape}  path={fits_path}")

    t_detect = time.perf_counter()
    centroids, _ = detect_centroids(image)
    t_detect = time.perf_counter() - t_detect

    n_sources = len(centroids)
    print(f"Detected {n_sources} sources  ({t_detect*1e3:.1f} ms)")

    if n_sources < 4:
        print(f"Fewer than 4 sources detected in {label} — cannot form any quads.")
        return []

    combos  = list(combinations(range(n_sources), 4))
    n_quads = len(combos)
    print(f"Building quads: C({n_sources}, 4) = {n_quads} quads...")

    combo_arr   = np.array(combos, dtype=np.int32)
    all_sources = jnp.array(centroids[combo_arr])

    t_hash = time.perf_counter()
    ia_j, ib_j, ic_j, id_j, A_j, B_j, bl_j, Cq_j, Dq_j = _hash_quads_batch(all_sources)
    bl_j.block_until_ready()
    t_hash = time.perf_counter() - t_hash

    t_total = time.perf_counter() - t_start

    ia_np = np.array(ia_j, dtype=int)
    ib_np = np.array(ib_j, dtype=int)
    ic_np = np.array(ic_j, dtype=int)
    id_np = np.array(id_j, dtype=int)
    A_np  = np.array(A_j)
    B_np  = np.array(B_j)
    bl_np = np.array(bl_j)
    Cq_np = np.array(Cq_j)
    Dq_np = np.array(Dq_j)

    quads = []
    for i, combo in enumerate(combos):
        quads.append({
            'src_indices': combo,
            'idx_a':       combo[ia_np[i]],
            'idx_b':       combo[ib_np[i]],
            'idx_c':       combo[ic_np[i]],
            'idx_d':       combo[id_np[i]],
            'A_px':        A_np[i],
            'B_px':        B_np[i],
            'baseline_px': float(bl_np[i]),
            'C_q':         Cq_np[i],
            'D_q':         Dq_np[i],
        })

    # ── Print results ──────────────────────────────────────────────────────────
    print(f"\n{'='*72}")
    print(f"  Quad hashing — {label}  (pair {PAIR_NUM})")
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

    baselines = bl_np
    Cxs, Cys  = Cq_np[:, 0], Cq_np[:, 1]
    Dxs, Dys  = Dq_np[:, 0], Dq_np[:, 1]

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
    for lbl, vals in [("Cx", Cxs), ("Cy", Cys), ("Dx", Dxs), ("Dy", Dys)]:
        print(f"  {lbl:<10} {vals.mean():>+10.4f}  {vals.std():>10.4f}  "
              f"{vals.min():>+10.4f}  {vals.max():>+10.4f}")

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

    # ── Save hashes to CSV ─────────────────────────────────────────────────────
    pair_dir = os.path.dirname(fits_path)
    pair_num = os.path.basename(pair_dir).split("_")[1]
    csv_path = os.path.join(pair_dir, f"quads_{label}_{pair_num}.csv")
    df = pd.DataFrame({
        'src_indices': [str(q['src_indices']) for q in quads],
        'idx_a':       [q['idx_a']       for q in quads],
        'idx_b':       [q['idx_b']       for q in quads],
        'idx_c':       [q['idx_c']       for q in quads],
        'idx_d':       [q['idx_d']       for q in quads],
        'A_px_x':      A_np[:, 0],
        'A_px_y':      A_np[:, 1],
        'B_px_x':      B_np[:, 0],
        'B_px_y':      B_np[:, 1],
        'baseline_px': bl_np,
        'Cx':          Cxs,
        'Cy':          Cys,
        'Dx':          Dxs,
        'Dy':          Dys,
    })
    df.to_csv(csv_path, index=False)
    print(f"Saved {n_quads} quads → {csv_path}")

    return quads


def main(haystack_path, needle_path):
    haystack_quads = hash_image(haystack_path, label="haystack")
    needle_quads   = hash_image(needle_path,   label="needle")
    return haystack_quads, needle_quads


if __name__ == "__main__":
    main(
        haystack_path=os.path.join(base, f"haystack_{PAIR_NUM}.fits"),
        needle_path  =os.path.join(base, f"needle_{PAIR_NUM}.fits"),
    )
