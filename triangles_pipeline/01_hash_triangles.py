import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
from math import comb
import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp
from itertools import combinations
from astropy.io import fits
import constants_astrometry
from utils import detect_centroids

PAIR_NUM = "0003"

base = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data_generation", "dataset", f"pair_{PAIR_NUM}")

if constants_astrometry.USE_GPU:
    try:
        jax.devices("gpu")
        print("JAX backend: GPU")
    except RuntimeError:
        jax.config.update("jax_platform_name", "cpu")
        print("JAX backend: CPU (no GPU found)")
else:
    jax.config.update("jax_platform_name", "cpu")
    print("JAX backend: CPU (USE_GPU=False)")

# All 3 index pairs from 3 sources, and the corresponding inner (remaining) source.
# PAIR_INDICES[k] = (i, j) means the k-th candidate anchor pair.
# INNER[k]        = the remaining source index when anchors are PAIR_INDICES[k].
_PAIR_INDICES = jnp.array([[0, 1], [0, 2], [1, 2]])
_INNER        = jnp.array([2, 1, 0])


def _hash_triangle(sources):
    """
    Compute the triangle hash for a single group of 3 sources (shape 3x2).

    Finds the furthest-apart pair (A, B), then applies a 45° rotation of
    the AB frame so that A → (0, 0) and B → (1, 1).  The projected
    coordinates of the remaining source C form the 2D hash.

    Returns local indices (into the 3-element group), pixel coords, baseline,
    and the hash coordinates (Cx, Cy).
    """
    diffs    = sources[_PAIR_INDICES[:, 0]] - sources[_PAIR_INDICES[:, 1]]
    dists_sq = jnp.sum(diffs ** 2, axis=1)
    k        = jnp.argmax(dists_sq)

    ia = _PAIR_INDICES[k, 0]
    ib = _PAIR_INDICES[k, 1]
    ic = _INNER[k]

    A        = sources[ia]
    B        = sources[ib]
    baseline = jnp.sqrt(dists_sq[k])
    u        = (B - A) / baseline
    v        = jnp.stack([-u[1], u[0]])

    def proj(P):
        rel = P - A
        pu  = jnp.dot(rel, u) / baseline
        pv  = jnp.dot(rel, v) / baseline
        # 45° rotation: [[1, -1], [1, 1]] @ [pu, pv]
        # → A maps to (0, 0),  B maps to (1, 1)
        return jnp.stack([pu - pv, pu + pv])

    Cp = proj(sources[ic])
    return ia, ib, ic, A, B, baseline, Cp


_hash_triangles_batch = jax.jit(jax.vmap(_hash_triangle))

# Pre-warm JAX JIT at import time so the first pair doesn't pay
# the XLA compilation cost; subsequent calls for cached shapes are instant.
_hash_triangles_batch(jnp.zeros((1, 3, 2)))[0].block_until_ready()



def hash_image(fits_path, label, save=True):
    """
    Detect sources in a FITS image, hash all C(n,3) triangles, print a summary,
    save results to CSV, and return the triangles DataFrame.

    Args:
        fits_path : path to the FITS file
        label     : display name used in printed headings (e.g. "haystack", "needle")
    """
    t_start = time.perf_counter()

    with fits.open(fits_path) as hdul:
        image = hdul[0].data.astype(np.float64)

    print(f"Loaded {label}: shape={image.shape}  path={fits_path}")

    t_detect = time.perf_counter()
    centroids, _, fluxes = detect_centroids(
        image,
        sigma=constants_astrometry.DETECTION_SIGMA,
        npixels=constants_astrometry.DETECTION_NPIXELS,
    )
    t_detect = time.perf_counter() - t_detect

    n_sources_total = len(centroids)
    print(f"Detected {n_sources_total} sources  ({t_detect*1e3:.1f} ms)")

    # Limit to brightest N sources
    max_src = constants_astrometry.MAX_HASH_SOURCES
    if max_src > 0 and n_sources_total > max_src:
        top_idx   = np.argsort(fluxes)[::-1][:max_src]
        centroids = centroids[top_idx]
        print(f"Using brightest {max_src} of {n_sources_total} sources for hashing")
    n_sources = len(centroids)

    if n_sources < 3:
        print(f"Fewer than 3 sources detected in {label} — cannot form any triangles.")
        return pd.DataFrame(), centroids

    n_triangles = comb(n_sources, 3)
    print(f"Building triangles: C({n_sources}, 3) = {n_triangles} triangles...")

    flat      = np.fromiter(
        (x for c in combinations(range(n_sources), 3) for x in c),
        dtype=np.int32,
        count=n_triangles * 3,
    )
    combo_arr   = flat.reshape(n_triangles, 3)
    all_sources = jnp.array(centroids[combo_arr])

    t_hash = time.perf_counter()
    ia_j, ib_j, ic_j, A_j, B_j, bl_j, Cq_j = _hash_triangles_batch(all_sources)
    bl_j.block_until_ready()
    t_hash = time.perf_counter() - t_hash

    t_total = time.perf_counter() - t_start

    ia_np = np.array(ia_j, dtype=int)
    ib_np = np.array(ib_j, dtype=int)
    ic_np = np.array(ic_j, dtype=int)
    A_np  = np.array(A_j)
    B_np  = np.array(B_j)
    bl_np = np.array(bl_j)
    Cq_np = np.array(Cq_j)

    combo_arr_int = combo_arr
    baselines     = bl_np
    Cxs, Cys      = Cq_np[:, 0], Cq_np[:, 1]

    # Precompute global source indices for print and DataFrame
    idx_a_all = combo_arr_int[np.arange(n_triangles), ia_np]
    idx_b_all = combo_arr_int[np.arange(n_triangles), ib_np]
    idx_c_all = combo_arr_int[np.arange(n_triangles), ic_np]

    # ── Print results ──────────────────────────────────────────────────────────
    if constants_astrometry.VERBOSE:
        print(f"\n{'='*72}")
        print(f"  Triangle hashing — {label}  (pair {PAIR_NUM})")
        print(f"{'='*72}")

        print(f"\n  Coordinate system")
        print(f"  {'-'*68}")
        print(f"  {'Origin (0, 0)':<32}  anchor source A  (furthest-apart pair)")
        print(f"  {'Unit point (1, 1)':<32}  anchor source B  (furthest-apart pair)")
        print(f"  {'Basis':<32}  45° rotation of the A→B frame")
        print(f"  {'Invariances':<32}  translation, rotation, scale")
        print(f"  {'Hash':<32}  (Cx, Cy)  - inner source C coords")

        print(f"\n  Summary")
        print(f"  {'-'*68}")
        print(f"  {'Sources detected':<32}  {n_sources}")
        print(f"  {'Triangles built':<32}  {n_triangles}")
        print(f"  {'Source detection time':<32}  {t_detect*1e3:.2f} ms")
        print(f"  {'Triangle hashing time':<32}  {t_hash*1e3:.2f} ms")
        print(f"  {'Total time':<32}  {t_total*1e3:.2f} ms")

        print(f"\n  Baseline statistics (|AB| in pixels)")
        print(f"  {'-'*68}")
        print(f"  {'mean':<16} {baselines.mean():>10.2f} px")
        print(f"  {'std':<16} {baselines.std():>10.2f} px")
        print(f"  {'min':<16} {baselines.min():>10.2f} px")
        print(f"  {'max':<16} {baselines.max():>10.2f} px")

        print(f"\n  Hash coordinate statistics  (all triangles)")
        print(f"  {'-'*68}")
        print(f"  {'coord':<10} {'mean':>10}  {'std':>10}  {'min':>10}  {'max':>10}")
        print(f"  {'-'*54}")
        for lbl, vals in [("Cx", Cxs), ("Cy", Cys)]:
            print(f"  {lbl:<10} {vals.mean():>+10.4f}  {vals.std():>10.4f}  "
                  f"{vals.min():>+10.4f}  {vals.max():>+10.4f}")

        n_show = min(10, n_triangles)
        print(f"\n  First {n_show} triangles (detailed)")
        print(f"  {'-'*68}")
        hdr = (f"  {'Tri':<6} {'A':>5} {'B':>5} {'C':>5}  "
               f"{'baseline(px)':>14}  {'Cx':>8} {'Cy':>8}")
        print(hdr)
        print(f"  {'-'*68}")
        for i in range(n_show):
            print(f"  {str(tuple(int(x) for x in combo_arr_int[i])):<6}  "
                  f"{idx_a_all[i]:>4} {idx_b_all[i]:>4} {idx_c_all[i]:>4}  "
                  f"{bl_np[i]:>14.2f}  "
                  f"{Cxs[i]:>+8.4f} {Cys[i]:>+8.4f}")

        print(f"\n{'='*72}\n")
    else:
        print(f"  [{label}] {n_sources} sources → {n_triangles} triangles  "
              f"(detect {t_detect*1e3:.0f} ms, hash {t_hash*1e3:.0f} ms)")

    # ── Build DataFrame ────────────────────────────────────────────────────────
    pair_dir = os.path.dirname(fits_path)
    pair_num = os.path.basename(pair_dir).split("_")[1]
    csv_path = os.path.join(pair_dir, f"triangles_{label}_{pair_num}.csv")

    df = pd.DataFrame({
        'idx_a':       idx_a_all,
        'idx_b':       idx_b_all,
        'idx_c':       idx_c_all,
        'A_px_x':      A_np[:, 0],
        'A_px_y':      A_np[:, 1],
        'B_px_x':      B_np[:, 0],
        'B_px_y':      B_np[:, 1],
        'baseline_px': bl_np,
        'Cx':          Cxs,
        'Cy':          Cys,
    })
    if save:
        df.insert(0, 'src_indices',
                  [str(tuple(combo_arr_int[i])) for i in range(n_triangles)])
        df.to_csv(csv_path, index=False)
        print(f"Saved {n_triangles} triangles → {csv_path}")

    return df, centroids


def main(haystack_path, needle_path):
    (_, _) = hash_image(haystack_path, label="haystack")
    (_, _) = hash_image(needle_path,   label="needle")


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    haystack_path = os.path.join(base, f"haystack_{PAIR_NUM}.fits")
    needle_path   = os.path.join(base, f"needle_{PAIR_NUM}.fits")

    _, haystack_centroids = hash_image(haystack_path, label="haystack", save=False)
    _, needle_centroids   = hash_image(needle_path,   label="needle",   save=False)

    with fits.open(haystack_path) as hdul:
        haystack_image = hdul[0].data.astype(np.float64)
    with fits.open(needle_path) as hdul:
        needle_image = hdul[0].data.astype(np.float64)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, image, centroids, label in [
        (axes[0], haystack_image, haystack_centroids, f"Haystack (pair {PAIR_NUM})"),
        (axes[1], needle_image,   needle_centroids,   f"Needle (pair {PAIR_NUM})"),
    ]:
        ax.imshow(image, origin="lower", cmap="viridis")
        if len(centroids):
            ax.scatter(centroids[:, 0], centroids[:, 1],
                       s=40, facecolors="none", edgecolors="red", linewidths=0.8)
        ax.set_title(f"{label}\n{len(centroids)} sources detected")
        ax.set_xlabel("x (px)")
        ax.set_ylabel("y (px)")

    fig.suptitle("Detected sources", fontsize=13)
    plt.tight_layout()
    plt.show()
