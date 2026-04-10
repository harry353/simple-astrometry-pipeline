import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import combinations

PAIR_NUM = "0004"

base = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data_generation", "dataset", f"pair_{PAIR_NUM}")


def triangle_descriptors(p0, p1, p2):
    # Compute the length of each of the three sides
    sides = np.array([
        np.linalg.norm(p1 - p0),
        np.linalg.norm(p2 - p1),
        np.linalg.norm(p0 - p2),
    ])

    # Sort sides ascending so a <= b <= c regardless of vertex order.
    sides_sorted = np.sort(sides)
    a, b, c = sides_sorted

    # Side ratios: normalise by the longest side c so the descriptors are
    # invariant to scale. Two ratios fully describe the triangle's shape.
    ratio_ab = a / c   # shortest / longest
    ratio_bc = b / c   # middle   / longest

    # Interior angles via the law of cosines: cos(A) = (b²+c²-a²) / (2bc)
    # where angle A is opposite side a, B opposite b, C opposite c.
    cos_A = np.clip((b**2 + c**2 - a**2) / (2*b*c), -1, 1)
    cos_B = np.clip((a**2 + c**2 - b**2) / (2*a*c), -1, 1)
    cos_C = np.clip((a**2 + b**2 - c**2) / (2*a*b), -1, 1)
    angle_A = np.degrees(np.arccos(cos_A))
    angle_B = np.degrees(np.arccos(cos_B))
    angle_C = np.degrees(np.arccos(cos_C))

    return sides_sorted, ratio_ab, ratio_bc, angle_A, angle_B, angle_C


def main(pair_dir, plot=False):
    pair_num = os.path.basename(pair_dir).split("_")[1]
    # Load matched centroids from previous step. Each row corresponds 
    # to a single matched source pair
    matched_csv = os.path.join(pair_dir, f"centroids_matched_{pair_num}.csv")
    output_csv  = os.path.join(pair_dir, f"triangles_{pair_num}.csv")

    try:
        df = pd.read_csv(matched_csv)
    except pd.errors.EmptyDataError:
        print("No matched centroids (empty CSV). Skipping.")
        return False
    print(f"Loaded {len(df)} matched centroids")

    # Extract the (x,y) coordinates of the matched centroids in both images 
    # as numpy arrays.
    pts_cand = df[['x_candidate', 'y_candidate']].values
    pts_det  = df[['x_detranslated', 'y_detranslated']].values

    triangle_data = []
    # Iterate over every unique combination of 3 matched centroid indices, meaning
    # C(n,3) = n(n-1)(n-2)/6 triangles total. Because the centroids are already 
    # matched (index i in candidate corresponds to index i in detranslated, see
    # detect_centroids.py), we can form the same triangle in both images using the 
    # same indices and compare their descriptors directly.
    for i, j, k in combinations(range(len(df)), 3):

        # Extract the three vertex positions for this triangle in each image
        c0, c1, c2 = pts_cand[i], pts_cand[j], pts_cand[k]
        d0, d1, d2 = pts_det[i],  pts_det[j],  pts_det[k]

        # Compute geometric descriptors for each triangle.
        # Side ratios and interior angles are invariant to rotation and scale,
        # so they should be identical in both images if the match is correct.
        sides_c, rab_c, rbc_c, aA_c, aB_c, aC_c = triangle_descriptors(c0, c1, c2)
        sides_d, rab_d, rbc_d, aA_d, aB_d, aC_d = triangle_descriptors(d0, d1, d2)

        triangle_data.append({
            # Vertex indices, used later to retrieve the actual coordinates
            'i': i, 'j': j, 'k': k,
            # Candidate triangle descriptors
            'c_side_a': sides_c[0], 'c_side_b': sides_c[1], 'c_side_c': sides_c[2],
            'c_ratio_ab': rab_c, 'c_ratio_bc': rbc_c,
            'c_angle_A': aA_c, 'c_angle_B': aB_c, 'c_angle_C': aC_c,
            # Detranslated triangle descriptors
            'd_side_a': sides_d[0], 'd_side_b': sides_d[1], 'd_side_c': sides_d[2],
            'd_ratio_ab': rab_d, 'd_ratio_bc': rbc_d,
            'd_angle_A': aA_d, 'd_angle_B': aB_d, 'd_angle_C': aC_d,
            # Delta calculations for triangle quality diagnostics. Large deltas indicate
            # a bad centroid match (two different sources got paired up) or a degenerate 
            # triangle (three collinear points). We can filter these out later.
            'delta_ratio_ab': rab_c - rab_d,
            'delta_ratio_bc': rbc_c - rbc_d,
            'delta_angle_A':  aA_c  - aA_d,
            'delta_angle_B':  aB_c  - aB_d,
            'delta_angle_C':  aC_c  - aC_d,
        })

    # Save all triangle data to a CSV for later analysis
    df_tri = pd.DataFrame(triangle_data)
    df_tri.to_csv(output_csv, index=False)
    # i, j, k are row indices into the matched centroids CSV. So i=0, j=2, k=5 means the
    # triangle is formed by the 1st, 3rd, and 6th matched source pairs.
    print(f"Built {len(df_tri)} triangles from {len(df)} matched points")
    if df_tri.empty:
        print("Not enough points to construct any triangles (need at least 3). Skipping.")
        return False
    print(df_tri[['i','j','k','c_ratio_ab','c_ratio_bc','c_angle_A','c_angle_B','c_angle_C']].to_string(index=False))
    print(f"Saved {output_csv}")
    return True

    if plot:
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        for ax, pts, title in [
            (axes[0], pts_cand, "Candidate"),
            (axes[1], pts_det,  "Detranslated"),
        ]:
            # Draw every triangle as a polygon. Low alpha keeps the
            # plot readable when too many triangles overlap heavily.
            for i, j, k in df_tri[['i','j','k']].values:
                tri = plt.Polygon([pts[i], pts[j], pts[k]],
                                  fill=False, edgecolor='steelblue', linewidth=0.6, alpha=0.4)
                ax.add_patch(tri)
            # Plot the source positions
            ax.scatter(pts[:, 0], pts[:, 1], s=40, color='red', zorder=3)
            # Label each point with its index so individual sources can be cross-referenced
            # with the CSV rows when diagnosing bad matches
            for idx, (x, y) in enumerate(pts):
                ax.annotate(str(idx), (x, y), fontsize=7, color='white',
                            ha='center', va='center')
            ax.set_aspect('equal')
            ax.invert_yaxis()
            ax.set_title(f"{title} - {len(df_tri)} triangles")
            ax.set_facecolor('#1a1a2e')
        plt.tight_layout()
        plt.show()

    return df_tri


if __name__ == "__main__":
    main(pair_dir=base, plot=True)
