import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import constants_astrometry as constants

PAIR_NUM = "0004"

base = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data_generation", "dataset", f"pair_{PAIR_NUM}")


def main(pair_dir, plot=False):
    pair_num = os.path.basename(pair_dir).split("_")[1]
    candidate_csv    = os.path.join(pair_dir, f"centroids_candidate_{pair_num}.csv")
    detranslated_csv = os.path.join(pair_dir, f"centroids_detranslated_{pair_num}.csv")
    output_csv       = os.path.join(pair_dir, f"centroids_matched_{pair_num}.csv")

    # Load the centroid lists produced by detect_centroids.py. Each row is a
    # single source with its flux-weighted pixel position (x, y) and integrated flux.
    df_cand = pd.read_csv(candidate_csv)
    df_det  = pd.read_csv(detranslated_csv)

    radius = constants.CENTROID_MATCH_RADIUS

    matches = []
    # For each source in the candidate image, find its nearest neighbour in the
    # detranslated image. Only accept the pair if the distance is within the
    # match radius — larger separations likely indicate a false pairing caused
    # by a mis-detected source or a centroid that shifted more than expected.
    for _, row_c in df_cand.iterrows():
        dists = np.sqrt((df_det['x'] - row_c['x'])**2 + (df_det['y'] - row_c['y'])**2)
        idx   = dists.idxmin()
        if dists[idx] <= radius:
            matches.append({
                # Pixel positions of the matched source in each image
                'x_candidate':    row_c['x'],
                'y_candidate':    row_c['y'],
                'flux_candidate': row_c['flux'],
                'x_detranslated': df_det.loc[idx, 'x'],
                'y_detranslated': df_det.loc[idx, 'y'],
                'flux_detranslated': df_det.loc[idx, 'flux'],
                # Separation between the two centroids, kept as a quality indicator
                'dist':           dists[idx],
            })

    # Save the matched pairs to CSV. Row i in this file corresponds to the same
    # physical source in both images, so build_triangles.py can safely use the
    # same index to address both point sets.
    df_matched = pd.DataFrame(matches)
    df_matched.to_csv(output_csv, index=False)
    print(f"Matched {len(df_matched)} / {len(df_cand)} candidate sources  (radius={radius} px)")
    print(df_matched.to_string(index=False))
    print(f"Saved {output_csv}")

    if plot:
        fig, ax = plt.subplots(figsize=(7, 7))

        # All detected sources in the candidate image, including unmatched ones
        ax.scatter(df_cand['x'],
                   df_cand['y'],
                   s=60, facecolors='none',
                   edgecolors='grey',
                   linewidths=0.8,
                   label='All candidate')

        # All detected sources in the detranslated image, including unmatched ones
        ax.scatter(df_det['x'],
                   df_det['y'],
                   s=60, facecolors='none',
                   edgecolors='darkgrey',
                   linewidths=0.8,
                   label='All detranslated')

        # Subset of candidate sources that found a match within the radius
        ax.scatter(df_matched['x_candidate'],
                   df_matched['y_candidate'],
                   s=80, marker='x',
                   color='red',
                   linewidths=1.2,
                   label='Matched (candidate)')

        # Corresponding matched sources in the detranslated image
        ax.scatter(df_matched['x_detranslated'],
                   df_matched['y_detranslated'],
                   s=80, marker='x',
                   color='orange',
                   linewidths=1.2,
                   label='Matched (detranslated)')

        # Draw a line between each matched pair to make mismatches easy to spot visually
        for _, row in df_matched.iterrows():
            ax.plot([row['x_candidate'], row['x_detranslated']],
                    [row['y_candidate'], row['y_detranslated']], 
                    'r-', linewidth=0.5, alpha=0.5)

        ax.set_aspect('equal')
        ax.invert_yaxis()
        ax.set_title(f"Matched centroids - pair {pair_num}  ({len(df_matched)} matches)")
        plt.tight_layout()
        plt.show()

    return df_matched


if __name__ == "__main__":
    main(pair_dir=base, plot=True)
