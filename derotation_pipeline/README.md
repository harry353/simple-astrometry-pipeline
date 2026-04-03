# Derotation Pipeline

This pipeline corrects the rotational WCS error in the needle. It takes as input the outputs of the detranslation pipeline: the candidate needle (a haystack cutout at the detected position) and the detranslated needle (the original needle with its position WCS corrected), and estimates the rotation between them using geometric hashing with triangles. The rotation is corrected by updating the WCS header; no pixel data is resampled.

---

## Pipeline Steps

### Step 1: Source Detection (`01_detect_centroids.py`)

Galaxies are detected in both images using **image segmentation** from `photutils`. The algorithm works as follows:

1. The background level and noise are estimated.
2. A detection threshold is set at `median + 5σ` — any pixel above this is considered part of a source.
3. Connected regions of pixels above the threshold are labelled using `detect_sources`. Regions smaller than `NPIXELS = 40` pixels are discarded to filter out noise spikes.
4. For each labelled region, a **flux-weighted centroid** is computed using `SourceCatalog`.

Detection is run independently on both images. The centroid lists are saved to CSV with columns `x`, `y`, `flux`.

**Output:** `centroids_candidate_XXXX.csv`, `centroids_detranslated_XXXX.csv`

---

### Step 2: Centroid Matching (`02_match_centroids.py`)

Each source in the candidate image is matched to its nearest neighbour in the detranslated image. A match is accepted only if the two centroids are within `CENTROID_MATCH_RADIUS` pixels of each other. Pairs beyond this distance are rejected as likely false matches caused by a misidentified source or a centroid that shifted more than expected under the small rotation.

The result is a table where row `i` corresponds to the same physical source in both images. This correspondence is relied on by the triangle-building step.

**Output:** `centroids_matched_XXXX.csv`

---

### Step 3: Triangle Descriptors (`03_build_triangles.py`)

Every unique combination of 3 matched source pairs forms a triangle, giving C(n, 3) = n(n−1)(n−2)/6 triangles in total. For each triangle, two rotation-invariant descriptors are computed:

**Side-length ratios:** The three side lengths are sorted ascending as a ≤ b ≤ c. Two ratios, a/c and b/c, describe the triangle's shape independently of its size or orientation.

**Interior angles:** The three interior angles are computed from the side lengths via the law of cosines:

```
cos(A) = (b² + c² − a²) / (2bc)
```

where angle A is opposite side a, and similarly for B and C.

These five numbers (2 ratios + 3 angles) are invariant to rotation and scale, so they should be identical for the same triangle in both images. Any significant difference (the **delta** columns in the output CSV) indicates either a bad centroid match or a degenerate triangle (three nearly collinear points).

**Output:** `triangles_XXXX.csv`

---

### Step 4: Rotation Estimation (`04_solve_rotation.py`)

A rotation angle is estimated from each triangle independently, then combined into a single refined estimate.

**Per-triangle least squares:** For a pure rotation, the relationship between source points and destination points is:

```
⎡x'⎤   ⎡cos θ  −sin θ⎤ ⎡x⎤
⎣y'⎦ = ⎣sin θ   cos θ⎦ ⎣y⎦
```

This can be rearranged into a linear system `A [cos θ, sin θ]ᵀ = b`, where each source-destination pair contributes two rows to A and b. Solving with least squares gives cos θ and sin θ, from which θ = atan2(sin θ, cos θ).

**Voting:** The per-triangle estimates are binned into a histogram with 0.1° bins. The centre of the most populated bin (the mode) is taken as the **voted angle**. This is robust to outliers: a degenerate or mismatched triangle will produce a wildly wrong angle estimate, but as long as the majority of triangles agree, the mode is unaffected.

**Refinement:** A single least-squares fit is also run over all matched source pairs at once to produce the **refined angle**. This uses more data than any individual triangle and gives higher precision, but it is more sensitive to outliers, which is why the voting step is done first to confirm the estimate is sensible.

**Output:** `voted_angle`, `refined_angle` (passed to the next step)

#### Extending the system

The same least-squares framework can be extended to solve for more complex transformations by adding unknowns to the system.

**Rigid transformation (rotation + translation):** Add `tx`, `ty` as two extra unknowns. Each point pair contributes:

```
              ⎡cos θ⎤   
⎡x  -y  1  0⎤ ⎢sin θ⎥   ⎡x'⎤
⎣y   x  0  1⎦ ⎢  tx ⎥ = ⎣y'⎦
              ⎣  ty ⎦
```

Solve for `(cos θ, sin θ, tx, ty)`. Needs ≥ 2 point pairs.

---

### Step 5: Apply Derotation (`05_apply_derotation.py`)

The rotation is corrected by updating the **CD matrix** in the WCS header. The CD matrix is a 2×2 matrix that maps pixel offsets to sky offsets:

```
⎡ΔRA ⎤   ⎡CD1_1  CD1_2⎤ ⎡Δx⎤
⎣ΔDec⎦ = ⎣CD2_1  CD2_2⎦ ⎣Δy⎦
```

To apply a rotation of −θ (the negative of the voted angle, to undo the error), the existing CD matrix is left-multiplied by a rotation matrix:

```
CD_new = R(−θ) · CD_old
```

This rotates the sky coordinate frame (the grid essentially) without touching the pixel data. So no interpolation, no resampling, no blurring. The corrected CD matrix is then written back to the header, and any conflicting `CDELT`/`CROTA`/`PC` keywords are removed to avoid ambiguity. The applied angle is stored in the header as `DROT_ANG` for traceability.

**Output:** `corrected_needle_XXXX.fits`

---

### Step 6: Statistics (`06_print_derotation_statistics.py`)

Evaluates accuracy by comparing the applied correction (`DROT_ANG` from the corrected needle header) against the true rotation error (`NANGLE` from the original needle header).

- **Residual** = applied angle − true angle. A perfect correction gives residual = 0.
- **Improvement** = (1 − mean|residual| / mean|true angle|) × 100%.

A per-pair table is printed first, followed by aggregate statistics across all processed pairs.

---

## Output Files

For each pair, the following files are written to `dataset/pair_XXXX/`:

| File | Description |
|---|---|
| `centroids_candidate_XXXX.csv` | Detected source positions in the candidate needle |
| `centroids_detranslated_XXXX.csv` | Detected source positions in the detranslated needle |
| `centroids_matched_XXXX.csv` | Matched source pairs, one row per physical source |
| `triangles_XXXX.csv` | Triangle descriptors and quality deltas for all C(n,3) triangles |
| `corrected_needle_XXXX.fits` | Detranslated needle with rotation-corrected WCS |

---

## Running

```bash
./run.sh derotation
./run.sh derotation --pairs 5 --angle-max 1.5
```

Or directly:

```bash
python 03_run_derotation_pipeline.py
```

Individual scripts can be run standalone for debugging a single pair:

```bash
cd derotation_pipeline
python 01_detect_centroids.py
python 04_solve_rotation.py
```
