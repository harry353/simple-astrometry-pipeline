# Detranslation Pipeline

This pipeline corrects the translation WCS error in the needle, meaning the offset between where the needle's header says it is on the sky and where it actually is. It does this by finding the needle inside the haystack using template matching, then rewriting the needle's WCS to reflect the detected position.

---

## Pipeline Steps

### Step 1: Template Matching (`01_apply_matched_filtering.py`)

The needle is localised within the haystack using **ZNCC** (zero-mean normalised cross-correlation), implemented via `skimage.feature.match_template`. ZNCC computes a normalised correlation score at every possible position of the needle within the haystack. The position with the highest score is where the needle best matches the haystack.

Before matching, both images are pre-filtered with a **Wiener filter**. The kernel size is larger for the needle (11 px) than for the haystack (7 px) because the needle has higher noise.

**Sub-pixel refinement** is applied to the correlation peak using a parabolic fit. The three samples around the peak in each axis are fitted to a parabola, and the sub-pixel maximum is found analytically:

```
offset = -b / (2a)
```

where `a` and `b` are the quadratic and linear coefficients of the fit. This improves localisation precision from ~1 px to a fraction of a pixel.

**Output:** `shift_x, shift_y` — the offset of the needle centre from the haystack centre, in pixels.

---

### Step 2: Export Candidate Needle (`02_export_candidate_needle.py`)

Using the detected shift, a square cutout of `NEEDLE_SIZE × NEEDLE_SIZE` pixels is extracted from the haystack at the detected position. This cutout (called the **candidate needle**) is used in the derotation pipeline as the reference image for centroid matching.

A valid WCS is built for the cutout by inheriting the haystack WCS and updating the reference pixel and reference coordinate to the cutout's centre.

**Output:** `candidate_needle_XXXX.fits`

---

### Step 3: Correct Needle WCS (`03_correct_needle_wcs.py`)

The needle's WCS is corrected in two steps:

1. The detected pixel position in the haystack is converted to sky coordinates (RA, Dec) using the haystack WCS.
2. The needle header is updated:
   - `CRPIX` is reset to the geometric centre of the needle image (`width/2 + 0.5`, `height/2 + 0.5` in FITS 1-based convention).
   - `CRVAL` is set to the sky coordinates computed in step 1.

The pixel data is not modified, only the header changes.

**Output:** `detranslated_needle_XXXX.fits`

---

### Step 4: Statistics (`04_print_detranslation_statistics.py`)

Evaluates the pipeline's accuracy by comparing the corrected WCS against the ground truth stored in the needle header at data generation time.

For each pair, the following is computed:

- **Initial error**: the WCS translation error injected at data generation (`WERR_XPX`, `WERR_YPX`), expressed as a Euclidean distance in pixels and arcseconds.
- **Residual error**: the distance between the corrected WCS position and the true position, measured by projecting both sky coordinates back into haystack pixels.
- **Improvement**: `(1 - residual / initial) × 100%` (the fraction of the original error removed).

A per-pair table is printed first, followed by mean, std, min, and max across all processed pairs.

---

## Output Files

For each pair, the following files are written to `dataset/pair_XXXX/`:

| File | Description |
|---|---|
| `candidate_needle_XXXX.fits` | Haystack cutout at the detected position, with valid WCS |
| `candidate_needle_XXXX.png` | PNG preview of the candidate |
| `detranslated_needle_XXXX.fits` | Original needle with corrected WCS (CRPIX + CRVAL updated) |
| `detranslated_needle_XXXX.png` | PNG preview of the detranslated needle |
| `correlation_map_XXXX.png` | ZNCC correlation map (if `SAVE_CORRELATION_MAP = True`) |

---

## Running

Run via the top-level entry point:

```bash
./run.sh detranslation
./run.sh detranslation --pairs 5 --wcs-error 20
```

Or directly:

```bash
python 02_run_detranslation_pipeline.py
```

Individual scripts can also be run standalone for debugging a single pair:

```bash
cd detranslation_pipeline
python 01_apply_matched_filtering.py
python 04_print_detranslation_statistics.py
```
