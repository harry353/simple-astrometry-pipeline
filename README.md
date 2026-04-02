# Simple Astrometry Pipeline

A pipeline for correcting World Coordinate System (WCS) errors in astronomical images. Given a large reference image (the *haystack*) and a smaller query image (the *needle*) with an imprecise WCS system, the pipeline recovers the correct position and orientation of the needle within the haystack.

---

## Approach

### 1. Detranslation - correcting the position error

The needle is localised within the haystack using **template matching** (cross-correlation). Both images are first pre-filtered with a Wiener filter to suppress (the artifically added) noise. The peak position is refined to sub-pixel accuracy using a parabolic fit.

The detected shift tells us where the needle actually sits in the haystack. The needle's WCS is corrected by resetting its reference pixel (CRPIX) to the geometric centre and updating the reference sky coordinate (CRVAL) to the sky position of the detected centre.

### 2. Derotation - correcting the rotation error

Once the position is corrected, the rotation correction is applied using **geometric hashing with triangles**:

1. Sources (galaxies) are detected in both the candidate cutout and the detranslated needle using image segmentation algorithm from `photutils`.
2. Detected sources are matched across the two images by nearest-neighbour search within a pixel radius.
3. Every unique combination of 3 matched sources forms a triangle. Each triangle is described by its side-length ratios and interior angles, quantities that are invariant to rotation, so they should be identical in both images for a correct match.
4. A rotation angle is estimated, one for each triangle, by solving a linear least-squares system of equations. The angle is then found by voting (histogram mode).
5. The rotation is corrected by updating the CD matrix in the WCS header. The pixel data is left untouched, meaning no resampling or interpolation is applied.

---

## Data Generation

Synthetic data can be generated to test the pipeline. Each pair consists of:

- A **haystack**: a large image populated with synthetic galaxies (2D Gaussians at random positions, sizes, and orientations), with Gaussian noise added.
- A **needle**: a small crop from the haystack, with independent noise, a deliberate WCS translation error (random offset up to `WCS_ERROR_MAX` pixels on each axis), and a small random rotation (up to `NEEDLE_ANGLE_MAX` degrees). This error can be set from the `constants.py` file.

The true WCS error is stored in the needle FITS header (`WERR_XPX`, `WERR_YPX`, `NANGLE`) for evaluation.

---

## Installation

Python 3.10 or later is recommended. Using a conda environment is the easiest way to get the dependencies right.

**With conda:**
```bash
conda create -n simple_astrometry_pipeline python=3.11
conda activate simple_astrometry_pipeline
conda install numpy scipy astropy matplotlib pandas
pip install photutils
```

**With pip:**
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install numpy scipy astropy photutils matplotlib pandas
```

Then clone or download the repository and make the run script executable:
```bash
git clone <repo-url>
cd simple_astrometry_pipeline
chmod +x run.sh
```

---

## Running

```bash
# Run the full pipeline (data generation → detranslation → derotation)
./run.sh

# Run only one stage
./run.sh data-gen
./run.sh detranslation
./run.sh derotation

# Override parameters from the command line
./run.sh all --num-pairs 20 --num-cores 8
./run.sh all --wcs-error 20 --angle-max 1.5
./run.sh detranslation --pairs 5
./run.sh all --haystack-size 2000 --needle-size 400 --seed 99

# Help
./run.sh --help
```

All parameters default to the values in `constants.py`. Command-line arguments override them for that run only, and `constants.py` is not modified.

---

## Parameters (`constants.py`)

| Parameter | Default | Description |
|---|---|---|
| `NUM_PAIRS` | 10 | Haystack/needle pairs to generate |
| `NUM_CORES` | 16 | Parallel workers for data generation |
| `HAYSTACK_SIZE` | 1500 | Haystack image width and height (px) |
| `NEEDLE_SIZE` | 300 | Needle cutout size (px) |
| `HAYSTACK_NOISE` | 0.1 | Gaussian noise sigma for haystack |
| `NEEDLE_NOISE` | 0.2 | Gaussian noise sigma for needle |
| `GALAXY_DENSITY` | 30 | Galaxies per 10⁶ pixels |
| `WCS_ERROR_MAX` | 15 | Max translation error injected (px) |
| `NEEDLE_ANGLE_MAX` | 0.8 | Max rotation error injected (deg) |
| `CENTROID_MATCH_RADIUS` | 3 | Max separation for centroid matching (px) |
| `RANDOM_SEED` | 25 | Seed for reproducible data generation |
| `TRANSLATION_PIPELINE_PAIRS` | 10 | Pairs to process (0 = all) |

---

## Dependencies

```
numpy
scipy
astropy
photutils
matplotlib
pandas
```
