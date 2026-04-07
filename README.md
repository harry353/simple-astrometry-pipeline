# Simple Astrometry Pipeline

A pipeline for correcting World Coordinate System (WCS) errors in astronomical images. Given a large reference image (the *haystack*) and a smaller query image (the *needle*) with an imprecise WCS, the pipeline recovers the correct position and orientation of the needle within the haystack.

---

## Approach

### 1. Detranslation: correcting the position error

The needle is localised within the haystack using **template matching** (cross-correlation). Both images are first pre-filtered with a Wiener filter to suppress noise. The peak position is refined to sub-pixel accuracy using a parabolic fit.

The detected shift tells us where the needle actually sits in the haystack. The needle's WCS is corrected by resetting its reference pixel (CRPIX) to the geometric centre and updating the reference sky coordinate (CRVAL) to the sky position of the detected centre.

### 2. Derotation: correcting the rotation error

Once the position is corrected, the rotation correction is applied using **geometric hashing with triangles**:

1. Sources (galaxies or stars) are detected in both the candidate cutout and the detranslated needle using image segmentation algorithm from `photutils`.
2. Detected sources are matched across the two images by nearest-neighbour search within a pixel radius.
3. Every unique combination of 3 matched sources forms a triangle. Each triangle is described by its side-length ratios and interior angles, quantities invariant to rotation.
4. A rotation angle is estimated per triangle by solving a linear least-squares system. The final angle is found by voting (histogram mode).
5. The WCS CD matrix is updated to apply the correction. Pixel data is left untouched; no resampling or interpolation.

---

## Data Generation

Synthetic data can be generated to test the pipeline. Each pair consists of:

- A **haystack**: a large image populated with synthetic galaxies (2D Gaussians at random positions, sizes, and orientations), with Gaussian noise added.
- A **needle**: a small crop from the haystack, with independent noise, a deliberate WCS translation error (random offset up to `WCS_ERROR_MAX` pixels on each axis), and a small random rotation (up to `NEEDLE_ANGLE_MAX` degrees). This error can be set from the `constants.py` file.

The true WCS error is stored in the needle FITS header (`WERR_XPX`, `WERR_YPX`, `NANGLE`) for evaluation.

### Haystack backends

Two backends are available, selected by the `USE_GALSIM` constant in `constants_datagen.py`:

**Gaussian-blob backend** (`USE_GALSIM = False`): fast, no extra dependencies. Each galaxy is a 2D Gaussian ellipse at a random position, size, and orientation. Gaussian noise is added to the final image.

**GalSim backend** (`USE_GALSIM = True`): produces physically realistic images modelled after VLT/MUSE white-light observations. Galaxy morphologies are drawn from the HST COSMOS catalog (I < 25.2 and I < 23.5 samples), convolved with a Moffat atmospheric PSF (FWHM = 0.8″). Sky background, Poisson shot noise, and CCD read noise are applied. The image is simulated at 0.2 arcsec/pixel over a 1500 × 1500 px field. Requires `galsim` to be installed, but falls back to the Gaussian-blob backend automatically if it is not.

---

## Installation

Python 3.11 or later is recommended. Using a conda environment is the easiest way to get the dependencies right.

**With conda (core pipeline):**
```bash
conda create -n simple_astrometry_pipeline python=3.11
conda activate simple_astrometry_pipeline
conda install numpy scipy astropy matplotlib pandas
pip install photutils
```

**With pip (core pipeline):**
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install numpy scipy astropy photutils matplotlib pandas
```

**GalSim backend (optional):**
```bash
pip install galsim
```


Then clone or download the repository and make the run script executable:
```bash
git clone <repo-url>
cd simple-astrometry-pipeline
chmod +x run.sh
```

---

## Running

```bash
# Run the full pipeline (data generation → detranslation → derotation)
./run.sh

# Run only one stage
./run.sh data
./run.sh detranslation
./run.sh derotation

# Override parameters at runtime (constants files are not modified)
./run.sh data --num-pairs 20 --num-cores 8
./run.sh data --use-galsim --nobj 200 --npix 2000
./run.sh data --no-galsim --haystack-size 1000
./run.sh all --wcs-error 20 --angle-max 1.5
./run.sh detranslation --pairs 5

# Help
./run.sh --help
```

---

## Parameters

### `constants_datagen.py`: data generation

| Parameter | Default | Description |
|---|---|---|
| `NUM_PAIRS` | 10 | Haystack/needle pairs to generate |
| `NUM_CORES` | 8 | Parallel worker processes |
| `RANDOM_SEED` | 25 | Seed for reproducibility |
| `USE_GALSIM` | `False` | Use GalSim backend if installed |
| `HAYSTACK_SIZE` | 1000 | Haystack image size in pixels (Gaussian backend) |
| `HAYSTACK_NOISE` | 0.1 | Gaussian noise sigma (Gaussian backend) |
| `GALAXY_DENSITY` | 30 | Galaxies per 10⁶ pixels (Gaussian backend) |
| `NEEDLE_SIZE` | 300 | Needle cutout size in pixels |
| `NEEDLE_NOISE` | 0.2 | Gaussian noise sigma added to needle |
| `WCS_ERROR_MAX` | 15 | Max translation error injected (px) |
| `NEEDLE_ANGLE_MAX` | 1.0 | Max rotation error injected (deg) |
| `MIN_GALAXIES` | 4 | Min detectable sources required in a needle region |
| `NOBJ` | 100 | Number of objects in the GalSim field |
| `N_PIX` | 1500 | GalSim image size in pixels |
| `PIXEL_SCALE` | 0.2 | GalSim pixel scale (arcsec/px) |
| `P_FAINT_GALAXY` | 0.1 | Cumulative probability threshold for faint galaxies |
| `P_STAR` | 0.2 | Cumulative probability threshold for stars |

### `constants_astrometry.py`: pipeline

| Parameter | Default | Description |
|---|---|---|
| `TRANSLATION_PIPELINE_PAIRS` | 0 | Pairs to run through correction pipelines (0 = all) |
| `DETECTION_SIGMA` | 5 | Source detection threshold (× background sigma) |
| `DETECTION_NPIXELS` | 40 | Minimum connected pixels to count as a source |
| `CENTROID_MATCH_RADIUS` | 3 | Max separation for centroid matching (px) |
| `SUBPIXEL_AREA` | 3 | Window size for sub-pixel peak refinement |

---

## Dependencies

**Core:**
```
numpy scipy astropy photutils matplotlib pandas
```

**GalSim backend (optional):**
```
galsim
```

---

## License

MIT License; see [LICENSE](LICENSE) for details.
