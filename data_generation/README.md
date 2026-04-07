# Data Generation

This directory generates synthetic astronomical image pairs used to test and evaluate the astrometry pipeline. Each pair consists of a **haystack** (a large reference image) and a **needle** (a smaller image taken from within the haystack, with a deliberately introduced WCS error).

---

## Files

| File | Description |
|---|---|
| `01_create_haystack_galsim.py` | GalSim haystack backend: physically realistic MUSE-like images |
| `01b_create_haystack_gaussian.py` | Gaussian-blob haystack backend: fast, no extra dependencies |
| `02_create_needle.py` | Cuts, transforms, and corrupts the needle from the haystack |
| `03_create_data_generation_diagnostics.py` | Generates a diagnostic PNG summarising the dataset |
| `04_plot_needle_haystack.py` | Standalone visualisation of a single haystack/needle pair |

Generated data is written to `dataset/pair_XXXX/`.

---

## What gets generated

For each pair, the following files are saved under `dataset/pair_XXXX/`:

| File | Description |
|---|---|
| `haystack_XXXX.fits` | Noisy haystack image with a valid WCS |
| `haystack_XXXX.png` | PNG preview of the noisy haystack |
| `needle_XXXX.fits` | Noisy needle with corrupted WCS (translation + rotation errors injected) |
| `needle_XXXX.png` | PNG preview of the noisy needle |
| `needle_XXXX_ground_truth.fits` | Clean needle with the correct WCS, used for evaluation only |
| `needle_XXXX_ground_truth.png` | PNG preview of the ground-truth needle |

---

## Haystack backends

The backend is selected by `USE_GALSIM` in `constants_datagen.py`. If `USE_GALSIM = True` but `galsim` is not installed, the pipeline falls back to the Gaussian-blob backend automatically.

### Gaussian-blob backend (`01b_create_haystack_gaussian.py`)

Each galaxy is modelled as a 2D Gaussian ellipse:

```
I(x, y) = A · exp( -(x_rot² / 2σ_x² + y_rot² / 2σ_y²) )
```

where `x_rot, y_rot` are pixel coordinates rotated by a random position angle, producing elliptical galaxies at a range of sizes, axis ratios, and orientations.

**Randomised per galaxy:**
- Centre position: uniform over the image (with a 100 px border)
- Major axis sigma (σ_y): drawn from `GALAXY_SIZE_Y = (min, max)` px
- Minor axis sigma (σ_x): drawn from `GALAXY_SIZE_X = (min, max)` px
- Position angle: uniform in [0°, 180°]
- Peak amplitude: uniform in [0.5, 1.0]

Galaxy count is set by `GALAXY_DENSITY` (per 10⁶ pixels), scaled to `HAYSTACK_SIZE`. Gaussian noise with sigma `HAYSTACK_NOISE` is added. The output is normalised to [0, 1].

### GalSim backend (`01_create_haystack_galsim.py`)

Produces physically realistic images modelled after VLT/MUSE white-light observations (λ = 480–930 nm, pixel scale 0.2 arcsec/px, 1500 × 1500 px field). Requires `galsim`.

**Object population**: `NOBJ` objects are drawn with probabilities:
- **Faint galaxies**: Sérsic exponential profiles (half-light radius ~ 0.3″) drawn at low flux, representing background sources.
- **Stars**: point sources convolved with the PSF, flux drawn from a log-normal distribution.
- **Bright galaxies**: larger exponential profiles (half-light radius ~ 2″) at higher flux, representing foreground/resolved galaxies.

Morphologies are drawn from the HST COSMOS catalog (I < 25.2 and I < 23.5 samples) and all objects are convolved with a **Moffat PSF** (β = 2.5, FWHM = 0.8″).

**Noise model** (in order of application):
1. Sky background pedestal: 200 ADU/pixel (Paranal dark sky, V ≈ 21.6 mag/arcsec²)
2. Poisson shot noise on the sky + source signal
3. Sky background subtracted
4. Gaussian read noise: σ = 3 e⁻/pixel (MUSE CCD spec)
5. Gain conversion at 1.0 e⁻/ADU
6. Quantisation to integer ADU

The image is **normalised to [0, 1]** using a percentile-based stretch (1st–99th percentile of the noisy image) before being passed to the needle pipeline. The noiseless version of the image is returned as the "clean" haystack so that needle region selection and ground-truth extraction use source positions rather than noise.

A tangent-plane WCS (RA---TAN / DEC--TAN) centred at (RA, Dec) = (0°, 0°) with CDELT = ±0.2″/px is written into the FITS header.

**Relevant constants** (`constants_datagen.py`):

| Constant | Default | Description |
|---|---|---|
| `NOBJ` | 100 | Total number of objects per image |
| `N_PIX` | 1500 | Image side length in pixels |
| `PIXEL_SCALE` | 0.2 | Arcsec per pixel |
| `P_FAINT_GALAXY` | 0.1 | Cumulative threshold for faint galaxies |
| `P_STAR` | 0.2 | Cumulative threshold for stars (bright galaxies fill remainder) |

---

## Needle generation (`02_create_needle.py`)

The needle is a crop from the **clean** haystack, transformed and corrupted to simulate a real observation of the same patch of sky.

### Region selection

A candidate region is drawn at a random position. It is accepted only if it contains at least `MIN_GALAXIES` detectable sources, with at least `MIN_CENTRAL_GALAXIES` of them within the central `CENTRAL_REGION_SIZE × CENTRAL_REGION_SIZE` px. This ensures the needle contains enough structure for the pipeline.

The search starts from a larger **buffer region** (`BUFFER_SIZE × BUFFER_SIZE`) so that the rotation step below does not wrap empty sky into the final crop.

### Transformations applied

1. **Rotation**: the buffer crop is rotated by a random angle drawn from U(−`NEEDLE_ANGLE_MAX`, +`NEEDLE_ANGLE_MAX`) degrees, using `scipy.ndimage.rotate`, then centre-cropped to `NEEDLE_SIZE × NEEDLE_SIZE`.
2. **Stretch**: optional zoom per axis (`NEEDLE_STRETCH_Y`, `NEEDLE_STRETCH_X`), both defaulting to 1.0.
3. **Noise**: Gaussian noise with sigma `NEEDLE_NOISE` is added. Intentionally higher than the haystack noise.

### WCS error injection

The needle WCS is derived from the haystack WCS at the extracted position, then deliberately corrupted:

- **Translation error**: CRPIX offset by random draws from U(−`WCS_ERROR_MAX`, +`WCS_ERROR_MAX`) independently on each axis.
- **Rotation error**: the pixel data is rotated but the WCS is not updated, leaving the header with the wrong orientation.

The true error values are stored in the header for evaluation:

| Keyword | Meaning |
|---|---|
| `WERR_XPX` | Translation error in x (pixels) |
| `WERR_YPX` | Translation error in y (pixels) |
| `WERR_XAS` | Translation error in x (arcseconds) |
| `WERR_YAS` | Translation error in y (arcseconds) |
| `NANGLE` | Rotation error (degrees) |

---

## Running

Data generation is run automatically by the top-level pipeline:

```bash
./run.sh data
./run.sh data --num-pairs 20 --num-cores 8

# GalSim backend
./run.sh data --use-galsim
./run.sh data --use-galsim --nobj 200 --npix 2000 --pixel-scale 0.15

# Gaussian-blob backend
./run.sh data --no-galsim --haystack-size 1000
```

Or directly:

```bash
python 01_run_data_generation_pipeline.py
```

All parameters are controlled via `constants_datagen.py` in the project root, or overridden from the command line via `run.sh`.
