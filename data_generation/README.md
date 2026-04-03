# Data Generation

This directory generates synthetic astronomical image pairs used to test and evaluate the astrometry pipeline. Each pair consists of a **haystack** (a large reference image) and a **needle** (a smaller image taken from within the haystack, with a deliberately introduced WCS error).

---

## What gets generated

For each pair, the following files are saved under `dataset/pair_XXXX/`:

| File | Description |
|---|---|
| `haystack_XXXX.fits` | Noisy haystack image with a valid WCS |
| `haystack_XXXX_clean.fits` | Noise-free haystack (used internally during needle extraction) |
| `needle_XXXX.fits` | Noisy needle with a corrupted WCS (translation + rotation errors injected) |
| `needle_XXXX_ground_truth.fits` | Clean needle with the correct WCS, used for evaluation only |

---

## Haystack generation (`create_haystack.py`)

The haystack is a synthetic image of a galaxy field. Each galaxy is modelled as a 2D Gaussian blob:

```
I(x, y) = A · exp( -(x_rot² / 2σ_x² + y_rot² / 2σ_y²) )
```

where `x_rot, y_rot` are the pixel coordinates rotated by a random position angle. This produces elliptical galaxies with a range of sizes, axis ratios, and orientations.

**Randomised per galaxy:**
- Centre position: uniform over the image (with a 100 px border)
- Major axis sigma (σ_y): drawn from `GALAXY_SIZE_Y = (min, max)` px
- Minor axis sigma (σ_x): drawn from `GALAXY_SIZE_X = (min, max)` px
- Position angle: uniform in [0°, 180°]
- Peak amplitude: uniform in [0.5, 1.0]

The number of galaxies is set by `GALAXY_DENSITY` (galaxies per 10⁶ pixels, meaning a 1000×1000 square), scaled to the image size. After all galaxies are drawn, Gaussian noise with sigma `HAYSTACK_NOISE` is added.

A TAN-projection WCS header is attached, centred at (RA, Dec) = (0°, 0°).

---

## Needle generation (`create_needle.py`)

The needle is a 300 × 300 px crop from the **clean** haystack, processed and corrupted to simulate a real observation of the same patch of sky.

### Region selection

A candidate region is drawn at a random position in the haystack. It is accepted only if it contains at least `MIN_GALAXIES` detectable sources, with at least `MIN_CENTRAL_GALAXIES` of them within the central `CENTRAL_REGION_SIZE × CENTRAL_REGION_SIZE` pixels. This ensures the needle contains enough structure for the pipeline to work with.

The search starts from a larger **buffer region** (`BUFFER_SIZE × BUFFER_SIZE`) so that the rotation step below does not wrap empty sky into the final crop.

### Transformations applied

1. **Rotation**: the buffer crop is rotated by a random angle drawn from U(−`NEEDLE_ANGLE_MAX`, +`NEEDLE_ANGLE_MAX`) degrees. The rotation is applied with `scipy.ndimage.rotate`, and the image is centre-cropped back to `NEEDLE_SIZE × NEEDLE_SIZE` after rotation.

2. **Stretch**: an optional zoom is applied along each axis (`NEEDLE_STRETCH_Y`, `NEEDLE_STRETCH_X`). Both default to 1.0 (no stretch).

3. **Noise**: Gaussian noise with sigma `NEEDLE_NOISE` is added. This is intentionally higher than the haystack noise to simulate the needle coming from a different, noisier instrument.

### WCS error injection

The needle's WCS is constructed from the haystack WCS at the extracted position, then deliberately corrupted:

- **Translation error**: CRPIX is offset by random draws from U(−`WCS_ERROR_MAX`, +`WCS_ERROR_MAX`) independently on each axis.
- **Rotation error**: the pixel data itself is rotated (above), but the WCS is not updated to reflect this — the WCS header retains the un-rotated orientation.

The true error values are stored in the header for later evaluation:

| Header keyword | Meaning |
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
./run.sh data-gen
./run.sh data-gen --num-pairs 20 --num-cores 8
```

Or directly:

```bash
python 01_run_data_generation_pipeline.py
```

All parameters are controlled via `constants.py` in the project root, or overridden from the command line via `run.sh`.
