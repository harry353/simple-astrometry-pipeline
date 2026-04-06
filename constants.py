import os

# ── Paths ─────────────────────────────────────────────────────────────────────
FITS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


# ── Reproducibility ───────────────────────────────────────────────────────────
RANDOM_SEED = 25


# ── Dataset generation ────────────────────────────────────────────────────────
NUM_PAIRS  = 100  # total haystack/needle pairs to generate
NUM_CORES  = 8    # parallel worker processes


# ── Haystack ──────────────────────────────────────────────────────────────────
HAYSTACK_SIZE    = 1500  # square image size (px)
HAYSTACK_NOISE   = 0.1   # Gaussian noise sigma
GALAXY_DENSITY   = 50    # blobs per 1,000,000 pixels
GALAXY_SIZE_Y    = (2, 10)   # Gaussian sigma range along major axis (px)   
GALAXY_SIZE_X    = (1, 5)   # Gaussian sigma range along minor axis (px)


# ── Needle ────────────────────────────────────────────────────────────────────
NEEDLE_SIZE      = 300   # square cutout size (px)
BUFFER_SIZE      = 400   # extraction region before centre-crop
NEEDLE_NOISE     = 0.2   # Gaussian noise sigma (higher than haystack by design)

NEEDLE_STRETCH_Y = 1.0   # zoom factor along y
NEEDLE_STRETCH_X = 1.0   # zoom factor along x

# Minimum galaxy content required in an accepted needle region
MIN_GALAXIES          = 5
MIN_CENTRAL_GALAXIES  = 1
CENTRAL_REGION_SIZE   = 50   # inner square (px) checked for central galaxies


# ── Needle error ───────────────────────────────────────────────────────────────
# Each needle's CRPIX is offset by a random draw from U(-WCS_ERROR_MAX, +WCS_ERROR_MAX)
# independently on each axis, simulating an imprecise initial astrometric solution.
# Similarly for NEEDLE_ANGLE_MAX, but for rotation instead of translation.
WCS_ERROR_MAX    = 15    # max CRPIX error in pixels, drawn from U(-WCS_ERROR_MAX, +WCS_ERROR_MAX)
NEEDLE_ANGLE_MAX = 1.0   # max rotation in degrees, needle drawn from U(-MAX, +MAX)


# ── Template matching ─────────────────────────────────────────────────────────
SUBPIXEL_AREA        = 3     # window size for quadratic peak refinement
SAVE_CORRELATION_MAP = True  # save PNG of correlation map per pair


# ── Pipeline control ──────────────────────────────────────────────────────────
TRANSLATION_PIPELINE_PAIRS = 100   # pairs to run through correction pipeline (0 = all)
SAVE_NEEDLE_COMPARISON     = True  # save needle comparison PNG per pair


# ── Candidate extraction ──────────────────────────────────────────────────────
CANDIDATE_NEEDLE_SIZE = 300


# ── Derotation ────────────────────────────────────────────────────────────────
CENTROID_MATCH_RADIUS = 3   # max distance (px) to match centroids across images

