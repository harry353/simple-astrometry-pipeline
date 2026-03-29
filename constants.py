import os

# ── Paths ─────────────────────────────────────────────────────────────────────
FITS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


# ── Reproducibility ───────────────────────────────────────────────────────────
RANDOM_SEED = 25


# ── Dataset generation ────────────────────────────────────────────────────────
NUM_PAIRS  = 200   # total haystack/needle pairs to generate
NUM_CORES  = 16    # parallel worker processes


# ── Haystack ──────────────────────────────────────────────────────────────────
HAYSTACK_SIZE    = 1500
HAYSTACK_NOISE   = 0.1   # Gaussian noise sigma
GALAXY_DENSITY   = 30    # blobs per 1,000,000 pixels


# ── Needle ────────────────────────────────────────────────────────────────────
NEEDLE_SIZE      = 300
BUFFER_SIZE      = 400   # extraction region before centre-crop
NEEDLE_NOISE     = 0.2   # Gaussian noise sigma (higher than haystack by design)

NEEDLE_ANGLE     = 0.0   # rotation in degrees
NEEDLE_STRETCH_Y = 1.0   # zoom factor along y
NEEDLE_STRETCH_X = 1.0   # zoom factor along x

# Minimum galaxy content required in an accepted needle region
MIN_GALAXIES          = 4
MIN_CENTRAL_GALAXIES  = 1
CENTRAL_REGION_SIZE   = 50   # inner square (px) checked for central galaxies


# ── WCS error ─────────────────────────────────────────────────────────────────
# Each needle's CRPIX is offset by a random draw from U(-WCS_ERROR_MAX, +WCS_ERROR_MAX)
# independently on each axis, simulating an imprecise initial astrometric solution.
WCS_ERROR_MAX = 15   # pixels


# ── Phase correlation ─────────────────────────────────────────────────────────
UPSAMPLE_FACTOR = 100   # subpixel precision = 1/UPSAMPLE_FACTOR px
SUBPIXEL_AREA   = 5     # window size for quadratic peak refinement


# ── Pipeline control ──────────────────────────────────────────────────────────
TRANSLATION_PIPELINE_PAIRS = 1     # pairs to run through correction pipeline (0 = all)
SAVE_NEEDLE_COMPARISON     = True  # save needle comparison PNG per pair


# ── Candidate extraction ──────────────────────────────────────────────────────
CANDIDATE_NEEDLE_SIZE = 300
