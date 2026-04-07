import os

# ── Paths ─────────────────────────────────────────────────────────────────────
FITS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


# ── Reproducibility ───────────────────────────────────────────────────────────
RANDOM_SEED = 25


# ── Dataset generation ────────────────────────────────────────────────────────
NUM_PAIRS  = 100  # total haystack/needle pairs to generate
NUM_CORES  = 8    # parallel worker processes


# ── Haystack ──────────────────────────────────────────────────────────────────
HAYSTACK_SIZE    = 1000      # square image size (px)
HAYSTACK_NOISE   = 0.1       # Gaussian noise sigma
GALAXY_DENSITY   = 30        # blobs per 1,000,000 pixels (Gaussian-blob backend)
GALAXY_SIZE_Y    = (2, 10)   # Gaussian sigma range along major axis (px)
GALAXY_SIZE_X    = (1, 5)    # Gaussian sigma range along minor axis (px)


# ── Needle ────────────────────────────────────────────────────────────────────
NEEDLE_SIZE      = 300   # square cutout size (px)
BUFFER_SIZE      = 400   # extraction region before centre-crop
NEEDLE_NOISE     = 0.2   # Gaussian noise sigma (higher than haystack by design)

NEEDLE_STRETCH_Y = 1.0   # zoom factor along y
NEEDLE_STRETCH_X = 1.0   # zoom factor along x

# Minimum galaxy content required in an accepted needle region
MIN_GALAXIES          = 4
MIN_CENTRAL_GALAXIES  = 1
CENTRAL_REGION_SIZE   = 50   # inner square (px) checked for central galaxies


# ── Needle WCS error ──────────────────────────────────────────────────────────
# Each needle's CRPIX is offset by a random draw from U(-WCS_ERROR_MAX, +WCS_ERROR_MAX)
# independently on each axis, simulating an imprecise initial astrometric solution.
WCS_ERROR_MAX    = 15    # max CRPIX error in pixels
NEEDLE_ANGLE_MAX = 1.0   # max rotation in degrees, drawn from U(-MAX, +MAX)


# ── Data generation backend ───────────────────────────────────────────────────
USE_GALSIM = True   # Use GalSim for haystack generation if galsim is installed;
                    # falls back to the simple Gaussian-blob generator otherwise.


# ── GalSim simulation parameters ─────────────────────────────────────────────
NOBJ        = 100       # number of objects to place in the simulated field
N_PIX       = 1500      # image side length in pixels
PIXEL_SCALE = 0.2       # arcsec/pixel

# Cumulative probability thresholds for object type selection (must end at 1.0)
P_FAINT_GALAXY  = 0.1   # faint background galaxies (I < 25.2 COSMOS catalog)
P_STAR          = 0.2   # stars -- drawn from a log-normal flux distribution
P_BRIGHT_GALAXY = 1.0   # bright/extended galaxies  (I < 23.5 COSMOS catalog)