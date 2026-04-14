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


# ── Source detection ──────────────────────────────────────────────────────────
DETECTION_SIGMA    = 3    # detection threshold in units of background sigma
DETECTION_NPIXELS  = 30   # minimum connected pixels to be counted as a source


# ── Quads pipeline ────────────────────────────────────────────────────────────
MAX_HASH_SOURCES         = 100    # brightest N sources used when forming triangles (0 = all)
TRIANGLE_MATCH_TOLERANCE = 0.02   # radius in 2D hash space (Cx, Cy) to accept a match
VERBOSE                  = False  # True = full per-step output; False = compact key results only
USE_GPU                  = True   # True = prefer GPU, fall back to CPU; False = always CPU

REFIT_DETECTION_SIGMA   = 3     # detection threshold for fit_transform source detection
REFIT_DETECTION_NPIXELS = 30    # minimum connected pixels for fit_transform source detection
INLIER_SEARCH_RADIUS    = 10.0  # px: wide radius used to score RANSAC hypotheses
INLIER_RADIUS           = 3.0   # px: tight radius used for the iterative refit
INLIER_MAX_ITERS        = 1     # max iterations for iterative refit convergence
FORCE_SCALE_ONE         = True  # True = force scale to 1 (no scale correction applied)
