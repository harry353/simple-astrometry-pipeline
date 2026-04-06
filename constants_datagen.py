OUTPATH     = 'output'  # directory where output FITS images are written
NOBJ        = 200       # number of objects to place in the simulated field
N_PIX       = 1500      # image side length in pixels
PIXEL_SCALE = 0.2       # arcsec/pixel

# Cumulative probability thresholds for object type selection (must end at 1.0)
P_FAINT_GALAXY  = 0.1   # faint background galaxies (I < 25.2 COSMOS catalog)
P_STAR          = 0.2   # stars -- drawn from a log-normal flux distribution
P_BRIGHT_GALAXY = 1.0   # bright/extended galaxies  (I < 23.5 COSMOS catalog)