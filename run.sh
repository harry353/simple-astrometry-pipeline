#!/usr/bin/env bash
# Usage:
#   ./run.sh [pipeline] [options]
#
# Pipelines:
#   data          Run data generation pipeline
#   detranslation Run detranslation pipeline
#   derotation    Run derotation pipeline
#   all           Run all three in order (default)
#
# Options (data generation):
#   --num-pairs N       Number of pairs to generate
#   --num-cores N       Number of parallel worker processes
#   --seed N            Random seed
#   --wcs-error N       Max WCS translation error in pixels
#   --angle-max N       Max needle rotation error in degrees
#   --haystack-size N   Haystack image size in pixels (Gaussian backend)
#   --needle-size N     Needle cutout size in pixels
#   --use-galsim        Use GalSim backend for haystack generation
#   --no-galsim         Use Gaussian-blob backend for haystack generation
#   --nobj N            Number of objects in GalSim field
#   --npix N            GalSim image size in pixels
#   --pixel-scale F     GalSim pixel scale in arcsec/pixel
#
# Options (pipeline):
#   --pairs N           Number of pairs to run through correction pipelines (0 = all)
#
#   -h, --help          Show this help message

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-python3}"

# ── Defaults ──────────────────────────────────────────────────────────────────
PIPELINE="all"
PAIRS=""
NUM_PAIRS=""
NUM_CORES=""
SEED=""
WCS_ERROR=""
ANGLE_MAX=""
HAYSTACK_SIZE=""
NEEDLE_SIZE=""
USE_GALSIM=""
NOBJ=""
NPIX=""
PIXEL_SCALE=""

# ── Argument parsing ──────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        data|detranslation|derotation|all)
            PIPELINE="$1"; shift ;;
        --pairs)         PAIRS="$2";        shift 2 ;;
        --num-pairs)     NUM_PAIRS="$2";    shift 2 ;;
        --num-cores)     NUM_CORES="$2";    shift 2 ;;
        --seed)          SEED="$2";         shift 2 ;;
        --wcs-error)     WCS_ERROR="$2";    shift 2 ;;
        --angle-max)     ANGLE_MAX="$2";    shift 2 ;;
        --haystack-size) HAYSTACK_SIZE="$2"; shift 2 ;;
        --needle-size)   NEEDLE_SIZE="$2";  shift 2 ;;
        --use-galsim)    USE_GALSIM="True"; shift ;;
        --no-galsim)     USE_GALSIM="False"; shift ;;
        --nobj)          NOBJ="$2";         shift 2 ;;
        --npix)          NPIX="$2";         shift 2 ;;
        --pixel-scale)   PIXEL_SCALE="$2";  shift 2 ;;
        -h|--help)
            sed -n '2,29p' "$0" | sed 's/^# \?//'
            exit 0 ;;
        *)
            echo "Unknown argument: $1"
            exit 1 ;;
    esac
done

# ── Build a python snippet that overrides constants before running a script ───
overrides() {
    $PYTHON -c "
import sys, importlib.util, os

def load_mod(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    sys.modules[name] = mod
    return mod

astrometry = load_mod('constants_astrometry', '${SCRIPT_DIR}/constants_astrometry.py')
datagen    = load_mod('constants_datagen',    '${SCRIPT_DIR}/constants_datagen.py')

# constants_astrometry overrides
for attr, env in [
    ('TRANSLATION_PIPELINE_PAIRS', 'OVERRIDE_PAIRS'),
]:
    val = os.environ.get(env)
    if val is not None:
        setattr(astrometry, attr, type(getattr(astrometry, attr))(val))

# constants_datagen overrides
for attr, env in [
    ('NUM_PAIRS',      'OVERRIDE_NUM_PAIRS'),
    ('NUM_CORES',      'OVERRIDE_NUM_CORES'),
    ('RANDOM_SEED',    'OVERRIDE_SEED'),
    ('WCS_ERROR_MAX',  'OVERRIDE_WCS_ERROR'),
    ('NEEDLE_ANGLE_MAX', 'OVERRIDE_ANGLE_MAX'),
    ('HAYSTACK_SIZE',  'OVERRIDE_HAYSTACK_SIZE'),
    ('NEEDLE_SIZE',    'OVERRIDE_NEEDLE_SIZE'),
    ('USE_GALSIM',     'OVERRIDE_USE_GALSIM'),
    ('NOBJ',           'OVERRIDE_NOBJ'),
    ('N_PIX',          'OVERRIDE_NPIX'),
    ('PIXEL_SCALE',    'OVERRIDE_PIXEL_SCALE'),
]:
    val = os.environ.get(env)
    if val is not None:
        orig = getattr(datagen, attr)
        if isinstance(orig, bool):
            setattr(datagen, attr, val.lower() in ('true', '1', 'yes'))
        else:
            setattr(datagen, attr, type(orig)(val))

# Run the requested script
script = sys.argv[1]
spec2 = importlib.util.spec_from_file_location('__main__', script)
m = importlib.util.module_from_spec(spec2)
m.__name__ = '__main__'
spec2.loader.exec_module(m)
" "$1"
}

# ── Export overrides as environment variables ─────────────────────────────────
[[ -n "$PAIRS"         ]] && export OVERRIDE_PAIRS="$PAIRS"
[[ -n "$NUM_PAIRS"     ]] && export OVERRIDE_NUM_PAIRS="$NUM_PAIRS"
[[ -n "$NUM_CORES"     ]] && export OVERRIDE_NUM_CORES="$NUM_CORES"
[[ -n "$SEED"          ]] && export OVERRIDE_SEED="$SEED"
[[ -n "$WCS_ERROR"     ]] && export OVERRIDE_WCS_ERROR="$WCS_ERROR"
[[ -n "$ANGLE_MAX"     ]] && export OVERRIDE_ANGLE_MAX="$ANGLE_MAX"
[[ -n "$HAYSTACK_SIZE" ]] && export OVERRIDE_HAYSTACK_SIZE="$HAYSTACK_SIZE"
[[ -n "$NEEDLE_SIZE"   ]] && export OVERRIDE_NEEDLE_SIZE="$NEEDLE_SIZE"
[[ -n "$USE_GALSIM"    ]] && export OVERRIDE_USE_GALSIM="$USE_GALSIM"
[[ -n "$NOBJ"          ]] && export OVERRIDE_NOBJ="$NOBJ"
[[ -n "$NPIX"          ]] && export OVERRIDE_NPIX="$NPIX"
[[ -n "$PIXEL_SCALE"   ]] && export OVERRIDE_PIXEL_SCALE="$PIXEL_SCALE"

# ── Run the requested pipeline(s) ─────────────────────────────────────────────
run_data() {
    echo "── Data generation ──────────────────────────────────────────"
    overrides "${SCRIPT_DIR}/01_run_data_generation_pipeline.py"
}

run_detranslation() {
    echo "── Detranslation ────────────────────────────────────────────"
    overrides "${SCRIPT_DIR}/02_run_detranslation_pipeline.py"
}

run_derotation() {
    echo "── Derotation ───────────────────────────────────────────────"
    overrides "${SCRIPT_DIR}/03_run_derotation_pipeline.py"
}

case "$PIPELINE" in
    data)          run_data ;;
    detranslation) run_detranslation ;;
    derotation)    run_derotation ;;
    all)           run_data; run_detranslation; run_derotation ;;
esac
