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
# Options:
#   --pairs N           Number of pairs to process (0 = all)
#   --num-pairs N       Number of pairs to generate (data generation only)
#   --num-cores N       Number of parallel worker processes
#   --wcs-error N       Max WCS translation error in pixels
#   --angle-max N       Max needle rotation error in degrees
#   --haystack-size N   Haystack image size in pixels
#   --needle-size N     Needle cutout size in pixels
#   --seed N            Random seed
#   -h, --help          Show this help message

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-python3}"

# ── Defaults (mirror constants.py) ────────────────────────────────────────────
PIPELINE="all"
PAIRS=""
NUM_PAIRS=""
NUM_CORES=""
WCS_ERROR=""
ANGLE_MAX=""
HAYSTACK_SIZE=""
NEEDLE_SIZE=""
SEED=""

# ── Argument parsing ──────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        data-gen|detranslation|derotation|all)
            PIPELINE="$1"; shift ;;
        --pairs)        PAIRS="$2";        shift 2 ;;
        --num-pairs)    NUM_PAIRS="$2";    shift 2 ;;
        --num-cores)    NUM_CORES="$2";    shift 2 ;;
        --wcs-error)    WCS_ERROR="$2";    shift 2 ;;
        --angle-max)    ANGLE_MAX="$2";    shift 2 ;;
        --haystack-size) HAYSTACK_SIZE="$2"; shift 2 ;;
        --needle-size)  NEEDLE_SIZE="$2";  shift 2 ;;
        --seed)         SEED="$2";         shift 2 ;;
        -h|--help)
            sed -n '2,20p' "$0" | sed 's/^# \?//'
            exit 0 ;;
        *)
            echo "Unknown argument: $1"
            exit 1 ;;
    esac
done

# ── Build a python snippet that overrides constants before running a script ───
overrides() {
    python3 -c "
import sys, importlib.util, types

# Load constants module
spec = importlib.util.spec_from_file_location('constants', '${SCRIPT_DIR}/constants.py')
mod  = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

# Apply overrides from environment variables set by the shell
import os
for attr, env in [
    ('TRANSLATION_PIPELINE_PAIRS', 'OVERRIDE_PAIRS'),
    ('NUM_PAIRS',                   'OVERRIDE_NUM_PAIRS'),
    ('NUM_CORES',                   'OVERRIDE_NUM_CORES'),
    ('WCS_ERROR_MAX',               'OVERRIDE_WCS_ERROR'),
    ('NEEDLE_ANGLE_MAX',            'OVERRIDE_ANGLE_MAX'),
    ('HAYSTACK_SIZE',               'OVERRIDE_HAYSTACK_SIZE'),
    ('NEEDLE_SIZE',                 'OVERRIDE_NEEDLE_SIZE'),
    ('RANDOM_SEED',                 'OVERRIDE_SEED'),
]:
    val = os.environ.get(env)
    if val is not None:
        setattr(mod, attr, type(getattr(mod, attr))(val))

sys.modules['constants'] = mod

# Now run the requested script
script = sys.argv[1]
spec2 = importlib.util.spec_from_file_location('__main__', script)
m = importlib.util.module_from_spec(spec2)
m.__name__ = '__main__'
spec2.loader.exec_module(m)
" "$1"
}

# ── Export overrides as environment variables ─────────────────────────────────
[[ -n "$PAIRS"        ]] && export OVERRIDE_PAIRS="$PAIRS"
[[ -n "$NUM_PAIRS"    ]] && export OVERRIDE_NUM_PAIRS="$NUM_PAIRS"
[[ -n "$NUM_CORES"    ]] && export OVERRIDE_NUM_CORES="$NUM_CORES"
[[ -n "$WCS_ERROR"    ]] && export OVERRIDE_WCS_ERROR="$WCS_ERROR"
[[ -n "$ANGLE_MAX"    ]] && export OVERRIDE_ANGLE_MAX="$ANGLE_MAX"
[[ -n "$HAYSTACK_SIZE" ]] && export OVERRIDE_HAYSTACK_SIZE="$HAYSTACK_SIZE"
[[ -n "$NEEDLE_SIZE"  ]] && export OVERRIDE_NEEDLE_SIZE="$NEEDLE_SIZE"
[[ -n "$SEED"         ]] && export OVERRIDE_SEED="$SEED"

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
