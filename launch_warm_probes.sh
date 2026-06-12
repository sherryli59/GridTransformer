#!/usr/bin/env bash
# Phase-3 warm-start probes (action-plan): A (joint d×Δs bias), F (near-contact RBF),
# G (kNN mask). Each probe fine-tunes the arc baseline for PROBE_EPOCHS epochs with
# its module zero-initialized, so epoch-0 loss == baseline and any movement in
# train/nll_jump vs train/nll_local is causally the probe's.
#
# Usage:
#   bash launch_warm_probes.sh            # run A, F, G sequentially
#   PROBES="A" bash launch_warm_probes.sh # run a subset
#
# Gate (read from lightning_logs after each run): jump-stratum NLL must beat the
# baseline's. C1/C2 are NOT here: the arc baseline has no rail, so probing rail
# placement on it would conflate "adding a rail" with the variant — they belong on
# the fixedrail baseline.
set -euo pipefail

cd /mnt/ssd/GridTransformer

BASELINE_CKPT="${BASELINE_CKPT:-/mnt/ssd/GridTransformer/lj_ckpts_lj27_pbc_arc_repr_norm/lj27_pbc_arc_repr_norm_fullcov/best.ckpt}"
PROBE_EPOCHS="${PROBE_EPOCHS:-10}"
PROBES="${PROBES:-A F G}"
PROBE_ROOT="${PROBE_ROOT:-/mnt/ssd/GridTransformer/lj_ckpts_lj27_pbc_arc_probes}"

# Shared baseline config (from the arc_repr_norm reproduce.sh), minus run-specific paths.
export PYTHON_BIN=/home/sherryli/xsli/softwares/anaconda3/envs/lightning/bin/python
export DATA_H5=/mnt/ssd/mcmc/lj_mcmc_sweep_3d/lj3d_L3_rho1.0_N27_T1.0.h5
export CACHE_PATH=/mnt/ssd/GridTransformer/lj_ckpts_lj27_pbc_arc_repr/lj27_pbc_arc_repr_fullcov/lj27_pbc_hilbert_cache.pt
export USE_CONTINUOUS_HEAD=1 CONTINUOUS_INPUT=1 NUM_MIXTURES=64 FULL_COVARIANCE=1
export USE_CURVE_RAIL=0 PREPROCESS_RANDOM_SHIFT=0 NUM_AUGMENTATIONS=1
export LAMBDA_VAR=0.0 LJ_KT=1.0 LJ_EPSILON=1.0 LJ_SIGMA=1.0 LJ_SPRING_CONSTANT=0.5
export PBC=1 SEED=0 ORDERING=hilbert HILBERT_RESOLUTION=64 WINDOW=3.0 BINS=64
export BATCH_SIZE="${PROBE_BATCH_SIZE:-2048}" LR=1e-3 NUM_WORKERS=8
export MODEL_DIM=512 MODEL_HEADS=4 MODEL_DEPTH=4 AR_ARCH=standard
export BOX_LENGTH=3.0 NUM_PARTICLES=27 ARC_REPR=1
export EPOCHS="${PROBE_EPOCHS}"
export WARM_START_CKPT="${BASELINE_CKPT}"
export RUN_PREPROCESS=0 RUN_SAMPLE=0   # reuse the baseline cache; sampling is Stage 2
unset RESUME_FROM BEST_CKPT SAMPLE_OUT TRAIN_LOG TRAIN_PID_FILE RUN_DIR 2>/dev/null || true

run_probe() {
  local name="$1"; shift
  echo "=== probe ${name}: $* ==="
  (
    export CKPT_ROOT="${PROBE_ROOT}"
    export RUN_SUBDIR="probe_${name}"
    export "$@"
    unset RUN_DIR BEST_CKPT SAMPLE_OUT TRAIN_LOG TRAIN_PID_FILE 2>/dev/null || true
    bash /mnt/ssd/GridTransformer/train_sample_lj27_pbc.sh
  )
}

for p in ${PROBES}; do
  case "${p}" in
    A) run_probe A USE_JOINT_ARC_BIAS=1 ;;
    F) run_probe F USE_REFINE_RBF_BIAS=1 ;;
    G) run_probe G KNN_MASK_K=12 ;;
    *) echo "unknown probe '${p}'" >&2; exit 1 ;;
  esac
done

echo "All probes done. Compare train/nll_jump vs baseline in ${PROBE_ROOT}/probe_*/lightning_logs."
