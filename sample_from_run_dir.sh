#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: bash sample_from_run_dir.sh RUN_DIR

Replays sampling for an existing run directory by invoking its reproduce.sh
with training and preprocessing disabled.

Environment overrides:
  RUN_PREPROCESS=0          Disabled by default
  RUN_TRAIN=0               Disabled by default
  RUN_SAMPLE=1              Enabled by default
  SAMPLE_NSAMPLES=...       Override number of samples
  SAMPLE_BATCH_SIZE=...     Override sampling batch size
  SAMPLE_SAVE_EACH_BATCH=1  Save cumulative output after each sampling batch
  SAMPLE_MODE=...           Override sampling mode
  TEMPERATURE=...           Override temperature
  TOP_K=...                 Override top-k
  SAMPLE_OUT=...            Override output .npz path
  BEST_CKPT=...             Override checkpoint path
  PYTHON_BIN=...            Override Python executable
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ "$#" -ne 1 ]]; then
  usage >&2
  exit 1
fi

RUN_DIR_INPUT="$1"
if [[ ! -d "${RUN_DIR_INPUT}" ]]; then
  echo "Run directory not found: ${RUN_DIR_INPUT}" >&2
  exit 1
fi

RUN_DIR="$(cd "${RUN_DIR_INPUT}" && pwd)"
REPRODUCE_SH="${RUN_DIR}/reproduce.sh"

if [[ ! -f "${REPRODUCE_SH}" ]]; then
  echo "Missing reproduce script: ${REPRODUCE_SH}" >&2
  exit 1
fi

: "${RUN_PREPROCESS:=0}"
: "${RUN_TRAIN:=0}"
: "${RUN_SAMPLE:=1}"
: "${RUN_DIR:=${RUN_DIR}}"

declare -A USER_OVERRIDES=()
OVERRIDE_VARS=(
  PYTHON_BIN
  RUN_PREPROCESS
  RUN_TRAIN
  RUN_SAMPLE
  RUN_DIR
  RUN_SUBDIR
  CKPT_ROOT
  RESUME_FROM
  BEST_CKPT
  SAMPLE_OUT
  SAMPLE_NSAMPLES
  SAMPLE_BATCH_SIZE
  SAMPLE_SAVE_EACH_BATCH
  SAMPLE_MODE
  TEMPERATURE
  TOP_K
)

for var_name in "${OVERRIDE_VARS[@]}"; do
  if [[ ${!var_name+x} ]]; then
    USER_OVERRIDES["${var_name}"]="${!var_name}"
  fi
done

TARGET_SCRIPT="$(sed -n '$s/^bash //p' "${REPRODUCE_SH}")"
if [[ -z "${TARGET_SCRIPT}" ]]; then
  echo "Could not determine target script from: ${REPRODUCE_SH}" >&2
  exit 1
fi

set -a
source <(sed '$d' "${REPRODUCE_SH}")
set +a

for var_name in "${!USER_OVERRIDES[@]}"; do
  export "${var_name}=${USER_OVERRIDES[${var_name}]}"
done

echo "[sample_from_run_dir] run_dir=${RUN_DIR}"
echo "[sample_from_run_dir] reproduce_sh=${REPRODUCE_SH}"
echo "[sample_from_run_dir] target_script=${TARGET_SCRIPT}"
echo "[sample_from_run_dir] RUN_PREPROCESS=${RUN_PREPROCESS} RUN_TRAIN=${RUN_TRAIN} RUN_SAMPLE=${RUN_SAMPLE}"
if [[ -n "${SAMPLE_OUT:-}" ]]; then
  echo "[sample_from_run_dir] SAMPLE_OUT=${SAMPLE_OUT}"
fi

bash "${TARGET_SCRIPT}"
