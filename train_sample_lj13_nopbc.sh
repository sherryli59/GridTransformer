#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
PREPROCESS_SCRIPT="${ROOT_DIR}/scripts/preprocess_lj_transferable.py"

DATA_H5="${DATA_H5:-/mnt/ssd/mcmc/lj_mcmc_sweep/lj_rho0.5_nopbc_N13_T1.0.h5}"
DATE="$(date +%m%d)"
CKPT_ROOT="${CKPT_ROOT:-${CKPT_DIR:-${ROOT_DIR}/lj_ckpts_lj13_nopbc_${DATE}}}"
RUN_SUBDIR="${RUN_SUBDIR:-}"
RUN_DIR="${RUN_DIR:-}"
RESUME_FROM="${RESUME_FROM:-}"

RUN_PREPROCESS="${RUN_PREPROCESS:-1}"
RUN_TRAIN="${RUN_TRAIN:-1}"
RUN_SAMPLE="${RUN_SAMPLE:-1}"
TRAIN_DATA_AUG="${TRAIN_DATA_AUG:-0}"
FACTORIZED="${FACTORIZED:-0}"
USE_CONTINUOUS_HEAD="${USE_CONTINUOUS_HEAD:-1}"
NUM_MIXTURES="${NUM_MIXTURES:-64}"

SEED="${SEED:-0}"
ORDERING="${ORDERING:-hilbert}"
HILBERT_RESOLUTION="${HILBERT_RESOLUTION:-64}"
WINDOW="${WINDOW:-3.0}"
BINS="${BINS:-64}"
USE_COORD_DEQUANT="${USE_COORD_DEQUANT:-0}"

# Exact bin width for optional coordinate dequantization noise (2 * W / B).
DEQUANT_WIDTH=$(awk "BEGIN {print 2.0 * ${WINDOW} / ${BINS}}")
if [[ "${USE_CONTINUOUS_HEAD}" == "1" ]]; then
  echo "[train] continuous head enabled; skipping coord_dequant_width"
elif [[ "${USE_COORD_DEQUANT}" == "1" ]]; then
  echo "[train] discrete head dequant width: ${DEQUANT_WIDTH}"
fi

EPOCHS="${EPOCHS:-400}"
BATCH_SIZE="${BATCH_SIZE:-512}"
LR="${LR:-1e-3}"
NUM_WORKERS="${NUM_WORKERS:-0}"
MODEL_DIM="${MODEL_DIM:-512}"
MODEL_HEADS="${MODEL_HEADS:-4}"
MODEL_DEPTH="${MODEL_DEPTH:-4}"
AR_ARCH="${AR_ARCH:-standard}"

SAMPLE_NSAMPLES="${SAMPLE_NSAMPLES:-1024}"
SAMPLE_BATCH_SIZE="${SAMPLE_BATCH_SIZE:-512}"
SAMPLE_MODE="${SAMPLE_MODE:-multinomial}"
TEMPERATURE="${TEMPERATURE:-0.9}"
TOP_K="${TOP_K:-}"

if [[ ! -f "${DATA_H5}" ]]; then
  echo "Missing data file: ${DATA_H5}" >&2
  exit 1
fi
if [[ ! -f "${PREPROCESS_SCRIPT}" ]]; then
  echo "Missing preprocess script: ${PREPROCESS_SCRIPT}" >&2
  exit 1
fi
if [[ -n "${RESUME_FROM}" ]]; then
  if [[ ! -f "${RESUME_FROM}" ]]; then
    echo "Missing resume checkpoint: ${RESUME_FROM}" >&2
    exit 1
  fi
  if [[ -z "${RUN_DIR}" ]]; then
    RUN_DIR="$(cd "$(dirname "${RESUME_FROM}")" && pwd)"
  fi
fi

readarray -t DATA_INFO < <("${PYTHON_BIN}" - "${DATA_H5}" <<'PY'
import h5py
import sys

path = sys.argv[1]
with h5py.File(path, "r") as h5f:
    box_attr = h5f.attrs["boxlength"]
    if hasattr(box_attr, "__len__") and len(box_attr) > 0:
        box_vals = [float(x) for x in box_attr]
    else:
        box_vals = [float(box_attr)]
    density = float(h5f.attrs.get("density", 0.0))
    traj = h5f["traj"]
    n_particles = int(traj.shape[-2])
    dim = int(traj.shape[-1])
    if len(box_vals) == 1:
        box_vals = box_vals * dim
    elif len(box_vals) < dim:
        box_vals = box_vals + [box_vals[-1]] * (dim - len(box_vals))

print(box_vals[0])
print(box_vals[1])
print(box_vals[2])
print(density)
print(n_particles)
print(dim)
PY
)

BOX_LENGTH="${BOX_LENGTH:-${DATA_INFO[0]}}"
BOX_LENGTH_Y="${BOX_LENGTH_Y:-${DATA_INFO[1]}}"
BOX_LENGTH_Z="${BOX_LENGTH_Z:-${DATA_INFO[2]}}"
DENSITY="${DENSITY:-${DATA_INFO[3]}}"
NUM_PARTICLES="${NUM_PARTICLES:-${DATA_INFO[4]}}"
DIM="${DIM:-${DATA_INFO[5]}}"

if [[ "${DIM}" != "3" ]]; then
  echo "Expected fully 3D data, but traj dim=${DIM} in ${DATA_H5}" >&2
  exit 1
fi

HEAD_TAG="discrete"
if [[ "${USE_CONTINUOUS_HEAD}" == "1" ]]; then
  HEAD_TAG="continuous"
fi

if [[ -z "${RUN_SUBDIR}" ]]; then
  if [[ -n "${RUN_DIR}" ]]; then
    RUN_SUBDIR="$(basename "${RUN_DIR}")"
  else
    CONFIG_ID="$(
      printf '%s\n' \
        "${DATA_H5}" \
        "${NUM_PARTICLES}" \
        "${DIM}" \
        "${ORDERING}" \
        "${HILBERT_RESOLUTION}" \
        "${WINDOW}" \
        "${BINS}" \
        "${USE_COORD_DEQUANT}" \
        "${TRAIN_DATA_AUG}" \
        "${FACTORIZED}" \
        "${USE_CONTINUOUS_HEAD}" \
        "${NUM_MIXTURES}" \
        "${SEED}" \
        "${EPOCHS}" \
        "${BATCH_SIZE}" \
        "${LR}" \
        "${NUM_WORKERS}" \
        "${MODEL_DIM}" \
        "${MODEL_HEADS}" \
        "${MODEL_DEPTH}" \
        "${AR_ARCH}" \
        "${SAMPLE_NSAMPLES}" \
        "${SAMPLE_BATCH_SIZE}" \
        "${SAMPLE_MODE}" \
        "${TEMPERATURE}" \
        "${TOP_K}" \
        | sha1sum | awk '{print substr($1, 1, 12)}'
    )"
    RUN_SUBDIR="N${NUM_PARTICLES}_${HEAD_TAG}_${CONFIG_ID}"
  fi
fi

if [[ -z "${RUN_DIR}" ]]; then
  RUN_DIR="${CKPT_ROOT}/${RUN_SUBDIR}"
fi

CKPT_DIR="${RUN_DIR}"
CACHE_PATH="${CACHE_PATH:-${CKPT_DIR}/lj13_nopbc_hilbert_cache.pt}"
BEST_CKPT="${BEST_CKPT:-${CKPT_DIR}/best.ckpt}"
SAMPLE_OUT="${SAMPLE_OUT:-${CKPT_DIR}/samples_lj13_nopbc.npz}"
TRAIN_LOG="${TRAIN_LOG:-${CKPT_DIR}/train_relative.log}"
TRAIN_PID_FILE="${TRAIN_PID_FILE:-${CKPT_DIR}/train_relative.pid}"

mkdir -p "${CKPT_ROOT}" "${CKPT_DIR}"

TOP_K_ARGS=()
if [[ -n "${TOP_K}" ]]; then
  TOP_K_ARGS=(--top_k "${TOP_K}")
fi

write_reproduce_script() {
  local reproduce_path="${CKPT_DIR}/reproduce.sh"
  local vars=(
    PYTHON_BIN
    DATA_H5
    CKPT_ROOT
    RUN_SUBDIR
    RUN_DIR
    RESUME_FROM
    CACHE_PATH
    BEST_CKPT
    SAMPLE_OUT
    TRAIN_LOG
    TRAIN_PID_FILE
    TRAIN_DATA_AUG
    FACTORIZED
    USE_CONTINUOUS_HEAD
    NUM_MIXTURES
    SEED
    ORDERING
    HILBERT_RESOLUTION
    WINDOW
    BINS
    USE_COORD_DEQUANT
    EPOCHS
    BATCH_SIZE
    LR
    NUM_WORKERS
    MODEL_DIM
    MODEL_HEADS
    MODEL_DEPTH
    AR_ARCH
    SAMPLE_NSAMPLES
    SAMPLE_BATCH_SIZE
    SAMPLE_MODE
    TEMPERATURE
    TOP_K
    BOX_LENGTH
    BOX_LENGTH_Y
    BOX_LENGTH_Z
    DENSITY
    NUM_PARTICLES
    DIM
  )

  {
    printf '#!/usr/bin/env bash\n'
    printf 'set -euo pipefail\n\n'
    printf 'cd %q\n\n' "${ROOT_DIR}"
    for var_name in "${vars[@]}"; do
      printf 'export %s=%q\n' "${var_name}" "${!var_name}"
    done
    printf '\nbash %q\n' "${ROOT_DIR}/train_sample_lj13_nopbc.sh"
  } > "${reproduce_path}"
  chmod +x "${reproduce_path}"
  echo "[run] wrote reproduce script: ${reproduce_path}"
}

echo "[run] ckpt_root=${CKPT_ROOT}"
echo "[run] run_dir=${CKPT_DIR}"
write_reproduce_script

report_cache_stats() {
  "${PYTHON_BIN}" - "${CACHE_PATH}" <<'PY'
import sys
import torch

cache_path = sys.argv[1]
try:
    payload = torch.load(cache_path, map_location="cpu", weights_only=False)
except TypeError:
    payload = torch.load(cache_path, map_location="cpu")
metadata = dict(payload.get("metadata", {}))
sample_length = payload.get("sample_length")
n_samples = int(sample_length.shape[0]) if sample_length is not None else -1
print(
    "[cache] "
    f"path={cache_path}, "
    f"n_samples={n_samples}, "
    f"max_abs_relative_displacement={metadata.get('max_abs_relative_displacement', 'unknown')}"
)
PY
}

if [[ "${RUN_PREPROCESS}" == "1" ]]; then
  if [[ -f "${CACHE_PATH}" ]]; then
    echo "[preprocess] cache already exists, skipping: ${CACHE_PATH}"
  else
    echo "[preprocess] ${DATA_H5} -> ${CACHE_PATH}"
    PREPROCESS_ARGS=(
      --data_h5 "${DATA_H5}"
      --output "${CACHE_PATH}"
      --no-periodic
      --ordering "${ORDERING}"
      --hilbert_resolution "${HILBERT_RESOLUTION}"
      --window "${WINDOW}"
      --bins "${BINS}"
      --seed "${SEED}"
    )
    if [[ "${FACTORIZED}" == "1" ]]; then
      PREPROCESS_ARGS+=(--factorized)
    fi
    "${PYTHON_BIN}" "${PREPROCESS_SCRIPT}" \
      "${PREPROCESS_ARGS[@]}"
  fi
  report_cache_stats
fi

if [[ "${RUN_TRAIN}" == "1" ]]; then
  echo "[train] dataset=${DATA_H5} ckpt_dir=${CKPT_DIR}"
  if [[ -n "${RESUME_FROM}" ]]; then
    echo "[train] resuming from ckpt=${RESUME_FROM}"
  fi
  TRAIN_ARGS=(
    --dataset lj_transferable
    --data_dir "${DATA_H5}"
    --no-lj_transfer_periodic
    --lj_transfer_ordering "${ORDERING}"
    --lj_transfer_hilbert_resolution "${HILBERT_RESOLUTION}"
    --lj_transfer_window "${WINDOW}"
    --lj_transfer_bins "${BINS}"
    --lj_transfer_disable_random_shift
    --epochs "${EPOCHS}"
    --batch_size "${BATCH_SIZE}"
    --lr "${LR}"
    --num_workers "${NUM_WORKERS}"
    --seed "${SEED}"
    --dim "${MODEL_DIM}"
    --heads "${MODEL_HEADS}"
    --depth "${MODEL_DEPTH}"
    --ckpt_dir "${CKPT_DIR}"
    --ar_arch "${AR_ARCH}"
  )
  if [[ "${TRAIN_DATA_AUG}" == "1" ]]; then
    echo "[train] using on-the-fly right-angle augmentation: random 90-degree rotations about each axis, then Hilbert-sorted relative-displacement tokenization per sample"
    TRAIN_ARGS+=(--lj_transfer_use_data_aug)
  else
    if [[ ! -f "${CACHE_PATH}" ]]; then
      echo "Missing cache for training: ${CACHE_PATH}. Run with RUN_PREPROCESS=1 or set TRAIN_DATA_AUG=1." >&2
      exit 1
    fi
    TRAIN_ARGS+=(--lj_transfer_preprocessed_path "${CACHE_PATH}")
  fi
  if [[ "${FACTORIZED}" == "1" ]]; then
    TRAIN_ARGS+=(--lj_transfer_factorized)
  fi
  if [[ "${USE_CONTINUOUS_HEAD}" == "1" ]]; then
    if [[ "${FACTORIZED}" == "1" ]]; then
      echo "USE_CONTINUOUS_HEAD=1 currently requires FACTORIZED=0." >&2
      exit 1
    fi
    echo "[train] enabling continuous MDN head with num_mixtures=${NUM_MIXTURES}"
    TRAIN_ARGS+=(--use_continuous_head --num_mixtures "${NUM_MIXTURES}")
  elif [[ "${USE_COORD_DEQUANT}" == "1" ]]; then
    TRAIN_ARGS+=(--coord_dequant_width "${DEQUANT_WIDTH}")
  fi
  if [[ -n "${RESUME_FROM}" ]]; then
    TRAIN_ARGS+=(--resume_from "${RESUME_FROM}")
  fi
  echo "[train] nohup log=${TRAIN_LOG} pid_file=${TRAIN_PID_FILE}"
  nohup "${PYTHON_BIN}" "${ROOT_DIR}/train.py" "${TRAIN_ARGS[@]}" >"${TRAIN_LOG}" 2>&1 &
  TRAIN_PID=$!
  echo "${TRAIN_PID}" > "${TRAIN_PID_FILE}"
  echo "[train] waiting for pid=${TRAIN_PID}"
  if ! wait "${TRAIN_PID}"; then
    echo "Training failed for pid=${TRAIN_PID}. Check log: ${TRAIN_LOG}" >&2
    exit 1
  fi
fi

if [[ "${RUN_SAMPLE}" == "1" ]]; then
  if [[ ! -f "${BEST_CKPT}" ]]; then
    echo "Missing checkpoint for sampling: ${BEST_CKPT}" >&2
    exit 1
  fi

  echo "[sample] ckpt=${BEST_CKPT} out=${SAMPLE_OUT}"
  SAMPLE_ARGS=(
    --mode relative
    --ckpt "${BEST_CKPT}"
    --Lx "${BOX_LENGTH}"
    --Ly "${BOX_LENGTH_Y}"
    --Lz "${BOX_LENGTH_Z}"
    --coord_dim "${DIM}"
    --num_particles "${NUM_PARTICLES}"
    --sample_mode "${SAMPLE_MODE}"
    --temperature "${TEMPERATURE}"
    --relative_window "${WINDOW}"
    --relative_bins "${BINS}"
    --nsamples "${SAMPLE_NSAMPLES}"
    --sample_batch_size "${SAMPLE_BATCH_SIZE}"
    --no-periodic
    --ar_arch "${AR_ARCH}"
    --save "${SAMPLE_OUT}"
  )
  if [[ "${FACTORIZED}" == "1" ]]; then
    SAMPLE_ARGS+=(--factorized)
  fi
  if [[ "${USE_CONTINUOUS_HEAD}" == "1" ]]; then
    SAMPLE_ARGS+=(--use_continuous_head)
  fi
  "${PYTHON_BIN}" "${ROOT_DIR}/sample_lj.py" \
    "${SAMPLE_ARGS[@]}" \
    "${TOP_K_ARGS[@]}"
fi
