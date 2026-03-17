#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

DATA_H5="${DATA_H5:-/mnt/ssd/mcmc/lj_mcmc_sweep/lj_N13_rho0.5_nopbc_N13_T1.0.h5}"
CKPT_DIR="${CKPT_DIR:-${ROOT_DIR}/lj_ckpts_lj13_nopbc}"
CACHE_PATH="${CACHE_PATH:-${CKPT_DIR}/lj13_nopbc_hilbert_cache.pt}"
BEST_CKPT="${BEST_CKPT:-${CKPT_DIR}/best.ckpt}"
SAMPLE_OUT="${SAMPLE_OUT:-${CKPT_DIR}/samples_lj13_nopbc.npz}"

RUN_PREPROCESS="${RUN_PREPROCESS:-1}"
RUN_TRAIN="${RUN_TRAIN:-1}"
RUN_SAMPLE="${RUN_SAMPLE:-1}"
TRAIN_DATA_AUG="${TRAIN_DATA_AUG:-1}"
FACTORIZED="${FACTORIZED:-0}"

SEED="${SEED:-0}"
ORDERING="${ORDERING:-hilbert}"
HILBERT_RESOLUTION="${HILBERT_RESOLUTION:-128}"
WINDOW="${WINDOW:-3.0}"
BINS="${BINS:-64}"

EPOCHS="${EPOCHS:-10}"
BATCH_SIZE="${BATCH_SIZE:-256}"
LR="${LR:-1e-3}"
NUM_WORKERS="${NUM_WORKERS:-0}"
MODEL_DIM="${MODEL_DIM:-128}"
MODEL_HEADS="${MODEL_HEADS:-4}"
MODEL_DEPTH="${MODEL_DEPTH:-3}"
AR_ARCH="${AR_ARCH:-ida}"

SAMPLE_NSAMPLES="${SAMPLE_NSAMPLES:-1024}"
SAMPLE_BATCH_SIZE="${SAMPLE_BATCH_SIZE:-512}"
SAMPLE_MODE="${SAMPLE_MODE:-multinomial}"
TEMPERATURE="${TEMPERATURE:-0.7}"
TOP_K="${TOP_K:-}"

if [[ ! -f "${DATA_H5}" ]]; then
  echo "Missing data file: ${DATA_H5}" >&2
  exit 1
fi

mkdir -p "${CKPT_DIR}"

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

TOP_K_ARGS=()
if [[ -n "${TOP_K}" ]]; then
  TOP_K_ARGS=(--top_k "${TOP_K}")
fi

if [[ "${RUN_PREPROCESS}" == "1" && "${TRAIN_DATA_AUG}" != "1" ]]; then
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
  "${PYTHON_BIN}" "${ROOT_DIR}/scripts/preprocess_lj_transferable.py" \
    "${PREPROCESS_ARGS[@]}"
elif [[ "${RUN_PREPROCESS}" == "1" && "${TRAIN_DATA_AUG}" == "1" ]]; then
  echo "[preprocess] skipped because TRAIN_DATA_AUG=1 requires on-the-fly tokenization"
fi

if [[ "${RUN_TRAIN}" == "1" ]]; then
  echo "[train] dataset=${DATA_H5} ckpt_dir=${CKPT_DIR}"
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
    TRAIN_ARGS+=(--lj_transfer_use_data_aug)
  else
    TRAIN_ARGS+=(--lj_transfer_preprocessed_path "${CACHE_PATH}")
  fi
  if [[ "${FACTORIZED}" == "1" ]]; then
    TRAIN_ARGS+=(--lj_transfer_factorized)
  fi
  "${PYTHON_BIN}" "${ROOT_DIR}/train.py" "${TRAIN_ARGS[@]}"
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
  "${PYTHON_BIN}" "${ROOT_DIR}/sample_lj.py" \
    "${SAMPLE_ARGS[@]}" \
    "${TOP_K_ARGS[@]}"
fi
