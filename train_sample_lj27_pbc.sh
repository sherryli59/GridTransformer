#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
PREPROCESS_SCRIPT="${ROOT_DIR}/scripts/preprocess_lj_transferable.py"
CODEBOOK_SCRIPT="${ROOT_DIR}/scripts/generate_codebook.py"

DATA_H5="${DATA_H5:-/mnt/ssd/mcmc/lj_mcmc_sweep_3d/lj3d_L3_rho1.0_N27_T1.0.h5}"
DATE="$(date +%m%d)"
CKPT_ROOT="${CKPT_ROOT:-${CKPT_DIR:-${ROOT_DIR}/lj_ckpts_lj27_pbc_${DATE}}}"
RUN_SUBDIR="${RUN_SUBDIR:-}"
RUN_DIR="${RUN_DIR:-}"
RESUME_FROM="${RESUME_FROM:-}"

RUN_PREPROCESS="${RUN_PREPROCESS:-1}"
RUN_TRAIN="${RUN_TRAIN:-1}"
RUN_SAMPLE="${RUN_SAMPLE:-1}"
TRAIN_DATA_AUG="${TRAIN_DATA_AUG:-0}"
NUM_AUGMENTATIONS_WAS_SET="${NUM_AUGMENTATIONS+x}"
FACTORIZED="${FACTORIZED:-0}"
POLAR="${POLAR:-0}"
USE_CONTINUOUS_HEAD="${USE_CONTINUOUS_HEAD:-1}"
CONTINUOUS_INPUT="${CONTINUOUS_INPUT:-0}"
DISCRETE="${DISCRETE:-0}"
BINNED_DISCRETE="${BINNED_DISCRETE:-0}"
CODEBOOK_PATH="${CODEBOOK_PATH:-${ROOT_DIR}/codebook.pt}"
CODEBOOK_SIZE="${CODEBOOK_SIZE:-4096}"
NUM_MIXTURES="${NUM_MIXTURES:-64}"
FULL_COVARIANCE="${FULL_COVARIANCE:-1}"
USE_CURVE_RAIL="${USE_CURVE_RAIL:-0}"
CURVE_RAIL_OFFSETS="${CURVE_RAIL_OFFSETS:-}"
CURVE_RAIL_MODE="${CURVE_RAIL_MODE:-fixed_template}"
CURVE_RAIL_WINDOW="${CURVE_RAIL_WINDOW:-1.0}"
CURVE_RAIL_K="${CURVE_RAIL_K:-8}"
CURVE_RAIL_REFERENCE="${CURVE_RAIL_REFERENCE:-absolute}"
CURVE_RAIL_RESIDUAL_TARGET="${CURVE_RAIL_RESIDUAL_TARGET:-0}"
ARC_REPR="${ARC_REPR:-0}"
PREPROCESS_RANDOM_SHIFT="${PREPROCESS_RANDOM_SHIFT:-0}"
NUM_AUGMENTATIONS="${NUM_AUGMENTATIONS:-5}"
if [[ "${USE_CURVE_RAIL}" == "1" ]] && [[ -z "${NUM_AUGMENTATIONS_WAS_SET}" ]]; then
  NUM_AUGMENTATIONS=1
fi
ENERGY_CHUNK_SIZE="${ENERGY_CHUNK_SIZE:-2048}"
CACHE_BUILD_CHUNK_SIZE="${CACHE_BUILD_CHUNK_SIZE:-16384}"
LAMBDA_VAR="${LAMBDA_VAR:-0.0}"
LJ_KT="${LJ_KT:-1.0}"
LJ_EPSILON="${LJ_EPSILON:-1.0}"
LJ_SIGMA="${LJ_SIGMA:-1.0}"
LJ_CUTOFF="${LJ_CUTOFF:-}"
LJ_SPRING_CONSTANT="${LJ_SPRING_CONSTANT:-0.5}"
PBC="${PBC:-1}"

SEED="${SEED:-0}"
ORDERING="${ORDERING:-hilbert}"
HILBERT_RESOLUTION="${HILBERT_RESOLUTION:-64}"
WINDOW="${WINDOW:-3.0}"   # token/displacement window W (sigma) for the AR tokenizer
SWIN_WINDOW="${SWIN_WINDOW:-8}"   # Swin attention window size (cells, int); distinct from WINDOW
BINS="${BINS:-64}"
USE_COORD_DEQUANT="${USE_COORD_DEQUANT:-0}"
AR_ARCH="${AR_ARCH:-standard}"

# Exact bin width for optional coordinate dequantization noise (2 * W / B).
DEQUANT_WIDTH=$(awk "BEGIN {print 2.0 * ${WINDOW} / ${BINS}}")
if [[ "${DISCRETE}" == "1" ]]; then
  FACTORIZED=0
  USE_CONTINUOUS_HEAD=0
  FULL_COVARIANCE=0
fi
if [[ "${BINNED_DISCRETE}" == "1" ]]; then
  USE_CONTINUOUS_HEAD=0
  FULL_COVARIANCE=0
  CONTINUOUS_INPUT=0
fi
if [[ "${USE_CONTINUOUS_HEAD}" == "1" ]]; then
  echo "[train] continuous head enabled; skipping coord_dequant_width"
elif [[ "${USE_COORD_DEQUANT}" == "1" ]]; then
  echo "[train] discrete head dequant width: ${DEQUANT_WIDTH}"
fi

if [[ "${POLAR}" == "1" ]] && [[ "${FACTORIZED}" == "1" ]]; then
  echo "POLAR=1 with FACTORIZED=1 is not supported in this checkout." >&2
  exit 1
fi
if [[ "${DISCRETE}" == "1" ]] && [[ "${POLAR}" == "1" ]]; then
  echo "DISCRETE=1 is incompatible with POLAR=1." >&2
  exit 1
fi
if [[ "${BINNED_DISCRETE}" == "1" ]] && [[ "${POLAR}" == "1" ]]; then
  echo "BINNED_DISCRETE=1 is incompatible with POLAR=1." >&2
  exit 1
fi
if [[ "${BINNED_DISCRETE}" == "1" ]] && [[ "${DISCRETE}" == "1" ]]; then
  echo "BINNED_DISCRETE=1 is incompatible with DISCRETE=1." >&2
  exit 1
fi
if [[ "${FULL_COVARIANCE}" == "1" ]] && [[ "${USE_CONTINUOUS_HEAD}" != "1" ]]; then
  echo "FULL_COVARIANCE=1 requires USE_CONTINUOUS_HEAD=1." >&2
  exit 1
fi
if [[ "${CONTINUOUS_INPUT}" == "1" ]] && [[ "${USE_CONTINUOUS_HEAD}" != "1" ]]; then
  echo "CONTINUOUS_INPUT=1 requires USE_CONTINUOUS_HEAD=1." >&2
  exit 1
fi
if [[ "${CONTINUOUS_INPUT}" == "1" ]] && [[ "${FACTORIZED}" == "1" ]]; then
  echo "CONTINUOUS_INPUT=1 is incompatible with FACTORIZED=1." >&2
  exit 1
fi
if [[ "${CONTINUOUS_INPUT}" == "1" ]] && [[ "${POLAR}" == "1" ]]; then
  echo "CONTINUOUS_INPUT=1 is incompatible with POLAR=1." >&2
  exit 1
fi

EPOCHS="${EPOCHS:-400}"
BATCH_SIZE="${BATCH_SIZE:-512}"
LR="${LR:-1e-3}"
NUM_WORKERS="${NUM_WORKERS:-0}"
MODEL_DIM="${MODEL_DIM:-512}"
MODEL_HEADS="${MODEL_HEADS:-4}"
MODEL_DEPTH="${MODEL_DEPTH:-4}"

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
if [[ "${DISCRETE}" == "1" ]] && [[ ! -f "${CODEBOOK_SCRIPT}" ]]; then
  echo "Missing codebook script for discrete mode: ${CODEBOOK_SCRIPT}" >&2
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
    if len(box_vals) < 3:
        box_vals = box_vals + [box_vals[-1]] * (3 - len(box_vals))

print(box_vals[0])
print(box_vals[1])
print(box_vals[2])
print(density)
print(n_particles)
print(dim)
PY
)

if [[ "${#DATA_INFO[@]}" -lt 6 ]]; then
  echo "Failed to read expected dataset metadata from ${DATA_H5}. Got ${#DATA_INFO[@]} fields." >&2
  printf 'DATA_INFO=%q\n' "${DATA_INFO[@]}" >&2
  exit 1
fi

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
if [[ "${PBC}" != "1" ]]; then
  echo "This script is for periodic 3D data. Set PBC=1." >&2
  exit 1
fi
if [[ "${USE_CURVE_RAIL}" == "1" ]]; then
  if [[ "${ORDERING}" != "hilbert" ]]; then
    echo "USE_CURVE_RAIL=1 requires ORDERING=hilbert." >&2
    exit 1
  fi
  if [[ "${AR_ARCH}" != "standard" ]]; then
    echo "USE_CURVE_RAIL=1 requires AR_ARCH=standard." >&2
    exit 1
  fi
  if [[ "${CURVE_RAIL_MODE}" != "lookahead" ]] && [[ "${CURVE_RAIL_MODE}" != "fixed_template" ]]; then
    echo "CURVE_RAIL_MODE must be lookahead or fixed_template, got ${CURVE_RAIL_MODE}." >&2
    exit 1
  fi
  if [[ "${FACTORIZED}" == "1" ]] && [[ "${CURVE_RAIL_MODE}" != "fixed_template" ]]; then
    echo "USE_CURVE_RAIL=1 with FACTORIZED=1 requires CURVE_RAIL_MODE=fixed_template." >&2
    exit 1
  fi
fi

HEAD_TAG="discrete"
if [[ "${USE_CONTINUOUS_HEAD}" == "1" ]]; then
  HEAD_TAG="continuous"
fi
if [[ "${DISCRETE}" == "1" ]]; then
  HEAD_TAG="codebook"
fi
if [[ "${BINNED_DISCRETE}" == "1" ]]; then
  HEAD_TAG="binned_discrete"
fi
if [[ "${POLAR}" == "1" ]]; then
  HEAD_TAG="${HEAD_TAG}_polar"
fi
if [[ "${FACTORIZED}" == "1" ]]; then
  HEAD_TAG="${HEAD_TAG}_factorized"
fi
if [[ "${CONTINUOUS_INPUT}" == "1" ]]; then
  HEAD_TAG="${HEAD_TAG}_continput"
fi
if [[ "${FULL_COVARIANCE}" == "1" ]]; then
  HEAD_TAG="${HEAD_TAG}_fullcov"
fi
if [[ "${USE_CURVE_RAIL}" == "1" ]]; then
  HEAD_TAG="${HEAD_TAG}_rail_${CURVE_RAIL_MODE}"
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
        "${POLAR}" \
        "${USE_CONTINUOUS_HEAD}" \
        "${CONTINUOUS_INPUT}" \
        "${DISCRETE}" \
        "${BINNED_DISCRETE}" \
        "${CODEBOOK_PATH}" \
        "${CODEBOOK_SIZE}" \
        "${NUM_MIXTURES}" \
        "${FULL_COVARIANCE}" \
        "${USE_CURVE_RAIL}" \
        "${CURVE_RAIL_OFFSETS}" \
        "${CURVE_RAIL_MODE}" \
        "${CURVE_RAIL_WINDOW}" \
        "${CURVE_RAIL_K}" \
        "${CURVE_RAIL_REFERENCE}" \
        "${CURVE_RAIL_RESIDUAL_TARGET}" \
        "${PREPROCESS_RANDOM_SHIFT}" \
        "${NUM_AUGMENTATIONS}" \
        "${ENERGY_CHUNK_SIZE}" \
        "${CACHE_BUILD_CHUNK_SIZE}" \
        "${LAMBDA_VAR}" \
        "${LJ_KT}" \
        "${LJ_EPSILON}" \
        "${LJ_SIGMA}" \
        "${LJ_CUTOFF}" \
        "${LJ_SPRING_CONSTANT}" \
        "${PBC}" \
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
CACHE_PATH="${CACHE_PATH:-${CKPT_DIR}/lj27_pbc_hilbert_cache.pt}"
DISCRETE_CACHE_PATH="${DISCRETE_CACHE_PATH:-${CKPT_DIR}/lj27_pbc_hilbert_cache_discrete.pt}"
CODEBOOK_SOURCE_CACHE="${CODEBOOK_SOURCE_CACHE:-${CKPT_DIR}/codebook_source_cache.pt}"
BEST_CKPT="${BEST_CKPT:-${CKPT_DIR}/best.ckpt}"
SAMPLE_OUT="${SAMPLE_OUT:-${CKPT_DIR}/samples_lj27_pbc.npz}"
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
    DISCRETE_CACHE_PATH
    CODEBOOK_SOURCE_CACHE
    BEST_CKPT
    SAMPLE_OUT
    TRAIN_LOG
    TRAIN_PID_FILE
    TRAIN_DATA_AUG
    FACTORIZED
    POLAR
    USE_CONTINUOUS_HEAD
    CONTINUOUS_INPUT
    DISCRETE
    BINNED_DISCRETE
    CODEBOOK_PATH
    CODEBOOK_SIZE
    NUM_MIXTURES
    FULL_COVARIANCE
    USE_CURVE_RAIL
    CURVE_RAIL_OFFSETS
    CURVE_RAIL_MODE
    CURVE_RAIL_WINDOW
    CURVE_RAIL_K
    CURVE_RAIL_REFERENCE
    CURVE_RAIL_RESIDUAL_TARGET
    PREPROCESS_RANDOM_SHIFT
    NUM_AUGMENTATIONS
    ENERGY_CHUNK_SIZE
    CACHE_BUILD_CHUNK_SIZE
    LAMBDA_VAR
    LJ_KT
    LJ_EPSILON
    LJ_SIGMA
    LJ_CUTOFF
    LJ_SPRING_CONSTANT
    PBC
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
    printf '\nbash %q\n' "${ROOT_DIR}/train_sample_lj27_pbc.sh"
  } > "${reproduce_path}"
  chmod +x "${reproduce_path}"
  echo "[run] wrote reproduce script: ${reproduce_path}"
}

echo "[run] ckpt_root=${CKPT_ROOT}"
echo "[run] run_dir=${CKPT_DIR}"
write_reproduce_script

report_cache_stats() {
  "${PYTHON_BIN}" - "$1" "$2" <<'PY'
import sys
import torch

cache_path, label = sys.argv[1:]
try:
    payload = torch.load(cache_path, map_location="cpu", weights_only=False)
except TypeError:
    payload = torch.load(cache_path, map_location="cpu")
metadata = dict(payload.get("metadata", {}))
sample_length = payload.get("sample_length")
n_samples = int(sample_length.shape[0]) if sample_length is not None else -1
print(
    "[cache] "
    f"label={label}, "
    f"path={cache_path}, "
    f"n_samples={n_samples}, "
    f"is_discrete={metadata.get('is_discrete', metadata.get('discrete', False))}, "
    f"max_abs_relative_displacement={metadata.get('max_abs_relative_displacement', 'unknown')}"
)
PY
}

cache_matches_mode() {
  "${PYTHON_BIN}" - "$1" "$2" "$3" <<'PY'
import sys
import torch

cache_path, want_discrete, want_codebook_path = sys.argv[1:]
want_discrete = bool(int(want_discrete))
try:
    payload = torch.load(cache_path, map_location="cpu", weights_only=False)
except TypeError:
    payload = torch.load(cache_path, map_location="cpu")
metadata = dict(payload.get("metadata", {}))
cache_is_discrete = bool(metadata.get("is_discrete", metadata.get("discrete", False)))
cache_codebook_path = metadata.get("codebook_path")

ok = cache_is_discrete == want_discrete
if want_discrete:
    ok = ok and (str(cache_codebook_path) == str(want_codebook_path))
raise SystemExit(0 if ok else 1)
PY
}

cache_is_discrete() {
  "${PYTHON_BIN}" - "$1" <<'PY'
import sys
import torch

cache_path = sys.argv[1]
try:
    payload = torch.load(cache_path, map_location="cpu", weights_only=False)
except TypeError:
    payload = torch.load(cache_path, map_location="cpu")
metadata = dict(payload.get("metadata", {}))
is_discrete = bool(metadata.get("is_discrete", metadata.get("discrete", False)))
raise SystemExit(0 if is_discrete else 1)
PY
}

continuous_cache_matches_settings() {
  "${PYTHON_BIN}" - "$1" "$2" "$3" "$4" "$5" "$6" "$7" "$8" "$9" "${10}" "${11}" "${12}" "${13}" "${14}" <<'PY'
import math
import sys
import torch

(
    cache_path,
    want_factorized,
    want_polar,
    want_ordering,
    want_hilbert_resolution,
    want_window,
    want_bins,
    want_use_curve_rail,
    want_curve_rail_mode,
    want_curve_rail_window,
    want_curve_rail_k,
    want_curve_rail_reference,
    want_curve_rail_residual_target,
    want_periodic,
) = sys.argv[1:]
want_factorized = bool(int(want_factorized))
want_polar = bool(int(want_polar))
want_hilbert_resolution = int(want_hilbert_resolution)
want_window = float(want_window)
want_bins = int(want_bins)
want_use_curve_rail = bool(int(want_use_curve_rail))
want_curve_rail_window = float(want_curve_rail_window)
want_curve_rail_k = int(want_curve_rail_k)
want_curve_rail_residual_target = bool(int(want_curve_rail_residual_target))
want_periodic = bool(int(want_periodic))

try:
    payload = torch.load(cache_path, map_location="cpu", weights_only=False)
except TypeError:
    payload = torch.load(cache_path, map_location="cpu")
metadata = dict(payload.get("metadata", {}))

ok = True
ok = ok and (not bool(metadata.get("is_discrete", metadata.get("discrete", False))))
ok = ok and bool(metadata.get("periodic", False)) == want_periodic
ok = ok and bool(metadata.get("factorized", False)) == want_factorized
ok = ok and bool(metadata.get("polar", False)) == want_polar
ok = ok and str(metadata.get("ordering")) == str(want_ordering)
ok = ok and int(metadata.get("hilbert_resolution", -1)) == want_hilbert_resolution
ok = ok and math.isclose(float(metadata.get("local_window", float("nan"))), want_window, rel_tol=0.0, abs_tol=1e-9)
ok = ok and int(metadata.get("local_bins", -1)) == want_bins
ok = ok and bool(metadata.get("use_curve_rail", False)) == want_use_curve_rail
if want_use_curve_rail:
    ok = ok and str(metadata.get("curve_rail_mode")) == str(want_curve_rail_mode)
    ok = ok and math.isclose(float(metadata.get("curve_rail_window", float("nan"))), want_curve_rail_window, rel_tol=0.0, abs_tol=1e-9)
    ok = ok and int(metadata.get("curve_rail_k", -1)) == want_curve_rail_k
    ok = ok and str(metadata.get("curve_rail_reference")) == str(want_curve_rail_reference)
    ok = ok and bool(metadata.get("curve_rail_residual_target", False)) == want_curve_rail_residual_target
    if str(want_curve_rail_mode) == "fixed_template":
        ok = ok and str(metadata.get("curve_rail_storage")) == "fixed_template_generated"

raise SystemExit(0 if ok else 1)
PY
}

codebook_matches_settings() {
  "${PYTHON_BIN}" - "$1" "$2" "$3" "$4" "$5" "$6" <<'PY'
import sys
import torch

codebook_path, want_periodic, want_k, want_ordering, want_hilbert_resolution, want_dim = sys.argv[1:]
want_periodic = bool(int(want_periodic))
want_k = int(want_k)
want_hilbert_resolution = int(want_hilbert_resolution)
want_dim = int(want_dim)

try:
    payload = torch.load(codebook_path, map_location="cpu", weights_only=False)
except TypeError:
    payload = torch.load(codebook_path, map_location="cpu")

if not isinstance(payload, dict):
    raise SystemExit(1)

metadata = dict(payload.get("metadata", {}))
centers = payload.get("centers", payload.get("codebook"))
if centers is None:
    raise SystemExit(1)
centers = torch.as_tensor(centers)

ok = (
    metadata.get("periodic") == want_periodic
    and str(metadata.get("ordering")) == want_ordering
    and int(metadata.get("coord_dim", -1)) == want_dim
    and int(metadata.get("k", -1)) == want_k
    and tuple(centers.shape) == (want_k, want_dim)
)
if want_ordering == "hilbert":
    ok = ok and int(metadata.get("hilbert_resolution", -1)) == want_hilbert_resolution

raise SystemExit(0 if ok else 1)
PY
}

build_continuous_cache() {
  local output_path="$1"
  echo "[preprocess] ${DATA_H5} -> ${output_path}"
  local preprocess_args=(
    --data_h5 "${DATA_H5}"
    --output "${output_path}"
    --periodic
    --ordering "${ORDERING}"
    --hilbert_resolution "${HILBERT_RESOLUTION}"
    --window "${WINDOW}"
    --bins "${BINS}"
    --lj_epsilon "${LJ_EPSILON}"
    --lj_sigma "${LJ_SIGMA}"
    --lj_spring_constant "${LJ_SPRING_CONSTANT}"
    --num_augmentations "${NUM_AUGMENTATIONS}"
    --energy_chunk_size "${ENERGY_CHUNK_SIZE}"
    --cache_build_chunk_size "${CACHE_BUILD_CHUNK_SIZE}"
    --seed "${SEED}"
  )
  if [[ -n "${LJ_CUTOFF}" ]]; then
    preprocess_args+=(--lj_cutoff "${LJ_CUTOFF}")
  fi
  if [[ "${FACTORIZED}" == "1" ]]; then
    preprocess_args+=(--factorized)
  fi
  if [[ "${POLAR}" == "1" ]]; then
    preprocess_args+=(--polar)
  fi
  if [[ "${PREPROCESS_RANDOM_SHIFT}" == "1" ]]; then
    preprocess_args+=(--random_shift)
  fi
  if [[ "${USE_CURVE_RAIL}" == "1" ]]; then
    preprocess_args+=(
      --use_curve_rail
      --curve_rail_mode "${CURVE_RAIL_MODE}"
      --curve_rail_window "${CURVE_RAIL_WINDOW}"
      --curve_rail_k "${CURVE_RAIL_K}"
      --curve_rail_reference "${CURVE_RAIL_REFERENCE}"
    )
    if [[ "${CURVE_RAIL_RESIDUAL_TARGET}" == "1" ]]; then
      preprocess_args+=(--curve_rail_residual_target)
    fi
    if [[ -n "${CURVE_RAIL_OFFSETS}" ]]; then
      read -r -a CURVE_RAIL_OFFSETS_ARR <<< "${CURVE_RAIL_OFFSETS}"
      preprocess_args+=(--curve_rail_offsets "${CURVE_RAIL_OFFSETS_ARR[@]}")
    fi
  fi
  "${PYTHON_BIN}" "${PREPROCESS_SCRIPT}" "${preprocess_args[@]}"
}

build_discrete_cache_from_source() {
  local source_cache="$1"
  local output_path="$2"
  echo "[preprocess-discrete] ${source_cache} -> ${output_path}"
  local preprocess_args=(
    --source_cache "${source_cache}"
    --output "${output_path}"
    --discrete
    --codebook_path "${CODEBOOK_PATH}"
    --cache_build_chunk_size "${CACHE_BUILD_CHUNK_SIZE}"
  )
  "${PYTHON_BIN}" "${PREPROCESS_SCRIPT}" "${preprocess_args[@]}"
}

if [[ "${RUN_PREPROCESS}" == "1" ]]; then
  if [[ -f "${CACHE_PATH}" ]] && cache_is_discrete "${CACHE_PATH}"; then
    echo "[preprocess] existing cache is discrete, rebuilding continuous cache: ${CACHE_PATH}"
    rm -f "${CACHE_PATH}"
  fi
  if [[ -f "${CACHE_PATH}" ]] && ! continuous_cache_matches_settings \
    "${CACHE_PATH}" \
    "${FACTORIZED}" \
    "${POLAR}" \
    "${ORDERING}" \
    "${HILBERT_RESOLUTION}" \
    "${WINDOW}" \
    "${BINS}" \
    "${USE_CURVE_RAIL}" \
    "${CURVE_RAIL_MODE}" \
    "${CURVE_RAIL_WINDOW}" \
    "${CURVE_RAIL_K}" \
    "${CURVE_RAIL_REFERENCE}" \
    "${CURVE_RAIL_RESIDUAL_TARGET}" \
    "${PBC}"; then
    echo "[preprocess] existing cache metadata does not match requested settings, rebuilding: ${CACHE_PATH}"
    rm -f "${CACHE_PATH}"
  fi
  if [[ -f "${CACHE_PATH}" ]]; then
    echo "[preprocess] continuous cache already exists, skipping: ${CACHE_PATH}"
  else
    build_continuous_cache "${CACHE_PATH}"
  fi

  if [[ "${DISCRETE}" == "1" ]]; then
    if [[ -f "${CODEBOOK_PATH}" ]] && codebook_matches_settings "${CODEBOOK_PATH}" 1 "${CODEBOOK_SIZE}" "${ORDERING}" "${HILBERT_RESOLUTION}" "${DIM}"; then
      echo "[codebook] existing codebook matches run settings, skipping: ${CODEBOOK_PATH}"
    else
      echo "[codebook] ${CACHE_PATH} -> ${CODEBOOK_PATH} (k=${CODEBOOK_SIZE})"
      CODEBOOK_ARGS=(
        --cache_pt "${CACHE_PATH}"
        --output "${CODEBOOK_PATH}"
        --k "${CODEBOOK_SIZE}"
      )
      "${PYTHON_BIN}" "${CODEBOOK_SCRIPT}" "${CODEBOOK_ARGS[@]}"
    fi

    if [[ -f "${DISCRETE_CACHE_PATH}" ]] && cache_matches_mode "${DISCRETE_CACHE_PATH}" 1 "${CODEBOOK_PATH}"; then
      echo "[preprocess] discrete cache already exists, skipping: ${DISCRETE_CACHE_PATH}"
    else
      TMP_DISCRETE_CACHE="${DISCRETE_CACHE_PATH}.tmp"
      rm -f "${TMP_DISCRETE_CACHE}"
      build_discrete_cache_from_source "${CACHE_PATH}" "${TMP_DISCRETE_CACHE}"
      mv "${TMP_DISCRETE_CACHE}" "${DISCRETE_CACHE_PATH}"
    fi
  fi
  report_cache_stats "${CACHE_PATH}" "continuous"
  if [[ "${DISCRETE}" == "1" ]] && [[ -f "${DISCRETE_CACHE_PATH}" ]]; then
    report_cache_stats "${DISCRETE_CACHE_PATH}" "discrete"
  fi
fi

if [[ "${RUN_TRAIN}" == "1" ]]; then
  echo "[train] dataset=${DATA_H5} ckpt_dir=${CKPT_DIR}"
  if [[ -n "${RESUME_FROM}" ]]; then
    echo "[train] resuming from ckpt=${RESUME_FROM}"
  fi
  if [[ "${TRAIN_DATA_AUG}" == "1" ]] && [[ "${LAMBDA_VAR}" != "0" ]] && [[ "${LAMBDA_VAR}" != "0.0" ]]; then
    echo "LAMBDA_VAR requires the preprocessed cache because target energies are cached during preprocessing. Set TRAIN_DATA_AUG=0 and RUN_PREPROCESS=1." >&2
    exit 1
  fi
  TRAIN_ARGS=(
    --dataset lj_transferable
    --data_dir "${DATA_H5}"
    --lj_transfer_periodic
    --lj_transfer_ordering "${ORDERING}"
    --lj_transfer_hilbert_resolution "${HILBERT_RESOLUTION}"
    --lj_transfer_window "${WINDOW}"
    --lj_transfer_bins "${BINS}"
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
    --lambda_var "${LAMBDA_VAR}"
    --lj_kT "${LJ_KT}"
    --lj_epsilon "${LJ_EPSILON}"
    --lj_sigma "${LJ_SIGMA}"
    --lj_spring_constant "${LJ_SPRING_CONSTANT}"
    --lj_boxlength "${BOX_LENGTH}"
  )
  if [[ "${TRAIN_DATA_AUG}" == "1" ]]; then
    echo "[train] using periodic on-the-fly augmentation"
    TRAIN_ARGS+=(--lj_transfer_use_data_aug)
  else
    TRAIN_CACHE_PATH="${CACHE_PATH}"
    if [[ "${DISCRETE}" == "1" ]]; then
      TRAIN_CACHE_PATH="${DISCRETE_CACHE_PATH}"
    fi
    if [[ ! -f "${TRAIN_CACHE_PATH}" ]]; then
      echo "Missing cache for training: ${TRAIN_CACHE_PATH}. Run with RUN_PREPROCESS=1 or set TRAIN_DATA_AUG=1." >&2
      exit 1
    fi
    TRAIN_ARGS+=(--lj_transfer_preprocessed_path "${TRAIN_CACHE_PATH}")
  fi
  if [[ "${FACTORIZED}" == "1" ]]; then
    TRAIN_ARGS+=(--factorized)
  fi
  if [[ "${POLAR}" == "1" ]]; then
    TRAIN_ARGS+=(--polar)
  fi
  if [[ "${DISCRETE}" == "1" ]]; then
    TRAIN_ARGS+=(--discrete --codebook_path "${CODEBOOK_PATH}")
  elif [[ "${BINNED_DISCRETE}" == "1" ]]; then
    TRAIN_ARGS+=(--binned_discrete)
  fi
  if [[ "${USE_CONTINUOUS_HEAD}" == "1" ]]; then
    echo "[train] enabling continuous MDN head with num_mixtures=${NUM_MIXTURES}"
    TRAIN_ARGS+=(--use_continuous_head --num_mixtures "${NUM_MIXTURES}")
    if [[ "${CONTINUOUS_INPUT}" == "1" ]]; then
      TRAIN_ARGS+=(--continuous_input)
    fi
    if [[ "${FULL_COVARIANCE}" == "1" ]]; then
      TRAIN_ARGS+=(--full_covariance)
    fi
  elif [[ "${USE_COORD_DEQUANT}" == "1" ]]; then
    TRAIN_ARGS+=(--coord_dequant_width "${DEQUANT_WIDTH}")
  fi
  # ARC_REPR is passed unconditionally so train.py's own validation fires on
  # invalid combinations (e.g. ARC_REPR=1 without USE_CONTINUOUS_HEAD/CONTINUOUS_INPUT)
  # instead of silently training the baseline.
  if [[ "${ARC_REPR}" == "1" ]]; then
    TRAIN_ARGS+=(--arc_repr 1)
  fi
  if [[ "${USE_CURVE_RAIL}" == "1" ]]; then
    TRAIN_ARGS+=(
      --lj_transfer_use_curve_rail
      --lj_transfer_curve_rail_mode "${CURVE_RAIL_MODE}"
      --lj_transfer_curve_rail_window "${CURVE_RAIL_WINDOW}"
      --lj_transfer_curve_rail_k "${CURVE_RAIL_K}"
      --lj_transfer_curve_rail_reference "${CURVE_RAIL_REFERENCE}"
    )
    if [[ "${CURVE_RAIL_RESIDUAL_TARGET}" == "1" ]]; then
      TRAIN_ARGS+=(--lj_transfer_curve_rail_residual_target)
    fi
    if [[ -n "${CURVE_RAIL_OFFSETS}" ]]; then
      read -r -a CURVE_RAIL_OFFSETS_ARR <<< "${CURVE_RAIL_OFFSETS}"
      TRAIN_ARGS+=(--lj_transfer_curve_rail_offsets "${CURVE_RAIL_OFFSETS_ARR[@]}")
    fi
  fi
  if [[ -n "${RESUME_FROM}" ]]; then
    TRAIN_ARGS+=(--resume_from "${RESUME_FROM}")
  fi
  if [[ "${PBC}" == "1" ]]; then
    TRAIN_ARGS+=(--lj_periodic)
  fi
  if [[ -n "${LJ_CUTOFF}" ]]; then
    TRAIN_ARGS+=(--lj_cutoff "${LJ_CUTOFF}")
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
    --periodic
    --ar_arch "${AR_ARCH}"
    --save "${SAMPLE_OUT}"
  )
  if [[ "${FACTORIZED}" == "1" ]]; then
    SAMPLE_ARGS+=(--factorized)
  fi
  if [[ "${DISCRETE}" == "1" ]]; then
    SAMPLE_ARGS+=(--discrete --codebook_path "${CODEBOOK_PATH}")
  elif [[ "${BINNED_DISCRETE}" == "1" ]]; then
    SAMPLE_ARGS+=(--binned_discrete)
  fi
  if [[ "${USE_CONTINUOUS_HEAD}" == "1" ]]; then
    SAMPLE_ARGS+=(--use_continuous_head)
    if [[ "${FULL_COVARIANCE}" == "1" ]]; then
      SAMPLE_ARGS+=(--full_covariance)
    fi
  fi
  if [[ "${POLAR}" == "1" ]]; then
    SAMPLE_ARGS+=(--polar)
  fi
  "${PYTHON_BIN}" "${ROOT_DIR}/sample_lj.py" \
    "${SAMPLE_ARGS[@]}" \
    "${TOP_K_ARGS[@]}"
fi
