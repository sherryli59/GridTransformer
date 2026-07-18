#!/bin/bash
#SBATCH --job-name=poly_ncmc
#SBATCH --array=0-39
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=24:00:00
#SBATCH --output=poly_chain_%a.out
# Sherlock array: replicated screened/unscreened NCMC chains at the swap arrest.
# 40 tasks = 2 temps x 10 reps x 2 modes. Budget 4M sweeps (2x the local runs).
# Requires: repo pulled, python env with numba+torch(cpu ok), sbatch from repo root.
set -e
cd "$SLURM_SUBMIT_DIR"
TEMPS=(0.047 0.052)
TID=$SLURM_ARRAY_TASK_ID
T=${TEMPS[$(( TID / 20 ))]}
REP=$(( (TID % 20) / 2 ))
MODE=$(( TID % 2 ))          # 0 = unscreened, 1 = screened
SEL=""
TAG="plain"
if [ "$MODE" = "1" ]; then
  SEL="--selector reports/logs-2026-07-17/poly_selector_model_T0.085.pt"
  TAG="screened"
fi
python reports/logs-2026-07-17/poly_ncmc_chain.py \
  --T "$T" --budget 4000000 --run 3 --mixes swap,swap+ncmc \
  --n_steps 800 --r_loc 3.0 --ncmc_every 50 $SEL \
  --seed_offset $(( 100 + REP )) \
  --out "reports/logs-2026-07-17/poly_cluster_${TAG}_T${T}_rep${REP}.pt"
