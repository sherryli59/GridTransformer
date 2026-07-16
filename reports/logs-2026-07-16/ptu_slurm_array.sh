#!/bin/bash
#SBATCH --job-name=ptu_gpts
#SBATCH --array=0-99
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=60:00:00
#SBATCH --output=reports/logs-2026-07-16/slurm_%A_%a.out

# Identity-bridge PT production: BCY-budget (1e7 sweeps/cavity) certified q_c via u-ladder.
# Embarrassingly parallel at the cavity level: each array task = ONE cavity, ONE core.
# Task mapping: tasks 0-49 -> R=2.0 offsets 0-49; tasks 50-99 -> R=2.3 offsets 0-49.
# Requirements per node: python env with numba + torch (CPU), repo checkout (branch mw-cell-q0),
# dataset liquid_coupling_flow/artifacts/ka3d_dataset_N512_T0.5.pt (committed in-repo).
# Outputs: reports/logs-2026-07-16/ptu_production_R{R}_off{off}.pt (incremental, chunked saves --
# safe to requeue/resume-inspect mid-run; per-chunk progress in the slurm .out).
# Analysis afterward: aggregate the per-cavity .pt files (keys: per-cavity records + "summary").
# NOTE R <= 2.4 only with the N512 box (2R + r_cut < L = 7.5283). R >= 2.6 needs a larger box.

cd "$SLURM_SUBMIT_DIR"
TASK=${SLURM_ARRAY_TASK_ID}
if [ "$TASK" -lt 50 ]; then
  R=2.0; OFF=$TASK
else
  R=2.3; OFF=$((TASK - 50))
fi
echo "task $TASK -> R=$R cav_offset=$OFF ($(date))"
python reports/logs-2026-07-16/ptu_production.py \
  --r "$R" --sw 10000000 --chunk 200000 --ncav 1 --max_attempt 1 --cav_offset "$OFF"
