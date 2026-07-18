# Running the poly NCMC campaign on Sherlock

1. `git pull` on the cluster (branch mw-cell-q0). Required inputs are committed:
   clean frame banks `poly_bank_fixed_run{1,2,3}.pt`, selector model
   `poly_selector_model_T0.085.pt`, all scripts.
2. Env: python>=3.10 with `numba`, `numpy`, `torch` (CPU build is fine for the chains).
3. Launch the replica fleet (40 single-core tasks, ~12-24h each at 4M sweeps):
   `sbatch reports/logs-2026-07-17/poly_slurm_chains.sh`
4. Collect: each task writes `poly_cluster_{plain|screened}_T{T}_rep{R}.pt` with full
   per-gridpoint time series (both arms). Analysis = replica-averaged C_sig(t) +
   accepted-exchange flux, screened vs plain; commit the .pt files back (~100-300KB each).

Notes:
- Incremental checkpointing throughout; re-running a task is safe (deterministic seeds).
- GPU not needed for chains. Flow trainings (if extended) use any 1-GPU partition.
- Bank generation for larger N ports trivially: `python reports/logs-2026-07-17/poly_bank.py <N> 10000000 <RUN>`.
