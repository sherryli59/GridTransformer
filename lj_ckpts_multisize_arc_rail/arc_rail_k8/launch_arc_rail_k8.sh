#!/usr/bin/env bash
# Multi-size arc_repr RAIL-CONDITIONING probe (action plan P7 / Arm B+ ingredient).
#
# Motivation: P3 showed the held-out Δs drift is a POSITION-dependent pacing
# miscalibration, not context corruption; P6 confirmed input-noise training did not
# move it. The fixed_template curve rail feeds decode(j*X)-derived waypoints as INPUT
# (cross-attention), supplying exactly the missing pacing reference. Inputs need not be
# size-invariant, so this sidesteps the N^(1/6) target issue entirely.
#
# Cache: the existing non-rail L3L5 cache is reused — fixed_template waypoints are
# sample-independent and synthesized on the fly (LJTransferableCachedDataset unlock).
# Warm-start: CurveRailAttention.out_proj is zero-initialized, so the rail is an exact
# no-op at step 0 and the baseline is reproduced byte-for-byte; any metric movement is
# causally the rail's.
set -euo pipefail
cd /mnt/ssd/GridTransformer

PY=/home/sherryli/xsli/softwares/anaconda3/envs/lightning/bin/python
RUN_DIR=lj_ckpts_multisize_arc_rail/arc_rail_k8
BASE_CKPT=lj_ckpts_multisize_arc_L3L5/multisize_arc_fullcov/best.ckpt

exec "$PY" train.py --dataset lj_transferable \
  --data_dir /mnt/ssd/mcmc/lj_mcmc_sweep_3d/lj3d_L3_rho1.0_N27_T1.0.h5 \
  --lj_transfer_periodic --lj_transfer_ordering hilbert \
  --lj_transfer_hilbert_resolution 64 --lj_transfer_cell_size 0.046875 \
  --lj_transfer_window 3.0 --lj_transfer_bins 64 \
  --epochs 20 --batch_size 512 --lr 1e-3 --num_workers 8 --seed 0 \
  --dim 512 --heads 4 --depth 4 \
  --ckpt_dir "$RUN_DIR" \
  --ar_arch standard --lambda_var 0.0 \
  --lj_kT 1.0 --lj_epsilon 1.0 --lj_sigma 1.0 --lj_spring_constant 0.5 --lj_boxlength 3.0 \
  --lj_transfer_preprocessed_path lj_caches_arc_transfer/lj_multisize_L3L5_cell0469_cache.pt \
  --use_continuous_head --num_mixtures 64 --continuous_input --full_covariance --arc_repr 1 \
  --warm_start_ckpt "$BASE_CKPT" \
  --lj_transfer_use_curve_rail \
  --lj_transfer_curve_rail_mode fixed_template \
  --lj_transfer_curve_rail_k 8 \
  --lj_transfer_curve_rail_window 1.0 \
  --lj_transfer_curve_rail_reference absolute \
  --lj_transfer_val_frac 0.05 \
  --lj_periodic
