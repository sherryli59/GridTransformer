#!/usr/bin/env bash
# Variant A (JointEdgeBias) multi-size probe — joint (pair distance, arc separation)
# attention bias. Previously validated single-size at N=27 (PROMOTE: only Phase-3 probe
# below baseline NLL); NEVER tested multi-size for held-out transfer until now.
#
# Motivation: P3 showed the held-out failure is a Δs-vs-position pacing structure. Variant
# A gives attention an explicit arc-separation channel (RBF(log1p Δs) × RBF(distance)),
# complementary to the rail (which conditions on the pacing reference as an INPUT). This
# probe asks whether the joint-bias ATTENTION STRUCTURE alone moves held-out L4 drift.
#
# arc_s is computed on the fly as cumsum(input_deltas[...,0]) — the existing non-rail
# multi-size cache works as-is (no arc_s field needed, no rebuild). JointEdgeBias output
# is zero-init ⇒ exact no-op at warm start ⇒ causally clean. 10 epochs (matches the
# round-1 protocol that gave a clear verdict) to fit the GPU-free window.
#
# LR is parameterized: default 1e-3 (variant A's validated setting). If the rail probe's
# Gate-A trajectory shows 1e-3 warm-start divergence, override LR=2e-4 at launch.
set -euo pipefail
cd /mnt/ssd/GridTransformer

PY=/home/sherryli/xsli/softwares/anaconda3/envs/lightning/bin/python
RUN_DIR=lj_ckpts_multisize_arc_variantA/variantA_jointbias
BASE_CKPT=lj_ckpts_multisize_arc_L3L5/multisize_arc_fullcov/best.ckpt
LR="${LR:-1e-3}"
EPOCHS="${EPOCHS:-10}"

exec "$PY" train.py --dataset lj_transferable \
  --data_dir /mnt/ssd/mcmc/lj_mcmc_sweep_3d/lj3d_L3_rho1.0_N27_T1.0.h5 \
  --lj_transfer_periodic --lj_transfer_ordering hilbert \
  --lj_transfer_hilbert_resolution 64 --lj_transfer_cell_size 0.046875 \
  --lj_transfer_window 3.0 --lj_transfer_bins 64 \
  --epochs "$EPOCHS" --batch_size 512 --lr "$LR" --num_workers 8 --seed 0 \
  --dim 512 --heads 4 --depth 4 \
  --ckpt_dir "$RUN_DIR" \
  --ar_arch standard --lambda_var 0.0 \
  --lj_kT 1.0 --lj_epsilon 1.0 --lj_sigma 1.0 --lj_spring_constant 0.5 --lj_boxlength 3.0 \
  --lj_transfer_preprocessed_path lj_caches_arc_transfer/lj_multisize_L3L5_cell0469_cache.pt \
  --use_continuous_head --num_mixtures 64 --continuous_input --full_covariance --arc_repr 1 \
  --warm_start_ckpt "$BASE_CKPT" \
  --use_joint_arc_bias 1 \
  --lj_transfer_val_frac 0.05 \
  --lj_periodic
