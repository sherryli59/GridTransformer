# IPL44 lever push: species-flow swap proposer + professor-forcing arm — design

**Date:** 2026-07-02 · **Branch:** liquid-coupling-flow · **Status:** approved (user delegated autonomous execution)
**Context:** `reports/2026-07-02-ersi-reproduction-status.md` (step-back), memory `joint-species-flow`,
`ipl44-ar-vs-ersi-verdict`, `glass-smc-corrector-headstart`.

## 1. Goal & claim under test

On **IPL44** (2D, N=44, 22:22 binary soft spheres r⁻¹², σ=[[1,1.2],[1.2,1.4]], ρ=0.5→L=√88, **T=0.1**,
10K Zenodo reference at `/mnt/ssd/GridTransformer/datasets/`): push the joint species-position flow
(`liquid_coupling_flow/ipl44/joint_flow.py`) from v1-minimal (23.7k params) to a serious model, and prove its
species **denoiser is a useful learned swap proposer**: MH-exact swap-MCMC with learned swaps reaches
equilibrium in **materially fewer sweeps** than with random swaps. Secondary: a bounded professor-forcing (PF)
bet on the AR curve-flow transformer. **Out of scope:** size transfer (next spec, once a lever passes its gate);
one-shot IS/ESS (proven dead); the EGNN full-cage refinement lever (deferred).

Strategic backdrop (user-set): make the transformer + EGNN-flow assets *useful*; the eventual headline is
train-small→sample-big on a hard glassy system; user judgment: architecture+training procedure matters more
than exposure-bias tricks (soft labels / scheduled sampling ran, didn't help); develop levers on the small
IPL44 system first.

## 2. Arm 1a — species-flow push (architecture + training procedure)

Extend `joint_flow.py` **in place**, keeping the v1 API:

- **Architecture:** configurable `hidden_nf`/`n_layers` for the shared EGNN trunk; grid = v1 (32|3-ish, 23.7k)
  / ~64|4 / ~128|5 (≈100k–500k). Optional k-NN neighbor trim. Species head: per-particle logits from node
  features (as v1) — deeper MLP allowed.
- **Training procedure** (lessons already paid for on this exact system):
  - **per-species minibatch OT** for the position channel (measured 6.5× loss drop at init on IPL44; v1 used
    global OT),
  - **EMA weights + LR warmup; lr ≤ 1e-3** (5e-3 diverged twice on IPL44 this week),
  - best-by-val + last checkpointing only (no per-epoch accumulation), checkpoints under
    `liquid_coupling_flow/ipl44/data/` (gitignored),
  - λ (species-CE weight) and Kawasaki-noise schedule tuned on the **denoiser metric** (G1), not position loss.
- **Ablation for the user's hypothesis:** v1-config vs scaled-v2 on identical data, compared on denoiser
  quality (G1) and downstream sweeps-saved (G2).

**Note (asset loss):** the old session scratchpad is gone — v1's training script and `joint_flow_T01.pt`
checkpoint no longer exist. The v1 config is therefore **retrained** as the small point of the grid; the swap
kernel is **reimplemented** (math documented in memory `joint-species-flow` and §3).

**Gate G1 (model-level, before any SMC):** denoiser species-recovery accuracy at t→1 on held-out reference
configs (v1 historical = 0.83) + calibration of p_θ(s_i=B|x). Pass = clear improvement over the retrained-v1
baseline. Kill = scaling doesn't move it.

## 3. Arm 1b — swap-proposer harness + benchmark

**New module `liquid_coupling_flow/ipl44/ipl_swap_smc.py`** (reimplements the verified scratchpad kernel):

- Position channel: single-particle Metropolis moves (sweep = N attempted moves), matching the earlier
  MCMC-relaxation setup.
- Swap channel: MH swap of an (A,B) pair. Learned proposer: weights w_B(i)=p_θ(s_i=B|x,s,t_prop) from the
  denoiser; propose A-particle i with prob w_B(i)/S_A (A-particles ranked by "wants to be B") and B-particle j
  with prob w_A(j)/S_B; **exact reverse-proposal correction** in the MH ratio (post-swap weights recomputed;
  forward g = (w_B(i)/S_A)(w_A(j)/S_B), reverse g' from the post-swap denoiser evaluation). Random proposer =
  all-ones weights (ratio 1). Count-preserving by construction.
- `t_prop` grid around 0.9 (historical sweet spot).
- **Benchmark protocol** at T=0.1: seeds {uniform, transformer bank (from `ipl44_curveflow.pt`), flow samples}
  × proposers {random, learned}; track energy-distribution distance to reference, g_AA/g_AB/g_BB partials,
  swap acceptance × moved-mass, vs sweeps AND wall-clock (denoiser forwards aren't free — report both).

**Gate G2 (lever-level):** learned swaps reach the equilibrium band — defined as median U within 5% of the
reference median (ref ⟨U⟩≈14.7 total) AND g_BB peak height within 15% of the reference peak (≈4.3) — in **≥2×
fewer sweeps** than random swaps from ≥1 seed type, MH-exact. Kill = <20% gain at every t_prop with the scaled
model. Known failure mode kept as an explicit axis: from uniform seeds positions dominate and learned ≡ random
(measured before); the lever's regime is decent-positions/frustrated-species seeds.

## 4. Arm 2 — professor forcing (bounded, gated)

- **Pre-gate G0-PF (cheap, decisive):** TF-vs-FR per-index decomposition on the IPL44 curve-flow transformer
  (`ipl44_curveflow.pt`; machinery pattern from the KA exposure campaign). PF attacks **drift**; if the gap is
  all residual (per-step half-cage, as on KA), kill the arm at ~zero cost.
- **PF (only if drift shows):** small discriminator (MLP over final-layer hidden states per AR step) separating
  TF from FR rollouts; generator adds non-saturating adversarial term (weight β_pf) to the NLL — density
  parameterization untouched ⇒ **exact likelihood preserved**. Cost control: PF loss every k-th batch; FR
  *segments* from TF prefixes.
- **Budget:** ≤5 configurations (β_pf, D size, rollout schedule). **Success:** FR discard 0.987 → <0.9 or
  excluded-volume core mass −30%, with NLL degradation <5%. **Kill:** budget exhausted / GAN instability /
  pre-gate all-residual.

## 5. Code layout & testing

```
liquid_coupling_flow/ipl44/
  joint_flow.py          # extended in place (arch config, per-species OT, EMA); v1 API kept
  train_joint_flow.py    # NEW training script: grid configs, warmup+EMA, best+last ckpt, G1 eval
  ipl_swap_smc.py        # NEW: position sweeps + swap-MH kernel (learned/random) + sweeps-to-eq benchmark
  ipl_pf.py              # NEW (arm 2, only past G0-PF): PF discriminator + training wrapper
liquid_coupling_flow/tests/test_swap_smc.py   # NEW (see below)
```

**Tests (TDD where the math can be wrong silently):**
- swap-MH detailed balance: long-run stationarity on a tiny exactly-checkable system (e.g., N=6, 3:3) — species
  marginals under learned proposer match those under random proposer (both target the same Boltzmann);
- count preservation under both proposers;
- forward/reverse proposal-ratio correctness (hand-computed case);
- joint-flow smoke: loss decreases, samples are exactly 22:22, denoiser eval runs.

Figures/reports: benchmark plots to `liquid_coupling_flow/ipl44/data/`, findings to a dated
`reports/<date>-ipl44-lever-push-results.md` written when results land.

## 6. Gates recap & handoff

| gate | measure | pass | kill |
|---|---|---|---|
| G1 model | denoiser acc/calibration vs retrained-v1 | clear improvement | scaling doesn't move it |
| G2 lever | sweeps-to-eq, learned vs random swaps | ≥2× from ≥1 seed type | <20% at all t_prop |
| G0-PF | TF-vs-FR decomposition on IPL44 | drift component exists | all-residual |
| G-PF | FR discard / core mass | <0.9 / −30%, NLL cost <5% | 5 configs, no signal |

**Handoff:** a lever passing its gate becomes the subject of the next spec — train-small→sample-big (KA glass
with PT references at N=36/100/256, or IPL multi-size with self-generated references).

## 7. Operational constraints

- GPU is shared with the user's own jobs — check `nvidia-smi` before launching; small-chunk + OOM-retry
  patterns; long runs via nohup + log-watchers.
- lr ≤1e-3 with warmup+EMA for all new trainings (standing lesson).
- Never conclude "no bug" — negative results get a concrete falsification check before being recorded.
