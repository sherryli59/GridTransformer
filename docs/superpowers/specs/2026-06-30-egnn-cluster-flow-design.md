# EGNN cluster flow — conditional CNF with traceable divergence + OT flow matching (design)

**Date:** 2026-06-30
**Status:** design (approved; → writing-plans)
**Context:** proposal-B (`ka_cluster_flow_b.py`, RealNVP parity-split coupling) failed the g(r) gate at 55% intra+context
clash — but the NO-GO was PREMATURE: root-caused (2026-06-30) to a **coupling-structure blind spot**, NOT the
hard-core wall. Diagnostic: 85% of intra-cluster clashes are SAME-parity pairs (vs 43% by construction); RealNVP
places a whole parity group in parallel conditioned only on the frozen group, so two same-parity particles never
see each other's transformed coordinate and structurally cannot push apart. The flow moved points 0.68 (≈σ_b, not
near-identity) yet couldn't de-conflict them. Fix = an **all-see-all** generator with no parity partition. See
[[cluster-proposal-b-equivariant-flow]], [[traceable-egnn]].

## Goal / success criterion
Pass the same committed g(r) gate (`ka_cluster_flow.gate_measure` / `diag`): single-cluster clash → ~0, g_BB peak →
~2.30, AND the intra-cluster same-parity signature must vanish (there is no parity partition here by construction).
**The gate decides, not the training loss.**

## Architecture: conditional EGNN CNF (frame-free)
A continuous normalizing flow. The **k=7 cluster particles move**; the **cage (fixed context) does not**. Integrate
`dx_cluster/dt = v_i(x_cluster, x_cage; species)` over t:0→1; the velocity is the periodic traceable EGNN central
force `v_i = Σ_j pot(r_ij, t, h, species)·minimg(x_i − x_j)` (`EGNN_dynamics.forward`, periodic branch — no COM
projection). E(2)-equivariance is intrinsic (relative positions); **no frame** (frame-free per design decision).
`cluster_slots`/`slot_order` are reused only to IDENTIFY the cluster + cage; `cluster_frame`/`to_frame` are dropped.

**NOT translation/rotation-invariant for the cluster alone (design keystone):** `q(cluster | cage)` pins the
cluster's absolute position AND orientation relative to the fixed cage (it fills one specific hole). Only the JOINT
move (cluster+cage together) leaves the density invariant. Therefore: **NO zero-COM subspace**, base is a full
2k-dim Gaussian, log-det over all 2k dims, and the reference is the FIXED cage (never a moving-cluster-COM
centering — that would delete the cage offset that pins position). The periodic EGNN branch already omits COM
centering, so this is automatic.

## The conditional divergence (exact, cluster-restricted)
`EGNN_dynamics.forward_and_divergence(xs, t, a)` returns `(vel [B,P,D], divergence [B])` where internally
`divergence_term [B,P]` is PER-PARTICLE (line ~253-257) and only summed over P at the end. The cluster CNF log-det
rate is `Σ_{i∈cluster} div_i = −divergence_term[:, cluster_idx].sum(-1)` — exact, because the Jacobian trace is
`Σ_i ∂v_i/∂x_i` per-particle and the cage is fixed (its dims are not in the log-det). Implementation exposes the
per-particle divergence (small edit or a wrapper) and sums over the k cluster rows. The `pot` radial derivative
`dpotdr` is a cheap exact 1D autograd on `rij` (flash-div), and species enter as position-independent labels so
the divergence stays exact.

## Base: equivariant full 2k-dim Gaussian at the cage centroid
`x(t=0) ~ N(c, σ_b²·I)` over all k particles independently, where **c = the cage centroid** (min-imaged mean of the
n_cage context positions) — equivariant, cage-derived, carries the absolute-position information. σ_b **derived from
data** (isotropic RMS of the true cluster particles' min-imaged distance from the cluster centroid; expected
~1.0–1.3). Full-space Gaussian (NOT zero-COM). `base_logp(x0) = Σ_i [ −|x0_i − c|²/(2σ_b²) − log(2π σ_b²) ]`.
The base carries the cluster species (fixed, `s[cluster_idx]`).

## Cage context (the "large enough" knob; size-transferable)
Cloud = cluster (k) + the **nearest n_cage cage particles** (fixed count → fixed cloud P=k+n_cage, required by
`EGNN_dynamics`). `n_cage ≈ 48`, EGNN `cutoff = r_c ≈ 3.0` (safely past the LJ range ~2.5; edges beyond r_c get zero
interaction). n_cage must exceed the count within r_c at ρ=1.2 (~24–32), so 48 guarantees the whole interacting cage
is present. The cutoff (not the count) sets the physics → **N-independent / size-transferable**. Cage selected by
min-image distance to the cluster centroid.

## Training: OT conditional flow matching (species-aware per-particle OT)
NOT maximum likelihood (no ODE-in-the-loop). Per (true cluster x, its cage) in a batch:
1. draw base `z ~ N(c, σ_b·I)` (k particles, carrying the cluster species);
2. solve the **species-aware per-particle OT assignment** `π = argmin_π Σ_i ‖z_i − x_{π(i)}‖²` restricted to
   same-species matches (Hungarian on the k×k min-image cost, ∞ cost across species; k=7 → trivial);
3. interpolate `x_t = (1−t) z + t x_π`, `t ~ U(0,1)`;
4. regress `v_EGNN(x_t, t; cage, species)` onto the target `(x_π − z)` by MSE (cage nodes fixed, only cluster
   velocities in the loss).
Cheap, stable, straight paths (fewer inference RK4 steps). σ_b matched to the cluster spread so base/data overlap.

## Inference (sample + exact log_q)
- `sample`: `z ~ base`, integrate `dx_cluster/dt = v` t:0→1 with a fixed-step RK4 (N_steps ~12–20), cage fixed;
  `logq = base_logp(z) − ∫ Σ_cluster div_i dt` (accumulate the cluster-restricted divergence each RK4 step).
- `log_q(xC)`: integrate t:1→0 to recover z; `logq = base_logp(z) + ∫ Σ_cluster div_i dt`.
- Fixed-step deterministic integrator ⇒ `sampler==scorer` to integration precision (~1e-3, same discipline as
  proposal-B). The analytic divergence is exact; only the ODE integral is discretized.

## Interface (drop-in for the gate)
`EGNNClusterFlow` mirrors `ClusterProposal`/`ClusterFlow`:
- `sample(pos[B,N,2], s[B,N], cluster_idx[k], sc[N,2], L) -> (xC_lab[B,k,2], logq[B])` (@no_grad)
- `log_q(pos[B,N,2], s[B,N], cluster_idx[k], xC_query[B,k,2], sc[N,2], L) -> logq[B]`
pos/s slot-ordered; reuse `slot_order`. So the committed `gate_measure`/`diag`/MH work unchanged.

## Files / components
- `liquid_coupling_flow/ka_cluster_egnn.py` — `EGNNClusterFlow` (base + cloud construction + conditional RK4
  integrator with cluster-restricted divergence + OT-FM `train()` + `gate()`), wrapping `EGNN_dynamics` (import from
  `liquid_coupling_flow.ipl44.learndiffeq.learndiffeq.particles.velocities.egnn_traceable`). If the vendored
  per-particle divergence isn't cleanly exposed, add a thin `forward_and_perparticle_divergence` alongside it.
- `liquid_coupling_flow/tests/test_ka_cluster_egnn.py` — exactness guards.
- Reuses `ka_cluster.py` (slot_order, cluster_slots only), `ka_cluster_flow.gate_measure`.

## Exactness guards (3, same discipline)
1. **divergence vs brute-force** (definitive): analytic `Σ_cluster div_i` == autograd trace of the cluster velocity
   Jacobian (`torch.autograd.functional.jacobian` on the k cluster dims), ~1e-5, at random t.
2. **sampler == scorer**: `log_q(sample())` == returned `logq`, ~1e-3 (RK4 reversibility).
3. **base_logp** analytic vs closed-form Gaussian.
Plus an **OT sanity** test (species-respecting assignment; cost ≤ identity assignment).

## Build sequence (for the plan)
1. Cloud construction (cluster + nearest n_cage cage, species, min-image) + base (cage-centroid Gaussian, σ_b from
   data) + tests.
2. Conditional velocity + cluster-restricted per-particle divergence wrapper over `EGNN_dynamics` + divergence-vs-
   brute-force test.
3. RK4 conditional integrator (sample/log_q) + sampler==scorer + base_logp tests.
4. Species-aware per-particle OT + interpolation + OT-FM `train()` + smoke (loss drops, straight paths).
5. `gate()` + real training run + the g(r) gate (GO/NO-GO), re-run the intra/parity split (must collapse).

## Risks / kill criteria
- **EGNN_dynamics COM/periodic assumptions**: confirm the periodic branch gives pure central-force (no COM) on our
  cloud; the cage must be a true fixed reference (verify a cluster-only translation changes logq).
- **Fixed cloud vs cutoff**: n_cage=48 must cover r_c; verify no interacting cage particle is dropped.
- **If the gate STILL floors** (clash stays high, intra-clash no longer same-parity-dominated): THAT would be the
  real expressiveness/precision wall — then the SMC corrector [[glass-smc-corrector-headstart]] is the call. Keep
  the bug door open [[never-refute-bug-hypothesis]]; the gate + concrete checks decide.
