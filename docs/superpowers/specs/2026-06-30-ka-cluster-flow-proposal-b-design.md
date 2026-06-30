# Proposal-B: `ClusterFlow` — equivariant exact-likelihood cluster generator (design)

**Date:** 2026-06-30
**Status:** design (awaiting user review → writing-plans)
**Context:** Proposal-A (the AR-in-frame binned cluster generator, `ka_cluster_flow.py:ClusterProposal`) **failed
the g(r) gate** even after sharpening (finer bins n_bins=64/bin_w 0.094, n_ctx=32, d_model=192, 40k steps):
single-cluster clash 44%, g_BB 1.35, seed places 0.75 from its true spot given the *true* cage (data: clash 0%,
g_BB 2.30). The diffuseness is fundamental to the AR-in-frame binned conditional, not a resolution/capacity
artifact. Per user directive, build **proposal-B = a from-scratch in-frame coupling FLOW**. The earlier
full-system flow (`ka_flow_coupling.py:KACouplingFlow`) had spline+species+geometric-attention and still floored —
but on the *much harder* full-N problem (build everything from uniform). The cluster setting is far easier: the
cage (x_R) is **given** and we place only **k=7** particles, in-frame.

## Goal / success criterion

Beat proposal-A on the **same committed g(r) gate** (`ka_cluster_flow.diag`): single-cluster clash% → ~0 and
3.3b g_BB peak → ~2.0 (vs proposal-A 44% / 1.35). **The gate decides, not the loss** — a flow can be diffuse too
(standing directive). If it also floors, that is strong evidence the local cage genuinely underdetermines the
position, and the validated SMC corrector is the call.

## Architecture

A coupling flow on the **2k=14 in-frame coordinates** of the k=7 cluster particles. Everything is in the
**x_R-only equivariant frame** (reuse `ka_cluster.py`), so E(2)-equivariance is by construction and the total
Jacobian is `flow-log-det × frame-Jacobian(=1)` — exact.

### Base (data-derived, fixed)
Per-particle isotropic Gaussian centered at that particle's **scaffold-anchor** `q_scaf_i` (the cluster slot's
in-frame scaffold position — config-independent), width **σ_b = 0.747** (FIXED, derived once from the reference:
isotropic std of the min-imaged in-frame displacement `true − anchor`; per-coord [0.76, 0.73]; the displacement
is well-described by a 2D Gaussian — mean|disp| 0.92 ≈ 1.25·σ_b). σ_b is saved in the checkpoint, not learned.

`base_logp(z) = Σ_i [ −log(2π σ_b²) − |z_i − q_scaf_i|² / (2 σ_b²) ]`.

Gaussian base ⇒ **R² support, no boundary/coverage concern** (clean win over proposal-A's bin support).

**Design note (slot-order non-locality tail):** 3.3% of cluster particles sit >2 (particle-spacings) from their
scaffold slot (only 0.11% >3) — the curve-order locality limit, so the displacement has a ~3% heavy tail beyond
the Gaussian bulk. The RQ spline's **linear tails** absorb it (base assigns low-but-finite density; the flow maps
it). Minor contributor at most; recorded, not blocking. Do NOT clip (would break exactness).

### Transform: standard (non-circular) rational-quadratic neural spline
Elementwise per-coordinate RQ neural spline with **linear tails** (standard NSF, Durkan et al.) — NOT the existing
circular/torus `transforms_spline.CircularRQSplineElementwise` (the in-frame box is not periodic; the base is
Gaussian on R²). A new `transforms_spline_nonperiodic.py:RQSplineElementwise` (tail_bound ~ a few·σ_b, e.g. 4.0;
num_bins ~8–12). Exactly invertible, analytic log-det. RQ chosen over affine for expressive sharp hard-core walls.

### Coupling
Split the k particles into two parity sets by canonical (cluster_slots) index — 4/3 for k=7. Each block transforms
one set's two coordinates conditioned on the frozen set + context; alternate sets across blocks; several cycles
(~4–6). Triangular Jacobian ⇒ exact log-det = Σ spline log-dets of the active block.

### Equivariance & exactness
In-frame coords only ⇒ E(2)-equivariant. `sample`: z ~ base (anchor Gaussians) → forward blocks → u (in-frame) →
`from_frame` → lab xC; `logq = base_logp(z) − Σ log-det`. `log_q(xC)`: `to_frame` → u → inverse blocks → z;
`logq = base_logp(z) + Σ log-det`. Frame from x_R only (bit-exact x_C-independent, inherited keystone).

## Conditioning — the must-get-right (species + local context, per-layer)

Every coupling block's spline parameters for the active particles come from a **per-layer geometric-attention
conditioner** (reuse `ka_flow_coupling.py:GeomEdgeBias` + `GeomAttnLayer`) attending over the token set
`{active-particle queries, frozen cluster particles, the n_ctx x_R cage}` with a **distance-RBF + pair-type
(species×species) attention bias**:

- **Context is LOCAL and PER-LAYER** — every block re-attends to the actual in-frame cage geometry (the n_ctx x_R
  positions+species), not a single global token. This is the specific failure the full-system flow had.
- **Species two ways** — embeddings on every token AND the pair-type attention bias — so type-dependent excluded
  volume (σ_BB 0.88 vs σ_AA 1.0) is directly expressible. `sp = s[:, cluster_idx]` are fixed-identity.
- Spline params for an active particle never depend on its OWN active coordinate being transformed (coupling
  invertibility): condition on the frozen set + the active set's *other* (fixed-this-block) coordinate.

## Interface (drop-in for the gate / diag / MH)

Mirror `ClusterProposal` exactly so the committed `diag`/`gr_gate`/MH kernel work unchanged:
- `sample(pos[B,N,2], s[B,N], cluster_idx[k], sc[N,2], L) -> (xC_lab[B,k,2], logq[B])` (@no_grad)
- `log_q(pos[B,N,2], s[B,N], cluster_idx[k], xC_query[B,k,2], sc[N,2], L) -> logq[B]` (grad-capable)
`pos`,`s` slot-ordered (slot j ↔ particle near sc[j]); reuse `slot_order`.

## Files / components

- `liquid_coupling_flow/transforms_spline_nonperiodic.py` — `RQSplineElementwise` (linear-tail RQ-NSF: forward,
  inverse, log-det).
- `liquid_coupling_flow/ka_cluster_flow_b.py` — `ClusterFlow(nn.Module)` (base + coupling stack + geometric
  conditioner, reusing `ka_cluster.py` frame/slot-order and `GeomEdgeBias`/`GeomAttnLayer`), plus `train()`
  (conditional MLE, identical recipe to proposal-A: slot-order → random seed-cluster → maximize
  `log_q(true cluster | true cage)`; checkpoint stores arch + σ_b) and `__main__` (train / `sharp` n/a).
- `liquid_coupling_flow/tests/test_ka_cluster_flow_b.py` — exactness guards.
- Reuse `ka_cluster_flow.diag` / `gr_gate` for the gate (same interface; load via a `_load`-style helper).

## Exactness guards (3, like the single-site/cluster work)

1. **Flow round-trip** — `forward(inverse(u)) == u` and `inverse(forward(z)) == z` to ~1e-4 (spline invertibility).
2. **Sampler == scorer** — `log_q(sample())` equals the `logq` the sampler returned to ~1e-4 (real conditioning).
3. **Frame bit-exactness** — inherited `test_frame_independent_of_xC_bit_exact` (move whole cluster → frame
   identical to the bit), guaranteeing Jacobian-1.
Plus an **analytic base_logp** check (numeric vs closed-form Gaussian) and a **log-det** check (autograd Jacobian
vs the analytic spline log-det on a small batch).

## The g(r) gate (decisive GO/NO-GO; reuse `diag`)

Single-cluster clash% (no co-placement) + seed spread + 3.3b self-consistent g_BB. **GO iff** clash → ~0 and g_BB
peak → ~2.0 (clearly beating proposal-A's 44% / 1.35). On GO → the cluster MH kernel (proposal-A's Tasks 6–8,
unchanged, since the interface matches). On NO-GO → the cluster-flow line is closed; SMC corrector is the call.

## Build sequence (for the plan)

1. `RQSplineElementwise` (non-periodic) + tests (round-trip, log-det).
2. Anchor-Gaussian base (σ_b from data, fixed) + base_logp test.
3. Geometric conditioner (adapt `GeomEdgeBias`/`GeomAttn` to the cluster token set).
4. Assemble `ClusterFlow` in-frame; exactness guards 1–3 + base/log-det checks.
5. Train (conditional MLE).
6. g(r) gate (`diag`) → GO/NO-GO.

## Risks / kill criteria

- **Flow equally diffuse** — parallel coupling may not beat the AR's per-particle breadth (memory warns parallel
  coupling struggles with hard cores). Mitigated by per-layer distance-attention; the gate is the test.
- **Diminishing returns like sharpening** — if the gate barely moves (clash ~40%, g_BB ~1.4), call it NO-GO and
  stop (don't sharpen the flow); the local cage underdetermines the position → SMC corrector.
- Keep the bug door open (standing directive): the gate, concrete checks — never conclude "no bug".
