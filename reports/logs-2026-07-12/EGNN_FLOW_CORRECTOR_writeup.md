# Exact equivariant flow corrector for 3D KA glass cavity blocks — proof-of-concept result (2026-07-12)

## One-line result

A boundary-conditioned, **exact-likelihood, E(3)-equivariant continuous flow** demonstrably **restructures**
an autoregressive (AR) glass-cavity block toward the correct amorphous basin — the thing every prior
*deterministic* corrector provably could not do — validated on held-out chains by the full distribution of an
inherent-structure metric. It is a **proof of concept, under-trained**: it fixes the coarse basin/relative
structure but not the fine declash. Plot: `gate_structure_dist.png`. Population: `gate_structure_pop.pt`.

## The problem it attacks

Large-block (K>=8) moves are the bottleneck for basin-crossing in the 3D Kob-Andersen cavity point-to-set
(PTS) campaign. AR-generated blocks are clashy because of **half-cage exposure** (each block particle is placed
blind to the not-yet-placed block particles), which bakes in the *wrong relative structure*, not merely
overlaps. Measured earlier (`deterministic-corrector-refuted`): even a full L-BFGS relaxation of an AR block
floors at a strained +6.9/particle inherent structure, vs +0.9 for a real basin — and *no deterministic map*
(relaxation, coupling-flow MLE, OT-transport, anchor-lookahead — five variants) can cross that gap, because
descent removes overlaps but cannot *restructure*.

## The approach (Option B)

An EGNN flow as an exact corrector composed on the frozen AR base:
`AR block sample x0 -> cage-conditioned EGNN velocity field, integrated x0->x1 (dopri5) -> corrected block`,
with composed log-q `logq_AR(x0) + integral -div v dt` (carry-the-latent, forward-only, no inversion).

Why it clears all four axes that killed the prior correctors:
1. **Trained by TRANSPORT (conditional flow-matching, AR-block->data-block, minibatch-OT), not composed-MLE**
   — dodges the frozen-AR-base non-smooth log-prob gradient that killed the coupling-flow corrector.
2. **Full-cage equivariant message passing** — the non-causal placement the AR lacks (the ingredient memory
   credits for g_BB 1.25->2.00 in `full-cage-lever-needs-energy`).
3. **A LEARNED field can restructure** — deterministic descent cannot.
4. **AR->data transport is SHORT** (AR gets species + coarse structure right) — sidesteps the uniform-base
   hard-core expressiveness wall.

## The build (7-task SDD, Tasks 1-6 executed; exactness front-loaded)

| task | deliverable | hard gate |
|------|-------------|-----------|
| 1 | `CavityCondEGNN` (3D isolated velocity + analytic divergence) | divergence vs autograd trace **2e-8** (double) |
| 2 | `CavityBlockFlow` (dopri5 augmented ODE + composed log-q) | round-trip **9e-8**, logdet cancel **4e-7**, composed-logq exact |
| 3 | `sample_corrected_block` (AR base wiring) | identity-flow reproduces AR **exactly (0.0)** + permutation-equiv **1e-13** |
| 4 | flow-matching data + per-block Hungarian OT | valid species-matched bijection; mode-coverage judged acceptable |
| 5 | FM training (chain-level zero-leakage split, early-stop) | overfit smoke 0.20->0.0074; full run held_fm **0.093**, no overfit |
| 6 | **structure-fix gate (make-or-break)** | **distributional PASS** (below) |

Bug found + fixed en route: a **latent sign error in the shared `egnn_traceable` core** (the isolated `L=None`
branch returned the divergence of `-vel`; unexercised before this — every prior caller was periodic). Fixed,
periodic path byte-identical, isolated now exact to 2e-8, with a permanent regression test.

## The result — read from the DISTRIBUTION, not a scalar (per the evaluate-distributions directive)

80 held-chain blocks (10 cavities x M=8), K=8, pos_temp=0.4, inherent-structure floor/particle:

- **Floor distribution:** AR is bimodal — a cluster near data + a fat *strained tail* to +15-20. FLOW
  **collapses that tail and tightens toward the data basin**: AR med +0.18 (IQR [-0.68,+3.28]) -> FLOW med
  -0.45 (IQR [-0.71,+0.50]) vs DATA -0.73. **62% of FLOW chains land within 0.5 of the data floor (vs 41% AR).**
  (The median alone badly understated this — it can't see a tail; this is the whole point of plotting the
  distribution.)
- **Before->after / improvement-vs-strain:** nearly every chain below y=x; **38% improve (>0.2), only 7%
  worsen**; the correction *scales with how strained the AR block is* (structure-dependent, strongest where
  needed) — one catastrophic outlier at +140 it could not fix.
- **Clash distribution shifts left** (FLOW 0-0.3 vs AR 0.25-0.65); **g(r) moves toward data** (less sub-0.8
  core density); **no mode collapse** (cross-chain diversity up, 0.5->0.7). g_BB under-sampled at K=8 (~1-2 B
  per block) — noisy, not conclusive.

## Honest caveat

The flow fixes the **coarse basin / relative structure** but NOT the **fine declash**: raw block energy stays
high (the FM loss plateaued at 0.093, not ->0). So this first **64-wide / 5000-step** model is a proof of
concept — structure ✓, deployment-clean ✗. The improvement-vs-strain plot shows headroom on the hardest
blocks, i.e. the limitation is capacity/training, not the architecture.

## Significance

Per the literature scan (`Jung-Biroli-Berthier 2024` flows≈PT<swap; `Galliano-Coslovich 2024` policy-MC
limited on KA; `Ghio-Krzakala-Zdeborova 2024` sampling-hardness theory), ML-for-glass work splits into bulk
Boltzmann sampling (loses to swap) and dynamics *prediction* (no sampling). **Boundary-conditioned,
exact-likelihood generation of constrained glass ensembles (cavities/PTS) appears unclaimed.** This result is
a concrete first step in that niche: an exact equivariant flow that restructures constrained glass
configurations, with the exact log-q the MTM/IS/free-energy program needs.

## Deferred (scale-up path, NOT done)

1. **Scale the flow** (wider/deeper, more steps, more data) to drive FM loss down and close the fine-declash
   gap; re-run the distribution gate. OR **coarse-flow + fine-polish** (flow fixes basin, a short exact
   energy-descent does the final declash).
2. **Task 7 deployment:** integrate the cage-truncation into `sample_corrected_block` (Task 3 predates the
   fixed-size scheme — currently the gate replicates the truncation externally); wrap flow moves in
   try/except -> reject on the dopri5 `max_num_steps` AssertionError (+ a wall-clock budget); then the
   frontier benchmark vs the `MSEED_MOVE` 4/12 baseline at R=2.5-3.5, and cavity free energies via the exact
   composed log-q.

## Reusable assets

`liquid_coupling_flow/ka3d_cavity_egnn.py` (exact 3D isolated flow + composed log-q, all gated);
`reports/logs-2026-07-12/{fm_data,train_cavity_egnn_flow,gate_structure_dist}.py`; the fixed shared-core
divergence sign bug; the trained checkpoint `artifacts/ka3d_cavity_egnn_flow_best.pt`; the 6-task exactness
test suite.
