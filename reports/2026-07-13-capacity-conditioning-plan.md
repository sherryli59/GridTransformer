# Anchor-free capacity/demand conditioning — staged plan

**Goal.** Tame the AR overstretch (early block members over-consume a zero-slack region; damage is
upstream, invisible until the cage fills — see `ar-overstretch-compounding` memory). Give the model the one
thing it lacks: **remaining demand + this candidate's effect on future feasible volume.** Motivated by the
valid oracle (+1.42 nats of early-placement sharpening available given the full cage) and by virtual_tail
being NULL (a relabel does nothing — the model only responds to signals it is TRAINED on).

**EXACTNESS INVARIANT (non-negotiable, gates every stage).** Every new feature depends ONLY on: the visible
causal prefix, the frozen boundary, the block definition (which slots, K), and remaining species counts.
It is recomputed IDENTICALLY in `sample_block_b` and `block_log_prob_b`. TRUE future particle coordinates
never enter. Gate: `|logq_sample - logq_score| < 1e-2` (float32) on a K=8 blob roundtrip, every stage.

**Warm-start.** `ka3d_cavity_ebm3ax_rho115_blockcond_knn24_best.pt` (the 50x-energy block-cond FT). One new
signal per stage (attribution). Zero-init every new output projection so warm-start reproduces the ckpt.

**Primary metric.** Not "do late ranks notice the cage is broken" — but "do ranks 1–4 shift *slightly* to
keep ranks 7–8 feasible." Track: rank-resolved clash-creation curve (esp. whether the rank-8 spike drops
AND ranks 1–4 rise a little), clash/particle, dE/particle, held NLL (blob8 + allmask base-guard), 5-seed
MTM K-ladder. Plus the exact-roundtrip gate.

## Stages (simple → complex)

**Stage 1 — remaining-demand embedding (START HERE).** Feed the heads a per-slot demand vector from the
already-tracked remaining block counts: `[rem_A/K, rem_B/K, rem_tot/K, K/n, n/R^3]` (species matters: small
B fits gaps A cannot). Cheap, exact (rem is tracked in sample, cumsum in score), anchor-free. = the user's
signal #1 / arm "counts only". Implement as a zero-init MLP added to `h`. FT + measure.

**Stage 2 — probe free-space capacity tokens.** Deterministic probe cloud; per probe, species-dependent
clearance `c_s(z)=min_i |z-x_i|/sigma_{s,s_i}` over the visible prefix+boundary. Pool into 16–32 capacity
tokens (radial shells / sectors: est. A- and B-capacity, largest-empty-sphere, clearance quantiles, global
margin = est_capacity − rem). Cross-attend (zero-init). = arm "counts + probe". Exact (prefix-only).

**Stage 3 — candidate capacity-destruction penalty.** `D(x,s)=C(prefix)−C(prefix∪(x,s))` — how much usable
suffix capacity this placement destroys. Add `−λD` to the position logits on ALL THREE axes a,b,c (c-only
is too late — the a/b plane already commits the bad placement, per the exclusion experiment). Deterministic
first (probe-clearance based), then optionally learned.

**Stage 4 — fuzzy/warped future-anchor tokens.** Future scaffold anchors as BROAD occupancy (width ~0.6–0.7
from observed data offsets), NOT hard spheres. Feed density/gradient/anisotropy; warp future anchors by a
local deformation field estimated from observed `x_i − a_i` residuals; carry uncertainty. Answers "where is
future demand likely" while allowing liquid rearrangement. = arm "counts + capacity + fuzzy anchors".

**Stage 5 (stretch) — suffix-survival value head.** `V(prefix, n_rem) ≈ Pr(clean completion)`; tilt logits
by `V(prefix+x, n_rem−1) − V(prefix, n_rem)`. Learned reservation, anchor-free. Labels from short
rollouts / SMC-repaired completions. Most general, most expensive — only if 1–4 plateau.

## Paired comparison (after Stages 1–4 exist)
K=8, matched cavities/seeds: (1) counts only; (2) counts + fuzzy anchors; (3) counts + probe capacity;
(4) counts + capacity + fuzzy anchors. Judge on the primary metric above.
