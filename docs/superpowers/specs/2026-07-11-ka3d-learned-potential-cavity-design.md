# 3D learned-potential cavity infiller — design

**Date:** 2026-07-11
**Status:** design (autonomous build to follow)
**Predecessor:** 2D validation in `ka_localframe_ebm.py` (memory: learned-potential-embedding) —
a general learned pairwise potential tilting the AR proposal lifted block-MTM acceptance (k=4 30->85%,
more trials now convert to acceptance) and raised every species-pair g(r) first peak toward data incl
g_BB, staying provably exact (sampler logq == scorer 0.0). This design ports that to 3D and targets the
actual PTS use: **cavity infilling** — regenerate a cavity interior (or a block of it) given a frozen
pinned boundary.

## Goal
An exact, size-transferable, boundary-conditioned 3D block-MTM proposal for KA cavity interiors whose
learned pairwise potential builds hard-core exclusion against BOTH retained interior and boundary
particles, uniformly. Train at N=100-scale cavities, zero-shot at larger radii.

## Two user directives this design answers
1. **"Boundary particles treated similarly as surrounding particles."** The learned potential
   `V_theta(x|cage) = sum_{i in cage} phi_theta(|x - x_i|, s_j, s_i)` sums over `cage = [boundary UNION
   placed interior]` with NO kind flag — boundary atoms are ordinary terms. The potential needs no
   ordering and no scaffold anchor (unlike the AR context), so the boundary integrates trivially. This
   is the clean unification the original two-stream `_frame_context` hack only approximated.
2. **"Maybe the radius should be embedded somehow."** `_slot_features` already emits
   `[slot_frac, |anchor|/R, R/2.5]`. The design (a) keeps that, (b) enriches `R/2.5` -> a small Fourier
   /MLP embedding of R added to the query, and (c) makes R-transfer an explicit validation gate (train
   multi-radius, zero-shot held-out radius). The potential itself is size-invariant (pairwise, local) and
   needs no R.

## Architecture

### Base
Subclass `KA3DScaffoldCatAR` (proven frameless Cat3Head over 3 local-frame axes a,b,c; fixed
Fibonacci ball scaffold; Morton AR order; soft ball map). Reuse `_contexts`/`_frame_context`/`_origins`
/`_slot_features` unchanged for the base logits and the frame `Rf`, origin.

### Potential tilt (the new part)
Tilt the FINAL axis `c | a,b` by the true 3D potential, exactly as 2D tilted `b|a` (bulk gain; the 2D
a-tilt added only ~0.05 nat, so start final-axis-only; add b/a tilts only if the c-tilt underperforms):

    logits_c(bin) = head_c(h + e + emb_a(ba) + emb_b(bb))  -  V_c(bin)
    V_c(bin) = sum_{i in cage(slot)} phi_theta(|x_cand(ba,bb,bin) - x_i|, s_j, s_i)
    x_cand(ba,bb,c) = origin + Rf^T . (a_ctr(ba), b_ctr(bb), c_ctr(bin))   # un-rotate local->physical

- `phi_theta`: MLP on `RBF(distance) (+) pair_emb(s_j, s_i)` -> scalar. NO LJ, sigma, or eps. Learns the
  potential-of-mean-force. `n_rbf~12` over [0, rbf_max~3].
- `cage(slot)`: the `knn_pot~8` nearest of `combined[valid_row]` to the slot (boundary + placed
  interior, pooled with NO kind distinction). Reuses distances already computed in `_frame_context`.
- Normaliser is O(n_bins) per particle (one c-column), so training is tractable (chunk over slots).

### Exactness
Any per-bin tilt is a valid categorical, so `sample()` logq == `log_prob()` scorer to fp precision,
independent of what phi learns and independent of the (deterministic, frozen) boundary. Gate before any
training: 0.0 with phi~0 warm start, reproducing the un-tilted cavity model.

### Radius embedding
`self.R_embed = MLP(fourier(R))` added to the query token alongside `slot_proj(slot_feat)`. Keeps the
existing per-slot `|anchor|/R` and `slot_frac`. Rationale: lets the base anchor-prior tighten near the
wall and enables interpolation across cavity radii for zero-shot transfer.

## Cavity block-MTM
Extend `ka3d_block.py`'s [retained; block] reordering so the block's cage includes the frozen boundary:
the potential and context both pool `[boundary; retained interior; earlier block]`. Block move regenerates
a spatial blob of interior conditioned on everything else; exact independence-MTM (Liu-Liang-Wong
reusable-batch) as in 2D.

## Training
- Data: carved cavities via `ka3d_cavity_carve.carve(X, S, center, R, L)` from the N=512 dataset (and/or
  the PT reference), MULTIPLE radii (e.g. R in {2.0, 2.6, 3.2}) to force R-generalization.
- Warm-start the shared parts from an existing boundary-trained scaffold checkpoint if available (else
  train the base first); phi init ~0 so the model starts == the base, then learns the tilt.
- Loss: -log_prob_pair / n (block-conditional MLE), same as base.

## Validation gates (in order)
1. **Exactness:** sampler logq == scorer 0.0 (phi~0 and trained).
2. **Conditional NLL:** EBM < base (the potential adds conditional likelihood, as in 2D 0.76->1.33).
3. **Structural g(r):** species-resolved block-centered g_AA/g_AB/g_BB vs TRUE cavity blocks; first peaks
   rise toward data, esp g_BB. Include a BOUNDARY-decomposed clash split (block<->boundary vs
   block<->retained vs block<->block) — the decomposition the free-cluster model could not give.
4. **Block-MTM acceptance:** K-sweep vs the un-tilted cavity base; more trials convert to acceptance.
5. **Size-transfer:** train multi-radius, zero-shot a held-out radius; gates 1-4 hold (no cliff).

## Files
- `liquid_coupling_flow/ka3d_scaffold_ebm.py` — `KA3DScaffoldEBM(KA3DScaffoldCatAR)`: potential net,
  `_V_c`, overridden `log_prob_pair`/`sample_pair`, `R_embed`, block methods.
- `reports/logs-2026-07-11/train_ebm3d_cavity.py` — multi-radius warm-start fine-tune.
- `reports/logs-2026-07-11/test_ebm3d_cavity.py` — exactness gate + K-sweep + boundary-decomposed clash.
- `reports/logs-2026-07-11/struct_ebm3d_gr.py` — species-resolved cavity-block g(r).

## Open levers / risks
- If final-axis c-tilt underperforms 2D's gain, add b|a and a tilts (2D machinery already generalizes).
- If R-transfer shows a cliff, enrich R embedding or widen the training radius range.
- The base cavity context (two-stream) is kept as-is; if the potential makes it redundant, a later
  simplification is to fold boundary into the single interior stream (unify context too, not just the
  potential) — deferred; not on the critical path.
