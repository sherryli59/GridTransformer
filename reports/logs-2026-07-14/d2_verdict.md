# D2 verdict: v10 canonical-lift shift-spread (re-rooted-suffix gate)

**Bank used:** `mw_bank_ambient_N64.pt` (T\*≈0.096, L=5.1953, 64000 configs, first 128 used) —
the supercooled bank (D0) was not ready when this ran; **this column is ambient-only**. Re-run
against `mw_bank_supercooled_N64.pt` once D0 lands before treating this as final for the
supercooled regime.

**Setup:** checkpoint `mw_gen_N64_v10_best_nll.pt`, `model.log_prob(x, L)` canonical lift.
128 configs × 32 independent `U[0,L)^3` torus shifts each (identity shift included as draw 0),
plus 128 configs × 8 draws combining an independent torus shift with a random O_h element
(signed coordinate permutation about the box center, `mw_generator._augment_batch` convention).

## Numbers (nats)

| | std median | std p90 | range median | range p90 |
|---|---|---|---|---|
| torus shift (M=32) | 8.749 | 11.228 | 37.756 | 51.856 |
| torus shift + O_h (M=8) | 9.078 | 12.386 | 25.520 | 37.340 |

Full per-config distributions: `d2_shift_spread.pt` (`lp` [32,128], `lp_oh` [8,128], plus the
std/range tensors). Histograms: `d2_shift_spread.png`.

Sanity: no NaN/inf in either `lp` or `lp_oh`; `log_prob` values range roughly -77 to +13
(unbounded-above continuous density, not a truncation artifact) — the spread is a real signal,
not numerical blow-up.

## GO/NO-GO

Criterion: GO iff median range ≲ 2 nats (comparable to per-move MH slack).

**Measured median range 37.8 nats (torus), 25.5 nats (torus+O_h) — roughly 19–25× the ~2-nat
bar. NO-GO** for a re-rooted-suffix MH move on v10 as-is (ambient-bank column).

## Read

v10's canonical lift is not close to shift-invariant. This is the expected consequence of an
AR model whose causal ordering is keyed to *absolute* box coordinates (curve/cell indexing) —
MEMORY `curve-conditioning-blocks-transfer` already established that this generator's
conditioning is curve-specific, not geometry-invariant. A torus shift permutes which particles
fall into which curve cell, which reshuffles the entire prefix/suffix structure the AR model
conditions on; the resulting log-density swing (tens of nats) dwarfs a single-particle MH
acceptance budget. An O_h element compounds this (comparable std, smaller max range only
because the M=8 sample is thinner than M=32).

**Concerns / caveats:**
- This is the ambient-bank measurement only; T*≈0.096 is not the target supercooled regime.
  The re-rooted-suffix move is more relevant near the target's hard-state regime, and spread
  could differ there (denser/more structured configs might have sharper or softer curve
  sensitivity) — D0's supercooled bank re-run is the pending confirmatory step, but a 19–25×
  overshoot leaves little room for the number to close.
- Sample sizes are small (M=32 torus, M=8 O_h) relative to the tails needed to trust p90/max
  precisely, but the median alone already fails the gate by more than an order of magnitude,
  so this doesn't change the verdict.
- Per brief: the O_h sampling is "for free" / best-effort, not the required part; it was
  completed within the time-box and corroborates the torus-shift NO-GO (same order of
  magnitude).

**Conclusion:** do not build the re-rooted-suffix kernel variant against v10 in its current
(curve-anchored) form. Fixing this would require a geometry-invariant canonical lift (e.g. the
`ka_localframe`-style local-frame conditioning noted in MEMORY), which is out of scope for this
diagnostic.
