# Stage 1 (audit) findings — K=1 rail size transfer

Audit script: `analyze_k1_rail_transfer.py` (deterministic; reproduced twice, identical).
Rail-input KS is reported on the L-normalized (fractional) waypoint, the model-visible
feature. `in_box` is the **raw decoded Hilbert cell** in-grid fraction (leakage diagnostic),
not the min-imaged waypoint (which is always ~1.0).

| strategy | N=64 rail_frac KS | N=125 rail_frac KS | min in_box (raw cell) | target-delta KS (64 / 125) | verdict |
|----------|-------------------|--------------------|-----------------------|----------------------------|---------|
| pow2     | 0.3000            | 0.1633             | 1.000                 | 0.0188 / 0.0234            | borderline |
| exact    | 0.3367            | 0.2867             | 0.556                 | 0.0362 / 0.0300            | disqualified (leakage) |

Per-size detail:

| strategy | size | R | cell | in_box |
|----------|------|---|------|--------|
| pow2  | N=27  | 64  | 0.0469 | 1.000 |
| pow2  | N=64  | 128 | 0.0313 | 1.000 |
| pow2  | N=125 | 128 | 0.0391 | 1.000 |
| exact | N=27  | 64  | 0.0469 | 1.000 |
| exact | N=64  | 85  | 0.0471 | 0.556 |
| exact | N=125 | 107 | 0.0467 | 0.653 |

## Interpretation

- **exact (non-pow2 R) is disqualified.** Holding cell_size exactly constant forces R=85/107,
  but with `bits=ceil(log2(R))` the Hilbert curve fills a 128³ cube while `X=R³//N` assumes an
  85³/107³ curve. 44% (N=64) / 35% (N=125) of waypoints decode to cells outside the physical
  [0,R) grid — the rail feature is curve-length-corrupted. in_box 0.556 << 0.95 threshold.
- **pow2 is the only viable strategy**, and it is **borderline**: N=64 rail_frac KS = 0.3000
  (exactly at the 0.30 go/no-go boundary), N=125 = 0.1633 (healthy). No leakage.
- **The bottleneck is the rail INPUT, not the target.** Cartesian-delta target KS < 0.024 for
  pow2 — what the model actually generates transfers cleanly. Only the conditioning waypoint
  distribution mismatches, and worst at N=64 (pow2 rounds L/cell_train 85→128, pushing cell to
  0.031, furthest from training 0.047). N=125 rounds 107→128, a gentler shift (cell 0.039).

## Decision

**Chosen strategy for Stage 2:** `pow2`
**Reason:** exact is disqualified by raw-cell leakage (in_box 0.556 < 0.95); pow2 has no
leakage and the target transfers cleanly. N=64 rail KS sits exactly on the boundary.

**Stage 1 go/no-go:** GO to Stage 2 (marginal). The strict gate (rail_frac maxKS < 0.30) is
not cleanly met at N=64 (=0.30), but the failure is borderline, isolated to N=64's pow2
rounding, and absent from the target distribution. Since zero-shot generation is cheap and
its g(r)/OTgap is the ground truth that resolves a borderline KS, proceed and let the samples
decide. **Watch N=64 as the at-risk size; expect N=125 to transfer at least as well.**
