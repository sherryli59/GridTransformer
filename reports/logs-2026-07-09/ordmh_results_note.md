# Ordered-space independence-MH: exactness + efficiency probe

**Date:** 2026-07-09  ·  **Code:** `liquid_coupling_flow/ordmh/`  ·  **System:** 2D single-species WCA liquid, ρ\*=0.5, T\*=1 (β=1), σ=ε=1, r_c=2^(1/6)

Minimal, independent, correctness-first check of whether ordered-space
independence-MH targeting π̃ ∝ e^{−βU} samples the physical Boltzmann
distribution, and of how its efficiency scales with N. The correctness test does
**not** rely on a good generator — a deliberately-mediocre proposal is used — so
"is the ordered-space MH correct" is isolated from "is the generator useful."

The whole probe is **pure numpy**, sharing no code with the torch campaign; the
energy is the only physics shared between reference and sampler (as it must be),
and it is verified against three independent implementations before use.

---

## (i) Gates

| Gate | What it checks | Result |
|------|----------------|--------|
| **Task 0** | WCA energy vs 3 independent oracles (analytic points; numpy min-image vs explicit-periodic-images; repo torch `lj_energy`) | **PASS** (machine precision) |
| **Gate R** | independent ground truth self-consistent: reference MC ⟨U⟩ vs **grid quadrature** ⟨U⟩ | **PASS** |
| **Gate E** | ordered-space MH (with the **bad** proposal) reproduces the independent reference | **PASS** |

**Gate R (ground truth is trustworthy).** Two methods sharing no code agree:

| N | grid quadrature ⟨U⟩ (non-MC bedrock) | displacement-MC ⟨U⟩ | \|diff\| | tol |
|---|---|---|---|---|
| 2 | 0.24100 (halving err 3e-5) | 0.23998 ± 0.00088 | 0.00102 | 0.00441 |
| **3** | **0.32595** (halving err 1.3e-4) | **0.32566 ± 0.00056** | **0.00028** | 0.00290 |

The N=3 grid quadrature is fully analytic (a 4D integral, no MC) — the bedrock
number, agreeing with reference MC to ~4 significant figures.

**Gate E (the load-bearing correctness result).** Run with the deliberately-
mediocre toy proposal:

| N | ground truth ⟨U⟩ | ordered-MH ⟨U⟩ | raw proposal E_q[U] (no MH) | P(U) total-var | g(r) max\|diff\| |
|---|---|---|---|---|---|
| 3 | 0.32595 ± 0.00013 (grid) | 0.32492 ± 0.00095 | ~3.6×10³² | 0.0028 | 0.026 |
| 5 | 0.63242 ± 0.00093 (MC) | 0.62863 ± 0.00439 | ~3.6×10²⁸ | 0.0078 | 0.063 |

The **raw proposal draws give a wildly wrong ⟨U⟩** (it happily proposes
overlapping particles → ~10³² energy); the accept/reject step alone drags the
estimate onto the exact grid-quadrature value. This proves exactness comes from
the **MH construction, not the generator** — the point of using a bad proposal.
Figure: `ordmh_exactness.png` (P(U) and g(r) overlays; crimson MH sits on black
reference).

Implementation guard verified: the reverse term log q̃(x) in the ratio is the
**cached value carried on the current ordering**, asserted equal to a fresh
`log_q_ordered` re-evaluation each step (never re-lifted from a projected
config). The proposal's per-step conditional integrates to 1 (grid-checked) and
has full support (density ≥ w_u/A everywhere), so no reference config has zero
proposal density.

---

## (ii) Efficiency N-scaling (Task 4) — measured only after Gate E

Correctness is N-independent (it's a property of the MH ratio). Efficiency is
not. Figure: `ordmh_efficiency.png`.

| N | acc (antipode) | acc (uniform control) | ordering-std (nats) | ordering-std / N | β²Var(U) |
|---|---|---|---|---|---|
| 3 | 0.1025 | 0.0811 | 0.088 | 0.029 | 0.42 |
| 4 | 0.0384 | 0.0297 | 0.112 | 0.028 | 0.46 |
| 5 | 0.0162 | 0.0139 | 0.138 | 0.028 | 0.54 |
| 6 | 0.0107 | 0.0101 | 0.165 | 0.027 | 0.76 |
| 7 | 0.0093 | 0.0091 | 0.186 | 0.027 | 0.93 |
| 8 | 0.0090 | 0.0088 | 0.204 | 0.025 | 1.11 |

- **Acceptance** collapses ~exponentially: antipode ~ exp(−0.48 N), uniform ~
  exp(−0.43 N).
- **Ordering-mismatch** (within-config std of log q̃ over random orderings of the
  *same* Boltzmann config — the target is flat over orderings, so this is pure
  penalty) grows **linearly** in N; **per particle it is flat at ~0.027
  nats/particle** (even mildly decreasing).
- The **uniform control** (w_u=1, permutation-invariant) has ordering-std = 0 to
  machine precision (8.9e-16) — a built-in correctness check on the measurement.

**Decomposition — the key finding.** The ordering penalty is a *fixed per-particle
tax, not a wall*: total ∝ N (same extensive scaling as the irreducible energy
cost β²Var(U)), per-particle constant. Moreover it is **not the bottleneck**: the
antipode and uniform acceptance curves nearly coincide and *converge* (0.0090 vs
0.0088 at N=8), and the antipode is in fact *slightly better* at every N (its
crude excluded-volume bias saves more than its ordering handicap costs). What
actually collapses acceptance is that this is a **global independence proposal**
(whole config proposed at once), whose acceptance decays exponentially in N *even
for the permutation-invariant control* — the generic dimensionality curse of
independence MH, not an ordering artifact.

---

## (iii) Local-canonical-order tie probe (Task 5)

Could a locally-defined canonical order remove the ordering penalty? Only if
particles carry a near-unique intensive local scalar. Figure:
`ordmh_tie_probe.png`.

| N | local-energy frac tied (5% tol) | coordination# frac tied | max tie-class / N |
|---|---|---|---|
| 3 | 0.69 | 0.97 | ~0.5–0.7 |
| 5 | 0.72 | 0.78 | ~0.55 |
| 8 | 0.81 | 0.85 | ~0.5 |

70–85% of particles sit in ambiguous tie-classes (by local energy; 78–97% by
coordination number), and the largest ambiguous group is ~half the system. A
homogeneous liquid has **no cheap distinguishing local order** — so the
ordered-space penalty *cannot* be canonicalized away.

---

## (iv) Honest verdict

**Correctness:** ordered-space independence-MH targeting e^{−βU} **is exact** — it
reproduces the physical Boltzmann distribution (⟨U⟩, P(U), g(r)) against an
independent ground truth, *with a deliberately-bad proposal*, at N=3 (vs analytic
grid-quadrature bedrock) and N=5 (vs MC). The construction is sound.

**Efficiency:** the **ordering-specific penalty does NOT scale like a wall** — it
is a fixed ~0.027 nats/particle tax (linear total, flat per-particle) and is not
even the dominant cost. The tie probe shows it also can't be cheaply removed by a
local canonical order in a homogeneous liquid, but since the penalty is mild,
being stuck with it costs little here.

The real efficiency ceiling is **not** the ordering issue: it is that a *global*
independence proposal has acceptance decaying exponentially in N (the uniform
control collapses just as hard). The lever for scaling is therefore a **local /
incremental** ordered-space move (propose a few particles at a time), which side-
steps the global-independence dimensionality curse while inheriting the exact
ordered-space construction validated here — not a fight against the ordering
penalty, which is benign.

**Caveat / scope.** These conclusions are for an ergodic single-species liquid at
ρ\*=0.5, T\*=1. In a denser or glassy system the ordering spread would be larger
and the tie problem worse; the benign ordering verdict should be re-measured
there before being relied on.

**Trust hierarchy:** the sharpest correctness result is the **N=3 grid-quadrature
agreement** — fully analytic, independent of all MC. Everything else (reference
MC) is independent-in-practice but still Monte Carlo.

---

### Files

Code (`liquid_coupling_flow/ordmh/`): `ordmh_energy.py`, `ordmh_reference.py`,
`ordmh_toy_proposal.py`, `ordmh_sampler.py`, `ordmh_efficiency.py`,
`ordmh_tie_probe.py`, `ordmh_exactness_report.py` + tests `test_ordmh_*.py`.
Logs/figures (`reports/logs-2026-07-09/`): `gate_R.out`, `gate_E.out`,
`task4_efficiency.out`, `task5_tie_probe.out`, `ordmh_exactness.png`,
`ordmh_efficiency.png`, `ordmh_tie_probe.png` (+ `*_data.pkl`).
Reference artifact (regenerable, gitignored): `ordmh/artifacts/ordmh_reference.pkl`.
