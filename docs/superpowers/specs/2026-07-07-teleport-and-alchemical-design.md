# Teleport pricing + alchemical (x,lambda) species kernel — design spec

**Date:** 2026-07-07 (user-approved queue: teleport pricing -> alchemical prototype -> 3-arm crossover;
flow-drift upgrade slotted behind the alchemical gate).

## 1. Teleport move — the distilled core of variable-membership blocks

Thinking the grand-canonical block through, its irreducible primitive is a PARTICLE TELEPORT: delete particle i,
re-insert elsewhere via a learned insertion density, MH with two-way densities. Both collective-move walls
vanish: selection = particle index + target slot (uniform, symmetric); k=1 => no precision compounding. The
insertion density ALREADY EXISTS: A2's beta-conditioned full-cage conditional (DB-exact at any N).

**Kernel** (`ka_teleport.py`): slot-ordered state; pick mover slot a and target slot b uniformly (symmetric).
- Densities on the FROZEN rest-state y = x \ {i}: q_fwd = A2(x' | cage_b, species_i, beta), q_rev =
  A2(x_i | cage_a, ...). A2's fixed-N interface can't delete a particle, so EXACTNESS requires the mover to be
  absent from both evaluation cages: **require a not-in frame_ctx_slots(b) AND b not-in frame_ctx_slots(a)**
  (deterministic in (a,b) => symmetric abort — "non-overlapping neighborhoods," legitimately this time).
  ~40% aborts at N=100 (24/100 context fraction per direction), fewer at larger N.
- alpha = min(1, e^{-beta dU} q_rev/q_fwd), dU via ka_pair_row (one mover; verified == ka_energy).
- OCCUPANCY note: teleports RELABEL slots massively (mover crosses the box) — but slot order here is only a
  bookkeeping frame for A2's cage lookup, evaluated per state consistently in each direction; the selection
  (slot pair) is symmetric because slots are geometric. The move itself is defined on particle i's position.

**Pricing gate (the point):** two-way acceptance at (i) equilibrium cold rung, (ii) the stuck A2-stack
population (density inhomogeneities = where pockets should exist), (iii) mid-anneal (beta~1.2). Expectation
honest: equilibrium ~0 (no pockets in a glass); the bet is (ii)/(iii). Verdict thresholds: >0.5% anywhere
interesting => wire into the SMC stack; ~0 everywhere => grand-canonical direction closes at the pricing gate.

## 2. Alchemical (x,lambda) species kernel — pathwise species change (NCMC)

Plain A<->B swaps are ENDPOINT moves and are measured dead (0/102,400) — the sigma_AB mismatch can't be paid
in one step; the oracle taught that paths beat endpoint jumps. Construction (classical, no learning):
- **Extended potential:** per-particle lambda_i in [0,1]; PAIR parameters by bilinear interpolation over the
  KA 2x2 tables (non-additive sigma_AB=0.8 handled exactly at corners):
  sigma(li,lj) = (1-li)(1-lj) s_AA + [li(1-lj)+lj(1-li)] s_AB + li lj s_BB (same for eps). Corners == binary KA.
- **Kernel = NCMC alchemical swap** (Crooks/annealed-candidate MC, exact): pick an unlike pair (i,j); drive
  (li,lj): (1,0)->(0,1) over T increments, each followed by a few position-MALA relaxation steps at fixed
  intermediate lambda (the cage breathes DURING the exchange — the pathwise advantage over SB's one-shot
  resample); accumulate protocol work W; accept the whole trajectory with min(1, e^{-beta W}). Composition
  exactly preserved (antisymmetric pair drive). Endpoints exactly binary => no reweighting needed.
- **Gate:** accepted exchanges per GPU-second vs swap-and-breathe's v2 (12.4x baseline) at beta=2 equilibrium;
  scan T (path length) — T=1 must reproduce ~plain-swap-dead as a sanity anchor.
- **Flow-drift upgrade (behind this gate):** replace/augment the x-relaxation MALA drift with the EGNN flow
  (equivariant global drift, density-free role — MALA/NCMC needs only gradients/work, never flow densities).

## Order & budget
Teleport build+price: ~2h build (TDD) + <1h runs. Alchemical: potential core + corner tests now; NCMC kernel
next; gate ~1h runs. Both queue behind PT2 (GPU frees ~21:30). Crossover (3-arm, N in {100,256,576}) after,
reusing PT2 arms + N=576 pre-checks. Records per results-durability rules.
