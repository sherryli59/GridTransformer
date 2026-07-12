# Deterministic position-corrector for large-K KA cavity block moves — DECISIVE DIAGNOSIS (2026-07-11)

Goal: make the learned cavity generator propose **clean, low-energy large-K (K=12) block rearrangements**
so a cold, exact MTM jump can cross glassy basins (local K=4 moves can't). Constant constraint: EXACT log_q.

## Chain of measured findings (all on ka3d_cavity_ebm3ax base, N512 T0.5 dataset, K=12, M=8-16 cavities)

### 1. Soft-core relaxation + exact logdet: BOTH achievable **on the proxy**
`sweep_relax_potential.py`. Soft-core power law `w(s^2/(r^2+a^2))^p` declashes the min-distance metric
40%->0% below the 29% full-cage floor, diversity survives, and the per-step Jacobian logdet
`sum log|det(I-eta*H)|` matches finite-difference to <1% for gentle configs. Force is monotone-in-1/r
(unlike a Gaussian, whose force vanishes at r->0 and can't separate the worst overlaps).

### 2. But the proxy is the WRONG metric — TRUE KA energy stays garbage
`test_relax_energy.py`. Under the real `ka_energy`, soft-core declash leaves the block at **+190/particle**
above equilibrium (raw AR +2e15). Reducing min-distance clash does NOT reduce `r^-12` energy: a pair the
soft core parks at r~0.7-0.8 still costs enormous true energy. Also the "safe" net-contracting config
**folds** (min per-step eig -9.3): net contraction does not imply every `I-eta*H > 0`.

### 3. The defect is AR POSITIONS, not species
`test_species_vs_position.py` (median dE/block-particle above equilibrium, L-BFGS inherent-structure floor):
```
A  AR pos   + AR species    +105
B  AR pos   + DATA species  +107   swapping in data species does NOT help
C  DATA pos + AR species     +2.25 AR species + good positions = FINE
D  DATA pos + DATA species   -0.68 control
```
Row C is the tell: **AR species are fine.** The frustration is entirely in AR positions.
(Earlier "species/topology" read was an L-BFGS-trapping artifact.)

### 4. The AR position tangles are SLIDABLE (not barrier-crossing) — deterministic reaches near-basin
`test_untangle.py` (median dE/block-particle; good basin ~+2):
```
(1) LBFGS direct                +73.6  stuck (can't untangle overlaps)
(2) softcore-declash -> LBFGS    +6.9  deterministic slide-apart unsticks it
(3) langevin-anneal  -> LBFGS    +6.6  stochastic barely better => NOT topological overlaps
 C  DATApos + ARspecies          +0.9  the good basin
```
(2)≈(3) => the particles are merely too-close, not interlocked => a deterministic map CAN untangle.

### 5. BUT a fixed-step (exact-able) flow cannot do the hard phase, AND the floor is still strained
`test_anneal_flow.py`. A fixed-step soft->hard homotopy (a: 0.7->0.15, p=6->true LJ):
```
flow-alone (deployable, exact-able)  +74.7   worse than +6.9
INJECTIVITY min per-step eig         -2.9e7  catastrophic FOLDS at the hard end (+ numerical blow-up)
```
The +6.9 result depended on **L-BFGS's line search** taming the stiff descent — and L-BFGS's Jacobian is
intractable, so it can never be part of an exact-log_q proposal. Fixed-step forward Euler on `r^-12` folds
and blows up. Only backward-Euler (implicit) could stay stable/injective.

## Verdict

A **deterministic position corrector on AR samples is refuted** for clean large-K basin proposals:
- (a) the exact-able forward relaxation cannot reach true-LJ-low without folding (would need a stiff implicit
  backward-Euler solve per step — heavy, unproven to converge from clashy starts), and
- (b) even the best (L-BFGS-assisted) untangling floors at **+6.9/particle strained**, not the +0.9 of a real
  basin — because the AR block's **relative structure is wrong** (half-cage generation), and relaxation fixes
  overlaps, not structure. +6.9/particle x K=12 x beta=2 => cold-MTM accept ~ e^{-166} ~ 0.

Root cause = the AR base generates block **positions** with intra-block half-cage blindness => wrong relative
structure. Consistent with campaign memory ("full-cage conditional gives clean g(r) but iterating alone
collapses"; "full-cage needs an energy guard").

## Implied next fork (NOT yet chosen)
- **A. Full-cage position proposal** (non-causal / iterative-refinement generator that sees ALL K block
  particles => correct relative structure), deployed as an **SMC/MH energy-guarded** block proposal (the guard
  supplies exactness and prevents the pure-iteration collapse). The "contractive denoiser / coupling flow"
  path from memory.
- **B.** Give up the one-shot cold jump; use the block move as an annealed SMC mutation (user has resisted
  leaning on classic enhanced sampling).
