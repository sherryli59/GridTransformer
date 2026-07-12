# Full-cage cheap-gate VERDICT (2026-07-11) — option B has no payoff on the N=512 dataset

Three-part oracle gate (no training; oracle = per-site heat-bath under the TRUE KA energy with teleport
candidates = the ceiling of any learned full-cage position model). Scripts + logs in this directory:
`test_fullcage_oracle_gate.*`, `test_oracle_rscan.*`, `test_oracle_headstart.*`.

## Findings

1. **Full-cage placement + energy guard WORKS mechanically** (R=2.0, K=12 blob): AR garbage (+3e9/particle)
   → +1.87/particle in 6 sweeps, guard-survival 74%, data-start control holds equilibrium (+0.08).
   Confirms the `full-cage-lever-needs-energy` thesis at the 3D cavity level.

2. **But every measurable cavity is PTS-PINNED.** Rearrangement vs the data arrangement: 2% at R=2.0,
   5-7% at R=2.5, data-start intrinsic 0%. Single basin — consistent with xi_PTS≈3.8 (Berthier 2016) at
   T=0.51: for R<xi the boundary admits ONE amorphous arrangement. **There is no basin-crossing problem at
   these radii** — the "K=4 can't cross basins" difficulty reframes as slow within-basin relaxation, which
   the teleport heat-bath solves outright.

3. **Hard geometric cap:** the N=512 box (L=7.528) allows only R ≤ (L−r_cut)/2 = **2.51** (periodic-cavity
   validity). The paper's overlap-decay regime (R ~ 3–6, where multiple basins exist and a learned
   basin-proposer could matter) is **physically unreachable on this dataset**.

4. **AR seeding adds almost nothing over uniform seeding** for the oracle sampler at the max measurable
   radius (R=2.5, whole-interior resample, 12 sweeps, 5 cavities):
   ```
   sweeps-to dE<3:   AR 4   UNIF 5    (saves 1)
   sweeps-to dE<2:   AR 8   UNIF 10   (saves 2)
   sweeps-to dE<1.5: AR 13  UNIF 13   (saves 0)
   ```
   1–2 sweeps of a ~74-site heat-bath ≈ nothing — the LJ-liquid-line lesson
   (`liquid-coupling-flow-pivot`: flow head-start ~1-2 sweeps) repeating in the pinned cavity.

## Verdict

- **Do NOT train the full-cage position flow (option B) against this dataset.** Its only unique payoff —
  proposing alternative basins — addresses a regime (R>xi) the N=512 box cannot host, and in the reachable
  (pinned) regime a no-learning teleport heat-bath (configurational-bias MC; exact-able as I-MTM/CBMC)
  equilibrates from ANY seed in ~10 sweeps. The gate did its job: it saved the training run.
- **The binding constraint is DATA, not MODEL:** measuring the paper's G_PTS(R) decay (and testing any
  learned basin-proposer where basins exist) requires a larger equilibrated reference box —
  R=4 needs L≥10.5 → N≥1400; R=5 needs L≥12.5 → N≥2350 at rho=1.2, T*=0.5 (PT/swap-MC compute, the
  validated ka_reference protocol scaled up 3-5x in N).

## Options going forward

- **(i) Data first:** equilibrate N~1400-2400 3D KA at T*=0.5 (PT + swap; days of compute), then measure
  the full G_PTS(R) curve — pinned branch AND decay — and only there re-pose the learned-proposer question.
- **(ii) Pinned-branch PTS now:** run the paper-protocol overlap measurement at R ≤ 2.5 on N=512 using the
  oracle heat-bath sampler (cheap, no training, delivers a real G_PTS measurement + the qc_pair estimator,
  bounded to the high-overlap branch).
- Both are compatible: (ii) now while (i) runs.
