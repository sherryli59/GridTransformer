# Liquid coupling-flow: honest consolidation (2026-06-16)

**One-line verdict.** The exact-likelihood periodic flow + annealed-SMC machinery is correct and
reproduces the LJ Boltzmann distribution in 2D *and* 3D. But the *scientific* thesis — that locality
gives a **size-transferable generator** — is **not demonstrated**: the flow does not transfer to
larger N, and at the (easy) state point we tested, the flow adds almost nothing over running SMC from
uniform noise. The robust, honest contribution is narrow; the make-or-break experiment (hard state
point) is identified but untested.

Branch: `liquid-coupling-flow`. Code in `liquid_coupling_flow/`, figures in
`liquid_coupling_flow/artifacts/`. This supersedes the running log in `PROGRESS.md`.

---

## 1. Goal

Size-transferable Boltzmann sampling of disordered liquids: either generate the exact Boltzmann
distribution (match all statistics) or give exact likelihoods so configurations can be importance-
reweighted. Design bet: liquids are homogeneous → **locality** (a cutoff-local, exact-likelihood
flow) should give **size-transferability**, and exact `log q` + SMC should give **exactness at scale**.

## 2. What was built (and is correct)

A periodic exact-likelihood flow stack, all unit-tested to machine precision (float64):
circular rational-quadratic splines (torus), an **autoregressive particle flow** (`particles_ar.py`,
`particles_ar_nd.py`; places particles one at a time, each coordinate from a spline shaped by
already-placed real neighbours — exact triangular `log q`, no auxiliary), periodic LJ energy, and an
**annealed-SMC corrector** (`smc.py`) with a single-particle Metropolis kernel.

Key engineering wins:
- **Diagnosed the original "stuck at identity" failure**: a coupling/augmented flow cannot create
  excluded volume; autoregressive placement (condition on *real* placed neighbours) fixes it.
- **Single-particle SMC kernel**: whole-config moves accept as ~(single-acc)^N ≈ 0.01 in a dense
  liquid → SMC degenerated to resampling-only (high ESS but 35% unique). One particle at a time →
  acc 0.47, genuine rejuvenation (92% unique).

## 3. What works — exactness at scale (SOLID)

flow + single-particle SMC, importance-**reweighted** (⟨A⟩ = Σ wᵢ A(xᵢ)) vs long-MCMC ground truth:

| system | reweighted ⟨U⟩/N | MCMC ⟨U⟩/N | ESS | figure |
|---|---|---|---|---|
| 2D LJ N=16, T\*=1, ρ=0.64 | −1.558 ± 0.002 | −1.559 | 96–99% | `ar_smc_reweighted.png` |
| 3D LJ N=32, T\*=1, ρ\*=0.65 | −2.776 ± 0.003 | −2.770 | 85% | `lj3d_firmed.png` |

Energy distribution **and** g(r) are indistinguishable from Boltzmann in both. The exact-`log q`
reweighting route works; the architecture generalizes to 3D. (Residual energy bias at fixed SMC
budget is pure under-relaxation: in 3D, n_bridge 30→50 moved ⟨U⟩/N −2.759→−2.776 onto MCMC.)

## 4. What does NOT work — transfer and the flow's value (the negatives)

### 4a. The raw flow does not size-transfer
Trained at N=16, evaluated at N=36/64 (same density). g(r) of flow+SMC matches at all sizes, but the
**raw flow degrades**: nearest-neighbour overlap fraction 0.055 → 0.22 → 0.43 (uniform-noise baseline
≈ 0.72). It beats noise but is a weak large-N proposal. Two targeted architecture fixes **both failed**:
- `_ctx0` sum→mean (intensive coord-0 context): **lateral** (helped large-N a bit, hurt train-size).
- `_ctx_c` sum→mean (intensive excluded-volume context): **worse** (N=64 overlaps 0.43→0.47). The
  extensive sum was useful *signal* (local crowdedness), not a bug.

Conclusion: the transfer barrier is **not** context extensivity. It is the **AR-placement
out-of-distribution regime** — at large N a late particle is conditioned on far more placed neighbours
than anything seen in training. (Caveat from a good critique: at fixed density the neighbour count
*within a cutoff* is constant; the residual extensivity is along the **perpendicular axis** of
coordinate-wise placement — placing a y-coordinate sees a full-height column. A genuinely
cutoff-local conditioner needs the *full* position, i.e. not coordinate-wise AR.)

### 4b. The flow barely helps SMC (at the tested state point)
The decisive control (`control_flow_vs_uniform.py`): run SMC from the **flow** proposal and from
**uniform noise** under the *identical* bridge + kernel; measure convergence.

At T\*=1, ρ=0.64: **both** reach zero overlaps in ~5 sweeps; the flow's head-start is only ~1–2 sweeps
and **shrinks with N** (N=16 ~2, N=64 ~1). At N=64 the flow-init is *no better* (slightly worse:
−1.512 vs uniform −1.606, MCMC −1.577) — anchoring the bridge to a bad proposal tethers the sampler.
**So the annealed SMC is the workhorse, and it is already size-agnostic.** "Size-transfer of the flow"
is moot if SMC samples any N from uniform in a few sweeps. `control_flow_vs_uniform.png`.

## 5. Methodological lessons (worth keeping)

- **"Verified on toys" ≠ works on the real system.** The SMC kernel passed every toy test but had a
  latent rank bug (`accept[:, None]` assumed flat `[M,D]`; particle configs are `[M,N,d]`) that only
  fired on first contact with particles.
- **ESS is not a reliability metric on its own.** It saturates (the `r⁻¹²` tail makes a flow with 0.37
  overlaps and uniform with 0.72 *both* read 0.04% plain-IS ESS) and it can read ~100% right after a
  terminal resample on a subtly-wrong ensemble. **Always check an independent observable** (energy,
  g(r), overlap fraction).
- **Test whether the learned component earns its keep.** The flow-vs-uniform-under-identical-SMC
  control is the clean way to ask "is the model doing the work, or is the corrector?"

## 6. Honest conclusion + the one decisive open question

What we can claim: a *correct* exact-likelihood periodic flow + SMC that reproduces 2D and 3D LJ
Boltzmann distributions and reweights within statistical error. What we **cannot** claim: a
size-transferable *generator*, or that the flow provides value beyond annealed SMC for liquids.

The single experiment that would settle the project's worth (not run): **re-run the flow-vs-uniform
control at a HARD state point** (low T\* ≈ 0.6, higher density, slow-mixing/metastable) — the only
regime where a learned global proposal can beat uniform+SMC. If the flow separates from uniform there,
the approach has a real niche (and *then* multi-size training / a full-position cutoff-local model are
worth pursuing). If it ties there too, the flow approach does not pay for this problem — a clean
negative.

## 7. Index

Code (branch `liquid-coupling-flow`): `particles_ar.py`, `particles_ar_nd.py`, `smc.py`, `energy.py`,
`mcmc.py`; drivers `train_ar_ckpt.py`, `apply_smc.py`, `plot_smc_reweighted.py`, `train_3d.py`,
`firm_3d.py`, `size_transfer.py`, `run_meanctx.py`, `run_ctxc.py`, `control_flow_vs_uniform.py`.
Figures (`liquid_coupling_flow/artifacts/`): `ar_smc_reweighted.png`, `lj3d_firmed.png`,
`size_transfer_gr_*.png`, `control_flow_vs_uniform.png`. Memory: `liquid-coupling-flow-pivot`.
