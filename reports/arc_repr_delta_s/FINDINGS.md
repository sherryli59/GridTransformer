# Step-3 audit findings — arc_repr (Δs, fine) size-invariance

Script: `analyze_arc_delta_transfer.py` (500 MCMC configs/size, seed 42, pow2 transfer
resolutions from the K=1 study; control row isolates cell-size from box-size effects).
Question: did arc_repr absorb the Hilbert-turn problem, or just move it into the Δs tail?

| case | cell | mean | p50 | p90 | p99 | p99.9 | max | P(Δs>5) | KS Δs | KS fine |
|------|------|------|-----|-----|-----|-------|-----|---------|-------|---------|
| N=27 L=3 R=64 (train ref) | 0.0469 | 0.986 | 0.824 | 1.831 | 3.01 | 4.09 | 5.5 | 8e-5 | — | — |
| N=64 L=4 R=128 | 0.0312 | 0.996 | 0.877 | 1.787 | 2.89 | 3.97 | 5.7 | 6e-5 | 0.044 | 0.009 |
| N=125 L=5 R=128 | 0.0391 | 0.997 | 0.855 | 1.906 | 3.03 | 4.09 | 6.5 | 8e-5 | 0.032 | 0.009 |
| N=27 L=3 R=128 (control) | 0.0234 | 0.986 | 0.824 | 1.831 | 3.01 | 4.09 | 5.5 | 8e-5 | 0.000 | 0.011 |

## Verdict: GO — the turn problem is genuinely absorbed, not moved

- **No heavy tail anywhere.** P(Δs>5) ≈ 1e-4 at every size, P(Δs>10) = 0, max Δs ≤ 6.5.
  Contrast the old relative-delta representation, where ~8% of steps at L=4/5 were
  window-busting turns the tokenizer could not express (the OTgap-2.26 bottleneck).
- **Why:** in code space, consecutive Hilbert-sorted particles are consecutive order
  statistics on a line — gaps concentrate near the mean X with sub-exponential tails
  (LJ repulsion fills space evenly). The big *cartesian* jump at a Hilbert fold is
  supplied by the fixed s→xyz decode, not by the model's prediction. The model only
  ever has to express small Δs and a sub-cell fine offset.
- **Size-invariance is excellent:** Δs KS ≤ 0.044, fine KS ≤ 0.011 vs the training
  reference — an order of magnitude tighter than the K=1 rail-input audit (KS 0.30),
  and on the *target* this time, not just the input.
- **Cell-size robustness:** the N=27 control at R=128 reproduces the R=64 Δs
  distribution to KS ≈ 0.000 — the Hilbert recursion is self-similar (codes scale ×8,
  X scales ×8), so the pow2 cell shrink at transfer (0.031/0.039 vs 0.047) does not
  shift the Δs target at all. This removes failure cause (c) from the K=1 post-mortem.

## Caveat (from the K=1 lesson)

Matching *target marginals* is necessary, not sufficient — the K=1 study failed in the
learned conditional despite passing its audit. What's different here: the arc_repr
*input* (continuous 4D arc deltas) is the same size-invariant quantity as the target,
and no absolute-position feature remains in the loop. The remaining risks are the
learned conditional and AR error compounding, which only training + zero-shot
generation can test.

## Next step

Train arc_repr at N=27 (preprocessed cache with absolute coords, `--arc_repr 1
--continuous_input --use_continuous_head`), check teacher-forced NLL parity across L,
then zero-shot N=64/125 and benchmark — the Stage-1/Stage-2 playbook from
`reports/k1_rail_transfer/`.

Plot: `arc_delta_s_transfer.png`.
