# Step-4a: teacher-forced NLL parity across box sizes (arc_repr)

> **MULTI-SIZE UPDATE (2026-06-11, epoch-18 snapshot of `multisize_arc_fullcov`, 18/100
> epochs):** the {L3,L5}-trained model already shows the hypothesized held-out recovery:
>
> | size | single-size (final) | multi-size ep18 | role |
> |------|--------------------:|----------------:|------|
> | L=3 N=27 | 0.115 | 0.168 | train |
> | **L=4 N=64** | **0.649** | **0.273** | **HELD OUT** |
> | L=5 N=125 | 0.656 | 0.187 | train |
>
> With **zero L4 training data** and only 1/5 of training done, the held-out L4
> conditional recovered >58% of the transfer gap; both train sizes sit near each other
> (0.168/0.187) with L4 *between* them — the model interpolates box size instead of
> extrapolating. This is the signature predicted by the flat-domain-shift diagnosis:
> once two sizes are seen, there is no size trend left to extrapolate. Expect further
> improvement by epoch 100; generation OTgap at held-out L4 is the remaining gate.

Script: `eval_arc_nll.py` (reuses `training_step` normalization; validated at L=3 against
the run's own `train/loss_epoch` ≈ 0.133 → eval 0.115, consistent since eval mode drops
training noise and best.ckpt is the best epoch). Checkpoint: **epoch-210 snapshot** of the
still-running `lj27_pbc_arc_repr_norm_fullcov` run. 2000 configs per size, constant-cell
pow2 caches (R=128 for L4/L5).

| size | cache | NLL/coord | Δ vs train | ≈ nats/particle |
|------|-------|-----------|-----------|------------------|
| L=3 N=27 (train) | R=64 (training cache) | 0.115 | — | — |
| L=4 N=64 zero-shot | R=128 | 0.649 | +0.53 | +2.1 |
| L=5 N=125 zero-shot | R=128 | 0.656 | +0.54 | +2.2 |

**The degradation is FLAT in size (L4 ≈ L5), not growing.** Contrast the old discrete
model, whose zero-shot collapse *exploded* with size (OTgap 2.34 at L4 → 8.77 at L5).
A flat step is the signature of a single train/test domain shift ("the conditional has
only ever seen L=3 contexts"), not of size-dependent extrapolation — i.e. exactly the
failure multi-size training is known to fix, and the best possible precondition for it:
once the model sees two sizes, there is no size trend left to extrapolate.

## Interpretation (epoch 210; re-run at the final checkpoint)

- **The catastrophic failure mode is gone.** The old discrete model's zero-shot L3→L4
  breakdown had two parts: turn tokens at ppl ~1e8 (unbounded) and a 13× local-conditional
  degradation (~2.6 nats/token). arc_repr shows NO unbounded component — the Δs/fine
  representation absorbed the turns, exactly as the Step-3 audit predicted.
- **What remains is the known second mechanism.** +2.1 nats/particle is strikingly close
  to the local-conditional overfit measured in the original diagnosis (the part that
  multi-size training previously recovered: ppl 1949→236 ≈ 2.1 nats). The size-invariant
  representation fixed mechanism #1 (expressing turns); mechanism #2 (the conditional
  overfits to the single training size — context length, edge-bias distance range
  L/2·√3: 2.6 at L3 vs 3.5 at L4) was always expected to need multi-size data.
- **Prediction for Stage-2 zero-shot generation:** degraded-but-structured samples
  (OTgap likely yellow/red but, unlike K=1's >1.0, not necessarily worse than noise,
  since there is no absolute-position extrapolation in the conditioning). If that holds,
  the pre-registered move is multi-size training ({L3,L5} hold out L4) on the arc
  representation — the combination the project's own evidence says is required.

## Commands

```bash
python eval_arc_nll.py --ckpt <ckpt> --cache <cache.pt> --limit 2000
# caches: lj_caches_arc_transfer/lj64_L4_R128_cache.pt, lj125_L5_R128_cache.pt
```
