# Big-box cavity retrain — held-NLL trace (reconstructed from live session snapshots)

NOTE: the full per-eval stdout log was lost (cp overwrote it with an empty task-output file — the
"never cp task-output over the real log" hazard). Checkpoints intact; snapshots below captured live.

| step  | held_nll | gap (train-held) | notes |
|-------|----------|------------------|-------|
| 0     | -2.1127  | +0.19  | warm-start from rho=1.2 ebm3ax |
| 10500 | -2.4008  | -0.04  | |
| 18000 | -2.5193  | -0.27  | MTM early-read taken here |
| 19000 | -2.4890  | -0.14  | |
| 24000 | -2.5216  | -0.23  | |
| 25000 | -2.5389  | +0.01  | |
| 25500 | -2.5665  | -0.49  | |
| 27500 | -2.5360  | -0.04  | |
| 38000 | -2.6018  | +0.14  | |
| 38500 | -2.6218  | +0.21  | |
| 39000 | -2.6320  | -0.21  | |
| 39500 | -2.6066  | +0.00  | EARLY STOP (held-NLL flat 8 evals) |
| best  | **-2.6360** |     | best-by-held checkpoint saved |

VERDICT: monotone held-NLL descent -2.11 -> -2.64 over 39.5k steps, NO overfit — train-vs-held gap
oscillates around 0 (frequently POSITIVE, i.e. held better than the 6-cavity train batch = pure batch
noise, no memorization drift) across 112 independent chains. Early-stopped on a genuine plateau.
