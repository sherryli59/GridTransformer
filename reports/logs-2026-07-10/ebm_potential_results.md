# Learned pairwise-potential (energy-like embedding) block-MTM proposal — 2D KA results

**Date:** 2026-07-11
**Model:** `liquid_coupling_flow/artifacts/ka_localframe_ebm_N100.pt` (warm-started from
`ka_localframe_N100_20k.pt`, 6000 fine-tune steps, -logq/N 0.76 -> 1.33)
**Method:** general learned pairwise radial potential `V_theta(x|cage) = sum_i phi_theta(|x-x_i|, s_j, s_i)`
tilting the b|a categorical of the local-frame AR head. NO LJ form baked in (answers "method must be
general, not depend on specific energy form"). phi = MLP on RBF(distance) + species-pair embedding.
Exactness: sampler logq == scorer logp to 0.0 (fp), MTM stays exact.

## MTM acceptance, suffix-blob k, N_trials 16/32/64 (20 configs, beta=2.0, N=100)
| k | factorized | EBM-potential |
|---|-----------|---------------|
| 4 | 25/50/50% | **65/65/85%** |
| 6 | 10/30/20% | 20/30/**40%** |
| 8 | 5/10/5%   | 15/10/**10%** |

Key: for factorized, more trials STOP helping (k=6 10->30->20); with the potential, more trials CONVERT to
acceptance (k=4 65->85, k=6 20->40) — the proposal crossed back over the threshold where IS works.

## Structural: species-resolved block-centered g(r), first-peak height + first-shell L2 vs TRUE data block
| pair | data peak | factorized peak / L2 | EBM peak / L2 |
|------|-----------|----------------------|---------------|
| g_AA | 1.91 | 1.02 / 0.301 | **1.23 / 0.216** |
| g_AB | 3.37 | 1.57 / 0.441 | **2.00 / 0.330** |
| g_BB | 1.27 | 0.74 / 0.207 | **0.82 / 0.152** |
Potential improves ALL three peaks + L2, incl. g_BB (historically the stubborn one). Plot: `ebm_block_gr.png`.

## Caveats
1. `<U_prop-U_true>` energy diagnostic in sweep_ebm_ksweep.py is UNINFORMATIVE: raw mean over LJ r^-12
   tail, dominated by rare near-overlaps (~1e15). Use median/core-clamped energy instead. MTM acceptance
   handles the rare overlaps correctly via logsumexp weights.
2. Peaks still undershoot data (g_AB 2.00 vs 3.37 worst). Partly expected (a proposal is broader than one
   config), partly because only the FINAL axis b|a is tilted. Next lever = tilt a too (full 2D sharpening).

## Files
- `liquid_coupling_flow/ka_localframe_ebm.py` — KALocalFrameEBM (potential head)
- `reports/logs-2026-07-10/train_ebm_localframe.py` — warm-start fine-tune
- `reports/logs-2026-07-10/test_ebm_block_mtm.py` — exactness gate + K-sweep
- `reports/logs-2026-07-10/sweep_ebm_ksweep.py` — K x N_trials + energy diag
- `reports/logs-2026-07-10/struct_ebm_gr.py` — species-resolved g(r)
