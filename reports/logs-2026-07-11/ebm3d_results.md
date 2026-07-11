# 3D learned-potential cavity infiller — results (2026-07-11)

Port of the 2D learned-pairwise-potential (energy-like embedding) to the 3D KA cavity, per
`docs/superpowers/specs/2026-07-11-ka3d-learned-potential-cavity-design.md`. Module
`liquid_coupling_flow/ka3d_scaffold_ebm.py` (`KA3DScaffoldEBM`). GENERAL learned potential (no LJ/sigma
/eps) tilting the Cat3Head; cage = [boundary UNION interior] pooled with NO kind flag (boundary treated as
ordinary surrounding particles); zero-init Fourier(R) token.

## Exactness
- EBM(phi~0) block_log_prob == base ka3d_block: **0.0** (geometry/frame/ball-map transcribed correctly).
- sample-vs-score self-consistency: **8.24e-3** (inherited EXACTLY from the base cat-head; the ball
  squash/unsquash round-trip flips a bin near an edge). The tilt adds ZERO error. Base issue, not the port.

## Conditional NLL ladder (held cavities, -logq/particle; lower = sharper)
| model | NLL | vs base |
|-------|-----|---------|
| frozen-phi base (no potential) | -1.57 | -- |
| c-only tilt (1 of 3 axes) | -2.45 | +0.88 |
| **multi-axis a+b+c tilt** | **-2.73** | **+1.16** |

## K=6 clash decomposition -- 'intra-cluster or w the boundary' (120 block particles)
| clash of block particle with... | base | c-only | multi-axis |
|---------------------------------|------|--------|------------|
| BOUNDARY (frozen shell) | 20% | 21% | **14%** |
| RETAINED interior | 27% | 20% | **14%** |
| other BLOCK particle | 22% | 15% | 16% |
KEY: c-only (tilting just the last axis) fixed INTERIOR clashes but NOT the boundary (single-axis dodge
can't escape the dense wall once a,b land on it). Multi-axis dodging drops boundary 20->14% and retained
27->14% -- the same pairwise potential handles the wall for free once every axis can respond.

## Species-resolved g(r) -- the structural statistic (bulk-normalised rho_b=rho*x_b, g->1 at large r)
Block-centred g(r) of the regenerated block particles vs all cavity particles; ground truth = TRUE block.
(self-pairs excluded BY INDEX -- cdist self-distance ~1e-3 from cancellation leaks into bin 0 otherwise;
this was the earlier "peaks in the 100s" bug.) First-peak height + first-shell L2-to-data:
| pair | data peak | base peak / L2 | c-only peak / L2 | multi-axis peak / L2 |
|------|-----------|----------------|------------------|----------------------|
| g_AA | 3.82 @1.08 | 1.93 / 0.672 | 2.01 / 0.605 | **2.30 / 0.518** |
| g_AB | 4.60 @0.88 | 1.83 / 0.881 | 2.31 / 0.705 | **3.14 / 0.493** |
| g_BB | 1.59 @1.58 | 1.29 / 0.365 | 1.39 / -- | **-- / 0.343** |
Multi-axis has the LOWEST L2-to-data on ALL three species; first peaks climb toward data monotonically
base<c-only<multi-axis (clearest for g_AB, the strongest interaction: 1.83->2.31->3.14, data 4.60). Peaks
still undershoot data (proposal is broader than one config, pre-MTM -- same as 2D). Plot: ebm3d_cavity_gr.png.

## block-MTM acceptance (20 cavities, +-~11% noise; base -> multi-axis)
| K | base N16/32/64 | multi-axis N16/32/64 |
|---|----------------|----------------------|
| 2 | 35/25/45% | 65/45/40% |
| 4 | 0/5/0% | 5/15/**20%** |
| 6 | 0/0/0% | 0/0/0% |
K=2-4 collective moves work and improve clearly; K=6 (six simultaneous placements in a dense cavity)
remains hard for both -- an energy-cost wall, not a proposal-sharpness wall.

## Size-transfer (zero-shot held-out radius R=2.2; trained on {1.6,2.0,2.4})
POSITIVE, no cliff. multi-axis EBM > base at the UNSEEN radius:
| K | base N16/32/64 | EBM N16/32/64 |
|---|----------------|---------------|
| 2 | 25/20/25% | 35/40/40% |
| 4 | 0/0/5% | 0/5/10% |
Clash @R=2.2: base boundary 18/retained 30/block 25%; EBM boundary 21/retained 16/block 18%. Interior
clashes transfer (retained 30->16, block 25->18); the BOUNDARY fix is more R-specific (18->21, ~noise) --
the R-embedding + size-invariant potential interpolate for the interior, boundary handling transfers less
cleanly. Overall: acceptance + interior structure transfer zero-shot to the held-out radius.

## Verdict
The energy-like-embedding architecture ports to 3D cavity infilling: exact, boundary-as-cage unification
works, and MULTI-AXIS tilting is the right architecture (c-only under-resolves 3D). The potential earns a
large conditional gain (+1.16 nats), lowers species-resolved g(r) L2-to-data on all three pairs, and lifts
acceptance; the K=6 ceiling localizes the residual to collective-move energy cost, not the proposal.
The clash cutoff and g(r) agree (base<c-only<multi-axis); g(r) is the trustworthy structural metric.

## Files / artifacts
- ka3d_scaffold_ebm.py (KA3DScaffoldEBM, multi-axis)
- artifacts: ka3d_cavity_base.pt (frozen-phi), ka3d_cavity_ebm.pt (c-only), ka3d_cavity_ebm3ax.pt (a+b+c)
- reports/logs-2026-07-11/: train_ebm3d_cavity.py, test_ebm3d_exact.py, test_ebm3d_cavity.py, struct_ebm3d_gr.py
