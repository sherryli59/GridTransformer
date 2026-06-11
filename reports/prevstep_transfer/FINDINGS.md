# prev_step K=1 rail: zero-shot size transfer verdict (epoch-150 snapshot)

Samples: `lj_ckpts_lj27_pbc_fixedrail_k1_prevstep/.../samples_ep150_N{27,64,125}.npz`
(generated 2026-06-11 ~09:00 via the `sample_k1_transfer.py` pow2 wrapper, R=64/128/128,
T=0.9, 256 samples, legacy sampler — predates the KV cache). This run is the controlled
A/B against the failed absolute-reference K=1 rail: same architecture, one knob changed
(`curve_rail_reference: absolute → prev_step`), i.e. option (2) from the K=1 post-mortem.

| size | prev_step ep150 | K=1 absolute (final) | arc_repr ep332 | uniform |
|------|-----------------|----------------------|----------------|---------|
| N=27 in-dist | 0.651 | 0.51 | 0.527 | 1.0 |
| N=64 zero-shot | **1.590** | 1.442 | **0.818** | 1.0 |
| N=125 zero-shot | **2.139** | 1.629 | **0.909** | 1.0 |

(gr_L1 at transfer sizes is *worse than uniform*: 0.327 vs 0.299 at N=64, 0.301 vs 0.247
at N=125 — structured-but-wrong, the K=1 failure mode again, and growing with size.)

## Verdict: translation-invariant rail reference does NOT fix transfer

Switching the rail reference to prev_step did not help and arguably hurt (OTgap above the
absolute-reference baseline at both sizes, with the size-growing signature of unbounded
autoregressive drift). This is consistent with the project's own earlier root-cause
analysis: a purely *relative* conditioning chain has no absolute anchor, so generation
drift compounds without bound — the prev_step reference removes exactly the anchor that
limited it. The arc representation keeps a translation-invariant *target* while anchoring
every decode to an absolute cell center, which is why its transfer degrades mildly
(0.82/0.91) instead of collapsing.

Caveat: ep150 of 400 (mid-training) vs arc's ep332 — but the in-distribution gap is small
(0.65 vs 0.53) while the transfer gap is catastrophic vs mild; further training does not
plausibly close a >1.2 OTgap difference (the fully-trained absolute K=1 confirms this
failure mode survives convergence).

## Implication for the roadmap

Option (2) of the K=1 post-mortem (translation-invariant rail) is now tested and ruled
out for zero-shot. arc_repr is the only live representation path; the next experiment
remains multi-size arc training ({L3,L5} hold out L4). Re-benchmark this prev_step run
at its final epoch only if it is being kept for an in-distribution purpose.
