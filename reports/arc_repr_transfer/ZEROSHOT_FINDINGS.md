# Stage-2 zero-shot findings — arc_repr size transfer (epoch-332 snapshot)

Checkpoint: epoch-332 snapshot of `lj27_pbc_arc_repr_norm_fullcov` (training still running
to 400). Sampler: fixed `sample_lj.py` with the new `--cell_size 0.046875` override
(constant physical cell → R=64/128/128; no monkey-patch). 256 samples/size, T=0.9, CPU.
Benchmark: `benchmark_lj27.py` vs size-matched MCMC.

| size | R | OTgap | gr_L1 | Uc/N (target) | uniform gr_L1 | band |
|------|---|-------|-------|----------------|----------------|------|
| N=27 in-dist | 64 | **0.527** | 0.270 | 27.7 (0.30) | 0.380 | green-edge |
| N=64 zero-shot | 128 | **0.818** | 0.257 | 34.4 (−0.34) | 0.299 | red, sub-noise |
| N=125 zero-shot | 128 | **0.909** | 0.196 | 35.2 (−0.61) | 0.247 | red, sub-noise |

Reference: K=1 rail zero-shot (same protocol) scored **1.442 / 1.629** — worse than
uniform noise. Old discrete model: OTgap 2.34 (L4) / 8.77 (L5).

## Verdict: MARGINAL — first sub-noise zero-shot transfer in the project

**Both transfer sizes are below OTgap 1.0 for the first time across all attempts.** The
samples are inside the region the EGNN flow already transports (closer to the target than
the flow's own uniform base), i.e. *refinable in principle*, though not yet in the green
band (≲0.6). gr_L1 beats uniform at every size (at N=125: 0.196 vs 0.247 — real pair
structure, not noise).

Consistency with the Step-4a NLL parity: teacher-forced degradation was flat in size
(+0.54 nats/coord at both L4 and L5) while generation OTgap grows mildly with size
(0.82 → 0.91) — the extra slope is autoregressive error compounding over 2.4×/4.8×
longer sequences, bounded by the arc decode's absolute cell anchoring (no collapse).

The three-mechanism scorecard from the original size-generalization diagnosis:
1. Turn expression — **fixed** by arc_repr (Step-3 audit + no unbounded NLL component).
2. Local-conditional overfit to L=3 — **present** (flat +2.1 nats/particle; the dominant
   remaining gap). Known cure: multi-size training (previously recovered 1949→236 ppl).
3. AR error compounding — **present but bounded** (0.82→0.91 slope; no K=1-style runaway
   because no absolute-position input extrapolates).

## Recommended next step

Multi-size arc_repr training: {L3, L5} train, hold out L4 (caches all built:
`lj_caches_arc_transfer/`). Success criterion: held-out L4 OTgap ≲ 0.6 (green), which
would clear the hand-off line for end-to-end refinement through learndiffeq. Re-run this
table with the final (epoch-400) checkpoint first — cheap, and the in-dist row may
improve a notch.
