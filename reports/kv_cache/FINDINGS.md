# KV-cached sampling: parity + speedup (review efficiency finding #21)

`GraphormerAR.forward_step` + `GenerationCache` replace the per-step full-prefix
re-forward in `sample_lj.py` (default ON; `--no-kv_cache` keeps the legacy path verbatim
for A/B). EdgeBias contributes only its last causal row (`bias_row`, same `_bias_from_diff`
math as the full matrix); past K/V are exact because the architecture has no RoPE and all
input embeddings are position-local. Factorized models and `ida_pre` fall back to legacy
automatically.

## Nothing changes (parity evidence)

1. **Unit level** (`tests/test_kv_cache.py`, 5 tests): bias row == last causal row of the
   full bias (torus on/off); incremental attention == full attention at every position;
   `forward_step` == full forward for the discrete config AND the arc/continuous/full-cov
   config of the real checkpoint; end-to-end argmax sampling identical cached vs legacy.
2. **Real checkpoint** (epoch-332 arc_repr, argmax, seed 123, CPU):
   - **N=125, 128 samples: 0 of 16,000 particle positions differ** (>1e-5, wrap-aware);
     max |logp| diff 6.1e-5 (float32 noise).
   - N=64, 64 samples: logp matches to 3.8e-5; 60/4096 positions differ — every one is
     the **last** particle, by **exactly one cell quantum**, where the end-of-curve clip
     puts the argmax fine-offset on the ±half-cell min-image boundary and float32
     reduction order (full matmul vs row-wise) flips the wrap. This knife edge is
     inherent to any two numerically-equivalent float implementations (changing BLAS
     thread count does the same); it is not a cache defect. Distribution unchanged.

## Just a speedup (timing, CPU, OMP_NUM_THREADS=16, incl. ~30s ckpt load each)

| case | legacy | KV cache | wall speedup |
|------|--------|----------|--------------|
| N=64, 64 samples | 66.5 s | 48.4 s | 1.4× |
| N=125, 128 samples | 451.7 s | 127.6 s | **3.5×** (≈4.3× loop-only) |

The FLOPs reduction is far larger than the wall ratio; on CPU the cached path is bounded
by fixed per-step overheads — the python loop, the full-covariance MDN head, and the
`_arc_decode_positions` per-step numpy round-trips (review efficiency finding #19, still
open). Those overheads are size-independent, so the speedup grows with N (1.4× → 3.5×
from N=64 → N=125) and large-`nsamples` GPU campaigns benefit most.

Side observation worth a look someday: the N=64 boundary case reveals that the *last*
particle's predicted fine-offset systematically sits on the cell edge (consequence of
clipping `c_next` at the end of the Hilbert curve) — a small end-of-sequence artifact of
the arc decode, independent of caching.
