# K=1 Rail Size-Transfer Feasibility Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Decide, without training, whether the trained K=1 rail model (N=27) transfers to N=64/N=125 — via a distribution audit that gates a zero-shot generation + benchmark.

**Architecture:** Stage 1 is a pure-analysis script that compares the (sample-independent) K=1 rail waypoint feature and the (MCMC-derived) cartesian-delta target across sizes under two R strategies, emitting a KS table + plots. Stage 2 reuses the existing `sample_lj.py` via a thin wrapper that injects a constant cell_size so the rail resolution scales with the box, then scores samples with `benchmark_lj27.py`.

**Tech Stack:** Python, numpy, scipy.stats (KS), h5py, matplotlib; existing repo helpers `fixed_template_waypoints`, `_hilbert3d_decode`, `_hilbert_bits`, `min_image_delta` in `grid_transformer/data/lj_transferable.py`.

**Key constants:**
- `CELL_TRAIN = 3.0 / 64 = 0.046875` (training cell size).
- Sizes: `(N=27, L=3.0)`, `(N=64, L=4.0)`, `(N=125, L=5.0)`, all ρ=1, T=1.
- Targets: `/mnt/ssd/mcmc/lj_mcmc_sweep_3d/lj3d_L{3,4,5}_rho1.0_N{27,64,125}_T1.0.h5`.
- Rail config (from checkpoint): `k=1`, `window_scale=1.0`, `reference="absolute"`, `mode=fixed_template`.
- Checkpoint: `lj_ckpts_lj27_pbc_fixedrail_k1_ablation/lj27_pbc_continput_fixedrail_k1_fullcov/best.ckpt`.
- KS go/no-go threshold: `0.30`. In-box fraction threshold (exact-R arm): `0.95`.
- Python: `/home/sherryli/xsli/softwares/anaconda3/envs/lightning/bin/python`.

---

## Task 1: Stage-1 audit core functions + unit tests

**Files:**
- Create: `analyze_k1_rail_transfer.py`
- Test: `tests/test_k1_rail_transfer.py`

The audit needs three pure functions: a resolution chooser (pow2 vs exact), a rail-waypoint
extractor (mirrors what the model is conditioned on), and an in-box-fraction check. The
cartesian-delta target reuses the existing `load_configs` pattern. Build the functions
test-first.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_k1_rail_transfer.py
import math
import numpy as np
import pytest

from analyze_k1_rail_transfer import (
    choose_R,
    rail_waypoints_absolute,
    in_box_fraction,
    CELL_TRAIN,
)


def test_choose_R_pow2():
    # pow2 strategy rounds up to next power of two of L/CELL_TRAIN
    assert choose_R(3.0, "pow2") == 64
    assert choose_R(4.0, "pow2") == 128
    assert choose_R(5.0, "pow2") == 128


def test_choose_R_exact():
    # exact strategy keeps cell ~constant: R = round(L/CELL_TRAIN)
    assert choose_R(3.0, "exact") == 64
    assert choose_R(4.0, "exact") == 85
    assert choose_R(5.0, "exact") == 107


def test_rail_waypoints_shape_and_determinism():
    # K=1 fixed-template rail is sample-independent: depends only on (N, R, L).
    wp1 = rail_waypoints_absolute(N=27, R=64, L=3.0)
    wp2 = rail_waypoints_absolute(N=27, R=64, L=3.0)
    assert wp1.shape == (26, 3)            # N-1 waypoints
    assert np.allclose(wp1, wp2)           # deterministic


def test_rail_waypoints_in_box_for_pow2():
    # pow2 R == curve side, so every decoded waypoint sits inside [0, L)^3.
    wp = rail_waypoints_absolute(N=64, R=128, L=4.0)
    assert in_box_fraction(wp, 4.0) == 1.0


def test_in_box_fraction_counts_out_of_box():
    pts = np.array([[0.1, 0.1, 0.1], [3.9, 0.1, 0.1], [4.5, 0.1, 0.1]])
    # with L=4, the third point (4.5) is out of box -> 2/3 in box
    assert in_box_fraction(pts, 4.0) == pytest.approx(2.0 / 3.0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /mnt/ssd/GridTransformer && /home/sherryli/xsli/softwares/anaconda3/envs/lightning/bin/python -m pytest tests/test_k1_rail_transfer.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'analyze_k1_rail_transfer'` (or ImportError on the names).

- [ ] **Step 3: Write minimal implementation**

```python
# analyze_k1_rail_transfer.py
"""Stage-1 distribution audit for K=1 rail size-transfer feasibility.

Compares the (sample-independent) K=1 fixed-template rail waypoint feature and the
(MCMC-derived) cartesian-delta target across N=27/64/125 under two R strategies, to decide
whether a zero-shot transfer of the trained K=1 checkpoint is worth running.

See docs/superpowers/specs/2026-06-10-k1-rail-size-transfer-feasibility-design.md
"""
from __future__ import annotations

import argparse
import math
import os

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import ks_2samp

from grid_transformer.data.lj_transferable import (
    fixed_template_waypoints,
    min_image_delta,
    _hilbert3d_encode,
    _hilbert_bits,
)

CELL_TRAIN = 3.0 / 64.0  # 0.046875

SIZES = [
    dict(N=27, L=3.0, path="/mnt/ssd/mcmc/lj_mcmc_sweep_3d/lj3d_L3_rho1.0_N27_T1.0.h5"),
    dict(N=64, L=4.0, path="/mnt/ssd/mcmc/lj_mcmc_sweep_3d/lj3d_L4_rho1.0_N64_T1.0.h5"),
    dict(N=125, L=5.0, path="/mnt/ssd/mcmc/lj_mcmc_sweep_3d/lj3d_L5_rho1.0_N125_T1.0.h5"),
]


def choose_R(L: float, strategy: str) -> int:
    """Grid resolution for a box of side L holding cell_size ~= CELL_TRAIN.

    strategy="pow2":  next power of two of L/CELL_TRAIN (cell shrinks; validated path).
    strategy="exact": round(L/CELL_TRAIN) (cell ~constant; may be non-power-of-two).
    """
    n = max(2, int(round(L / CELL_TRAIN)))
    if strategy == "exact":
        return n
    if strategy == "pow2":
        return int(1 << int(math.ceil(math.log2(n))))
    raise ValueError(f"unknown strategy {strategy!r}")


def rail_waypoints_absolute(N: int, R: int, L: float) -> np.ndarray:
    """K=1 fixed-template rail waypoints the model is conditioned on: [N-1, 3].

    Exactly mirrors the training/sampling rail config (k=1, window_scale=1.0,
    reference="absolute"): waypoint for particle j is the box-frame cell center of
    decode((j+1)*X), X = R**3 // N. Sample-independent by construction.
    """
    box = np.array([L, L, L], dtype=np.float64)
    pred_idx = np.arange(1, N, dtype=np.int64)  # predicting particles 1..N-1
    wp = fixed_template_waypoints(
        pred_idx, N, int(R), box,
        k=1, window_scale=1.0, periodic=True, reference="absolute",
    )  # [N-1, 1, 3]
    return wp.reshape(-1, 3).astype(np.float64)


def in_box_fraction(points: np.ndarray, L: float) -> float:
    """Fraction of points whose every coordinate lies in [0, L)."""
    inside = np.all((points >= 0.0) & (points < L), axis=-1)
    return float(np.mean(inside))


def load_configs(path: str, N: int, n_configs: int, seed: int = 42) -> np.ndarray:
    """Return float32 [n_configs, N, 3] sampled from the h5 MCMC trajectory."""
    rng = np.random.default_rng(seed)
    with h5py.File(path) as f:
        traj = f["traj"]  # (n_chains, n_frames, N, 3)
        n_chains, n_frames = traj.shape[:2]
        total = n_chains * n_frames
        idx = rng.choice(total, size=min(n_configs, total), replace=False)
        ci, fi = idx // n_frames, idx % n_frames
        configs = np.stack([traj[c, f] for c, f in zip(ci, fi)]).astype(np.float32)
    return configs


def hilbert_sorted_codes(pos: np.ndarray, L: float, R: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (sorted_pos [N,3], order) by Hilbert code at resolution R."""
    box = np.array([L, L, L], dtype=np.float64)
    bits = _hilbert_bits(R)
    cell = box / float(R)
    grid = np.clip(np.floor(np.mod(pos, box) / cell).astype(np.int64), 0, R - 1)
    codes = _hilbert3d_encode(grid[:, 0], grid[:, 1], grid[:, 2], bits=bits)
    order = np.argsort(codes, kind="stable")
    return pos[order], order


def cartesian_delta_targets(configs: np.ndarray, L: float, R: int) -> np.ndarray:
    """Min-image (pos_{t+1} - pos_t) over Hilbert-sorted particles: [n*(N-1), 3]."""
    box = np.array([L, L, L], dtype=np.float64)
    out = []
    for pos in configs:
        sp, _ = hilbert_sorted_codes(pos.astype(np.float64), L, R)
        d = min_image_delta(sp[1:] - sp[:-1], box)
        out.append(d)
    return np.concatenate(out, axis=0)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /mnt/ssd/GridTransformer && /home/sherryli/xsli/softwares/anaconda3/envs/lightning/bin/python -m pytest tests/test_k1_rail_transfer.py -q`
Expected: PASS (5 passed).

- [ ] **Step 5: Commit**

```bash
cd /mnt/ssd/GridTransformer
git add analyze_k1_rail_transfer.py tests/test_k1_rail_transfer.py
git commit -m "Add K=1 rail transfer audit core + tests"
```

---

## Task 2: Audit main() — KS table, raw-cell leakage check, plots

**Files:**
- Modify: `analyze_k1_rail_transfer.py` (append `raw_cell_inbox_fraction`, `tile_to`, `_ks_max`, `main`, `__main__`)
- Modify: `tests/test_k1_rail_transfer.py` (add `raw_cell_inbox_fraction` tests)

Compose the core functions into a report. Waypoints are few (N-1 per size); tile them to a
floor sample count so KS has power (mirrors `tests/test_size_transfer_parity.py`).

**Important correctness note (discovered in Task 1):** `fixed_template_waypoints` with
`reference="absolute"` and `periodic=True` applies `min_image_delta`, so every returned
waypoint lands in `[-L/2, L/2)`. That means `in_box_fraction` on the *returned* waypoints
always reads ~1.0 and cannot detect exact (non-pow2) R curve-length leakage. The exact-R
disqualifier must instead inspect the **raw decoded Hilbert cell** (before min-image)
against `[0, R)`. Add `raw_cell_inbox_fraction` for this and use it as the disqualifier.

- [ ] **Step 1: Add the raw-cell leakage helper + tests (TDD)**

Append to `tests/test_k1_rail_transfer.py`:

```python
def test_raw_cell_inbox_pow2_is_one():
    # pow2 R == curve side: every decoded waypoint cell sits in [0, R).
    from analyze_k1_rail_transfer import raw_cell_inbox_fraction
    assert raw_cell_inbox_fraction(64, 64) == 1.0
    assert raw_cell_inbox_fraction(64, 128) == 1.0


def test_raw_cell_inbox_exact_is_valid_fraction():
    # exact (non-pow2) R may leak cells outside [0, R); result is a valid fraction.
    from analyze_k1_rail_transfer import raw_cell_inbox_fraction
    frac = raw_cell_inbox_fraction(64, 85)
    assert 0.0 <= frac <= 1.0
```

Append to `analyze_k1_rail_transfer.py` (needs `_hilbert3d_decode` added to the existing
import from `grid_transformer.data.lj_transferable`):

```python
def raw_cell_inbox_fraction(N: int, R: int) -> float:
    """Fraction of K=1 waypoints whose RAW decoded Hilbert cell lies fully in [0, R).

    Detects exact (non-power-of-two) R curve-length leakage. With bits=ceil(log2(R)) the
    Hilbert curve fills a 2**bits cube; decode((j+1)*X) can land on cells outside the
    physical [0, R) grid even though the model-visible (min-imaged) waypoint always appears
    in-box. Use this — not in_box_fraction on min-imaged waypoints — as the exact-R
    disqualifier. For power-of-two R this is always 1.0.
    """
    total = R ** 3
    X = max(1, total // int(N))
    bits = _hilbert_bits(R)
    j = np.arange(1, N, dtype=np.float64)            # predicting particles 1..N-1
    idx = np.clip(np.rint((j + 1.0) * X), 0, total - 1).astype(np.int64)  # (j+1)*X
    cx, cy, cz = _hilbert3d_decode(idx, bits=bits)
    cells = np.stack([cx, cy, cz], axis=-1)
    inside = np.all(cells < R, axis=-1)              # cells are >= 0 by construction
    return float(np.mean(inside))
```

Run: `cd /mnt/ssd/GridTransformer && /home/sherryli/xsli/softwares/anaconda3/envs/lightning/bin/python -m pytest tests/test_k1_rail_transfer.py -q`
Expected: 7 passed.

- [ ] **Step 2: Append report logic and CLI**

```python
def tile_to(arr: np.ndarray, n: int) -> np.ndarray:
    """Tile rows of arr up to at least n rows (for KS power on small waypoint sets)."""
    if len(arr) >= n:
        return arr
    reps = int(math.ceil(n / max(1, len(arr))))
    return np.tile(arr, (reps, 1))[:n]


def _ks_max(a: np.ndarray, b: np.ndarray) -> float:
    """Max per-axis KS statistic between two [*,3] sets."""
    return max(float(ks_2samp(a[:, d], b[:, d]).statistic) for d in range(3))


def main(args) -> None:
    os.makedirs(args.out_dir, exist_ok=True)
    MIN_SAMPLES = 300

    for strategy in ("pow2", "exact"):
        print(f"\n{'='*68}\nStrategy: {strategy}\n{'='*68}")
        rails, deltas, meta = {}, {}, {}
        for s in SIZES:
            N, L = s["N"], s["L"]
            R = choose_R(L, strategy)
            wp = rail_waypoints_absolute(N, R, L)
            ib = raw_cell_inbox_fraction(N, R)  # raw-cell leakage (not min-imaged waypoint)
            cfg = load_configs(s["path"], N, args.n_configs, seed=args.seed)
            dl = cartesian_delta_targets(cfg, L, R)
            rails[N], deltas[N], meta[N] = wp, dl, dict(R=R, cell=L / R, in_box=ib)
            print(f"  N={N:3d} L={L} R={R:4d} cell={L/R:.5f} in_box={ib:.3f} "
                  f"rail_n={len(wp)} delta_n={len(dl)}")

        # KS vs the N=27 reference, on both absolute and L-normalized rail features.
        ref = 27
        print(f"\n  Rail-input KS (vs N=27)         abs        /L(frac)   target-delta KS")
        ks_summary = {}
        for N in (64, 125):
            ra = _ks_max(tile_to(rails[ref], MIN_SAMPLES), tile_to(rails[N], MIN_SAMPLES))
            rf = _ks_max(
                tile_to(rails[ref] / SIZES[0]["L"], MIN_SAMPLES),
                tile_to(rails[N] / [s for s in SIZES if s["N"] == N][0]["L"], MIN_SAMPLES),
            )
            dk = _ks_max(deltas[ref], deltas[N])
            ks_summary[N] = dict(rail_abs=ra, rail_frac=rf, delta=dk)
            print(f"    N=27 vs N={N:3d}              {ra:8.4f}   {rf:8.4f}     {dk:8.4f}")

        # Plots: rail fractional position + delta magnitude overlays
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        fig.suptitle(f"K=1 rail transfer audit — strategy={strategy}", fontsize=12)
        colors = {27: "#1f77b4", 64: "#ff7f0e", 125: "#2ca02c"}
        for N in (27, 64, 125):
            L = [s for s in SIZES if s["N"] == N][0]["L"]
            axes[0].hist((rails[N] / L)[:, 0], bins=40, density=True, alpha=0.5,
                         color=colors[N], label=f"N={N}")
            axes[1].hist(np.linalg.norm(rails[N], axis=1), bins=40, density=True, alpha=0.5,
                         color=colors[N], label=f"N={N}")
            axes[2].hist(np.linalg.norm(deltas[N], axis=1), bins=60, density=True, alpha=0.5,
                         color=colors[N], label=f"N={N}")
        axes[0].set_title("rail x / L (fractional)"); axes[0].legend()
        axes[1].set_title("|rail| (absolute)"); axes[1].legend()
        axes[2].set_title("|cartesian delta| target"); axes[2].legend()
        plt.tight_layout()
        p = os.path.join(args.out_dir, f"k1_rail_audit_{strategy}.png")
        plt.savefig(p, dpi=140, bbox_inches="tight"); plt.close()
        print(f"  Saved {p}")

        # Go/no-go for this strategy (use the more meaningful L-normalized rail KS)
        max_rail = max(ks_summary[N]["rail_frac"] for N in (64, 125))
        min_ib = min(meta[N]["in_box"] for N in (64, 125))
        verdict = "PASS" if max_rail < 0.30 else "FAIL"
        print(f"  -> rail_frac maxKS={max_rail:.4f} (<0.30? {verdict}), "
              f"min in_box={min_ib:.3f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_configs", type=int, default=500)
    ap.add_argument("--out_dir", type=str, default="reports/k1_rail_transfer")
    ap.add_argument("--seed", type=int, default=42)
    main(ap.parse_args())
```

- [ ] **Step 2: Run the audit**

Run: `cd /mnt/ssd/GridTransformer && /home/sherryli/xsli/softwares/anaconda3/envs/lightning/bin/python analyze_k1_rail_transfer.py`
Expected: prints two strategy blocks with R/cell/in_box per size, a KS table, two saved PNGs, and a PASS/FAIL verdict per strategy. No exceptions.

- [ ] **Step 3: Commit**

```bash
cd /mnt/ssd/GridTransformer
git add analyze_k1_rail_transfer.py reports/k1_rail_transfer
git commit -m "K=1 rail transfer audit report (KS table + plots)"
```

---

## Task 3: Record the audit decision

**Files:**
- Create: `reports/k1_rail_transfer/STAGE1_FINDINGS.md`

Capture the chosen R strategy and the reasoning so Stage 2 is unambiguous. This is a
decision gate, not code.

- [ ] **Step 1: Write findings from the audit output**

Fill this template with the actual printed numbers:

```markdown
# Stage 1 (audit) findings — K=1 rail size transfer

| strategy | N=64 rail_frac KS | N=125 rail_frac KS | min in_box | target-delta KS (64/125) | verdict |
|----------|-------------------|--------------------|------------|--------------------------|---------|
| pow2     | <fill>            | <fill>             | <fill>     | <fill>/<fill>            | <fill>  |
| exact    | <fill>            | <fill>             | <fill>     | <fill>/<fill>            | <fill>  |

**Chosen strategy for Stage 2:** <pow2 | exact>
**Reason:** <e.g. pow2 passes KS<0.30 with in_box=1.0; exact does not lower KS enough and/or in_box<0.95>

**Stage 1 go/no-go:** <GO to Stage 2 | NO-GO> — <one line>
```

Decision rule (from spec): choose `exact` over `pow2` only if it lowers the max rail_frac KS
AND keeps min in_box ≥ 0.95; otherwise use `pow2`. Proceed to Stage 2 only if the chosen
strategy's rail_frac maxKS < 0.30.

- [ ] **Step 2: Commit**

```bash
cd /mnt/ssd/GridTransformer
git add reports/k1_rail_transfer/STAGE1_FINDINGS.md
git commit -m "Record Stage-1 audit decision for K=1 rail transfer"
```

---

## Task 4: Zero-shot sampling wrapper (cell_size / R injection)

**Files:**
- Create: `sample_k1_transfer.py`

The checkpoint has `cell_size=None`, so `_rail_resolution_for_box` returns R=64 at every box
(cell grows — wrong). This wrapper forces the Stage-1-chosen R per box, then delegates to
`sample_lj.main()` (à la `sample_arc_legacy.py`). For `pow2` it sets a constant cell so the
existing helper scales R; for `exact` it overrides the resolver to return `round(L/cell)`.

- [ ] **Step 1: Write the wrapper**

```python
# sample_k1_transfer.py
"""Zero-shot sampling of the trained K=1 rail checkpoint at a larger box.

Forces the rail grid resolution so the physical cell size matches training
(CELL_TRAIN), since the checkpoint stores cell_size=None (which would otherwise
pin R=64 and let the cell grow with the box). Usage mirrors sample_lj.py; add
--transfer_strategy {pow2,exact}.

Example:
  python sample_k1_transfer.py --transfer_strategy pow2 \
    --mode relative --ckpt <best.ckpt> --Lx 4 --Ly 4 --Lz 4 \
    --coord_dim 3 --num_particles 64 --periodic --ar_arch standard \
    --use_continuous_head --full_covariance --nsamples 256 \
    --sample_batch_size 128 --temperature 0.9 --relative_window 3.0 \
    --relative_bins 64 --device cpu --save <out.npz>
"""
import sys
import math
import numpy as np

import sample_lj

CELL_TRAIN = 3.0 / 64.0


def _strategy_R(L: float, strategy: str) -> int:
    n = max(2, int(round(L / CELL_TRAIN)))
    if strategy == "exact":
        return n
    return int(1 << int(math.ceil(math.log2(n))))  # pow2


def main() -> None:
    # Pop our extra flag before sample_lj parses argv.
    strategy = "pow2"
    if "--transfer_strategy" in sys.argv:
        i = sys.argv.index("--transfer_strategy")
        strategy = sys.argv[i + 1]
        del sys.argv[i:i + 2]

    _orig = sample_lj._rail_resolution_for_box

    def _forced(box_np, hilbert_resolution, cell_size):
        L = float(np.max(np.asarray(box_np, dtype=np.float64)))
        R = _strategy_R(L, strategy)
        print(f"[transfer] strategy={strategy} L={L} -> R={R} cell={L / R:.5f}")
        return int(R)

    sample_lj._rail_resolution_for_box = _forced
    try:
        sample_lj.main()
    finally:
        sample_lj._rail_resolution_for_box = _orig


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke-test the override wiring (tiny run)**

Run (N=27 in-distribution, 8 samples — just checks the wrapper runs and prints the forced R):
```bash
cd /mnt/ssd/GridTransformer
CUDA_VISIBLE_DEVICES="" /home/sherryli/xsli/softwares/anaconda3/envs/lightning/bin/python sample_k1_transfer.py \
  --transfer_strategy pow2 --mode relative \
  --ckpt lj_ckpts_lj27_pbc_fixedrail_k1_ablation/lj27_pbc_continput_fixedrail_k1_fullcov/best.ckpt \
  --Lx 3 --Ly 3 --Lz 3 --coord_dim 3 --num_particles 27 --periodic --ar_arch standard \
  --use_continuous_head --full_covariance --nsamples 8 --sample_batch_size 8 \
  --temperature 0.9 --relative_window 3.0 --relative_bins 64 --device cpu \
  --save /tmp/k1_smoke.npz 2>&1 | tail -5
```
Expected: a `[transfer] strategy=pow2 L=3.0 -> R=64 cell=0.04688` line and `Saved /tmp/k1_smoke.npz`.

- [ ] **Step 3: Commit**

```bash
cd /mnt/ssd/GridTransformer
git add sample_k1_transfer.py
git commit -m "Add zero-shot K=1 rail transfer sampling wrapper"
```

---

## Task 5: Run zero-shot generation at N=64 and N=125

**Files:**
- Output: `reports/k1_rail_transfer/samples_N64.npz`, `samples_N125.npz`

Use the strategy chosen in Task 3 (shown as `<STRAT>` below — substitute `pow2` or `exact`).
nsamples=256 for feasibility; N=125 sequences are long on CPU.

- [ ] **Step 1: Sample N=64 (L=4)**

```bash
cd /mnt/ssd/GridTransformer
CUDA_VISIBLE_DEVICES="" /home/sherryli/xsli/softwares/anaconda3/envs/lightning/bin/python sample_k1_transfer.py \
  --transfer_strategy <STRAT> --mode relative \
  --ckpt lj_ckpts_lj27_pbc_fixedrail_k1_ablation/lj27_pbc_continput_fixedrail_k1_fullcov/best.ckpt \
  --Lx 4 --Ly 4 --Lz 4 --coord_dim 3 --num_particles 64 --periodic --ar_arch standard \
  --use_continuous_head --full_covariance --nsamples 256 --sample_batch_size 128 \
  --temperature 0.9 --relative_window 3.0 --relative_bins 64 --device cpu \
  --save reports/k1_rail_transfer/samples_N64.npz 2>&1 | tail -5
```
Expected: `[transfer] ... L=4.0 -> R=...` line and `Saved reports/k1_rail_transfer/samples_N64.npz`.

- [ ] **Step 2: Sample N=125 (L=5)**

```bash
cd /mnt/ssd/GridTransformer
CUDA_VISIBLE_DEVICES="" /home/sherryli/xsli/softwares/anaconda3/envs/lightning/bin/python sample_k1_transfer.py \
  --transfer_strategy <STRAT> --mode relative \
  --ckpt lj_ckpts_lj27_pbc_fixedrail_k1_ablation/lj27_pbc_continput_fixedrail_k1_fullcov/best.ckpt \
  --Lx 5 --Ly 5 --Lz 5 --coord_dim 3 --num_particles 125 --periodic --ar_arch standard \
  --use_continuous_head --full_covariance --nsamples 256 --sample_batch_size 64 \
  --temperature 0.9 --relative_window 3.0 --relative_bins 64 --device cpu \
  --save reports/k1_rail_transfer/samples_N125.npz 2>&1 | tail -5
```
Expected: `Saved reports/k1_rail_transfer/samples_N125.npz` (out-of-box count should be 0).

- [ ] **Step 3: Commit**

```bash
cd /mnt/ssd/GridTransformer
git add reports/k1_rail_transfer/samples_N64.npz reports/k1_rail_transfer/samples_N125.npz
git commit -m "Zero-shot K=1 rail samples at N=64 and N=125"
```

---

## Task 6: Benchmark both sizes and write the go/no-go summary

**Files:**
- Create: `reports/k1_rail_transfer/STAGE2_FINDINGS.md`

- [ ] **Step 1: Benchmark N=64**

```bash
cd /mnt/ssd/GridTransformer
CUDA_VISIBLE_DEVICES="" /home/sherryli/xsli/softwares/anaconda3/envs/lightning/bin/python benchmark_lj27.py \
  --target /mnt/ssd/mcmc/lj_mcmc_sweep_3d/lj3d_L4_rho1.0_N64_T1.0.h5 \
  --pattern "reports/k1_rail_transfer/samples_N64.npz" --device cpu 2>&1 \
  | grep -vE "Warning|warn" | tail -12
```
Expected: a one-row table with `gr_L1`, `OTgap`, `Uc/N`, `clash%` for N=64.

- [ ] **Step 2: Benchmark N=125**

```bash
cd /mnt/ssd/GridTransformer
CUDA_VISIBLE_DEVICES="" /home/sherryli/xsli/softwares/anaconda3/envs/lightning/bin/python benchmark_lj27.py \
  --target /mnt/ssd/mcmc/lj_mcmc_sweep_3d/lj3d_L5_rho1.0_N125_T1.0.h5 \
  --pattern "reports/k1_rail_transfer/samples_N125.npz" --device cpu 2>&1 \
  | grep -vE "Warning|warn" | tail -12
```
Expected: a one-row table with metrics for N=125.

- [ ] **Step 3: Write the Stage-2 findings + final go/no-go**

Fill with actual numbers; apply the spec's OTgap bands (green ≲0.6, yellow 0.6–0.8, red ≳0.8;
in-dist N=27 K=1 OTgap ≈ 0.51, uniform = 1.0):

```markdown
# Stage 2 (zero-shot) findings — K=1 rail size transfer

Strategy used: <STRAT>   (from Stage 1)

| size | R | gr_L1 | OTgap | Uc/N | clash% | g(r) peak r | band |
|------|---|-------|-------|------|--------|-------------|------|
| N=64  | <fill> | <fill> | <fill> | <fill> | <fill> | <fill> | <green/yellow/red> |
| N=125 | <fill> | <fill> | <fill> | <fill> | <fill> | <fill> | <green/yellow/red> |

Reference: in-dist N=27 K=1 OTgap≈0.51, g(r) L1≈0.19; uniform OTgap=1.0.

**Verdict:** <FEASIBLE (zero-shot) | MARGINAL (fine-tune) | NOT FEASIBLE>
**Recommended next step:** <proceed to multi-size training | fine-tune at target size | stop / revisit representation>
```

- [ ] **Step 4: Commit**

```bash
cd /mnt/ssd/GridTransformer
git add reports/k1_rail_transfer/STAGE2_FINDINGS.md
git commit -m "K=1 rail size-transfer feasibility: Stage-2 benchmark + go/no-go"
```

---

## Self-Review notes

- **Spec coverage:** Stage 1 audit (Tasks 1–2), both R strategies + in-box check (choose_R/in_box_fraction, Task 2 loop), rail-input + cartesian-delta KS (Task 2), decision gate (Task 3), zero-shot generation with cell_size injection (Tasks 4–5), benchmark + tiered OTgap go/no-go (Task 6). All spec sections map to a task.
- **Sample-independence:** rail waypoints come from `fixed_template_waypoints` (depends only on N/R/L/j), matching the model's actual conditioning; MCMC data is used only for the cartesian-delta target — consistent with the spec's "cell_size enters only the rail input" finding.
- **Naming consistency:** `choose_R`, `rail_waypoints_absolute`, `in_box_fraction`, `CELL_TRAIN`, `cartesian_delta_targets`, `_strategy_R` used identically across tasks and tests.
- **Strategy placeholder:** `<STRAT>` in Tasks 5–6 is intentionally resolved by the Task-3 decision (pow2 or exact), not a plan gap.
