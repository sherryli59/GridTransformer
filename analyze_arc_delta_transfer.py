"""Step-3 audit: is the arc_repr (Δs, fine) target size-invariant across box sizes?

Context (CODE_REVIEW_hilbert-arc-repr_2026-06-10.md §5, roadmap step 3): arc_repr absorbs
Hilbert turns into the fixed s↔xyz map, but Δs is not turn-free — a turn appears as a
large Δs outlier the MDN must express. Before training, check whether the Δs tail at
L=4/L=5 (pow2 constant-cell transfer setting) matches L=3. Heavy tails that grow with L
mean the turn-expression problem was moved, not removed.

Sizes/resolutions mirror the K=1 study (pow2: L=3→R=64, L=4→128, L=5→128). A control
row (N=27 at R=128) isolates the cell-size effect from the box-size effect.

Outputs: stats table + KS vs the N=27 reference + overlay plots in reports/arc_repr_delta_s/.
"""
from __future__ import annotations

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import ks_2samp

from analyze_k1_rail_transfer import CELL_TRAIN, SIZES, choose_R, load_configs
from grid_transformer.data.lj_transferable import (
    _hilbert3d_encode,
    _hilbert_bits,
    hilbert_arc_delta,
)


def tail_fraction(x: np.ndarray, threshold: float) -> float:
    """Fraction of values strictly greater than ``threshold``."""
    return float(np.mean(np.asarray(x) > threshold))


def sorted_arc_targets(pos: np.ndarray, L: float, R: int) -> np.ndarray:
    """(Δs, fine) targets for one configuration, exactly as training computes them:
    Hilbert-sort at resolution R, then ``hilbert_arc_delta``. Returns [N-1, 4]."""
    box = np.array([L, L, L], dtype=np.float64)
    bits = _hilbert_bits(R)
    cell = box / float(R)
    grid = np.clip(
        np.floor(np.mod(pos.astype(np.float64), box) / cell).astype(np.int64), 0, R - 1
    )
    codes = _hilbert3d_encode(grid[:, 0], grid[:, 1], grid[:, 2], bits=bits)
    order = np.argsort(codes, kind="stable")
    return hilbert_arc_delta(pos[order], codes[order], box, R, periodic=True)


def arc_targets_for_configs(configs: np.ndarray, L: float, R: int) -> np.ndarray:
    """Concatenated [n*(N-1), 4] arc targets over a batch of configurations."""
    return np.concatenate([sorted_arc_targets(p, L, R) for p in configs], axis=0)


def _summarize(name: str, ds: np.ndarray) -> dict:
    q = np.quantile(ds, [0.5, 0.9, 0.99, 0.999])
    return dict(
        name=name,
        mean=float(ds.mean()),
        p50=float(q[0]),
        p90=float(q[1]),
        p99=float(q[2]),
        p999=float(q[3]),
        max=float(ds.max()),
        tail2=tail_fraction(ds, 2.0),
        tail5=tail_fraction(ds, 5.0),
        tail10=tail_fraction(ds, 10.0),
    )


def main(args) -> None:
    os.makedirs(args.out_dir, exist_ok=True)

    cases = []
    for s in SIZES:
        R = choose_R(s["L"], "pow2")
        cases.append(dict(label=f"N={s['N']} L={s['L']} R={R}", N=s["N"], L=s["L"], R=R, path=s["path"]))
    # Control: training size at the transfer resolution — isolates cell-size effect.
    cases.append(dict(label="N=27 L=3.0 R=128 (control)", N=27, L=3.0, R=128, path=SIZES[0]["path"]))

    results = {}
    for c in cases:
        cfgs = load_configs(c["path"], c["N"], args.n_configs, seed=args.seed)
        arc = arc_targets_for_configs(cfgs, c["L"], c["R"])
        results[c["label"]] = dict(ds=arc[:, 0].astype(np.float64), fine=arc[:, 1:4].astype(np.float64), **c)

    ref_label = cases[0]["label"]  # N=27 R=64, the training configuration
    ref = results[ref_label]

    print(f"\nΔs target statistics (X = R³//N; mean Δs ≈ 1 by construction; cell={CELL_TRAIN:.4f} at train)")
    hdr = f"{'case':32s} {'cell':>6s} {'mean':>6s} {'p50':>6s} {'p90':>6s} {'p99':>7s} {'p99.9':>8s} {'max':>9s} {'P>2':>8s} {'P>5':>8s} {'P>10':>8s} {'KS_ds':>6s} {'KS_fine':>7s}"
    print(hdr)
    print("-" * len(hdr))
    for label, r in results.items():
        st = _summarize(label, r["ds"])
        ks_ds = float(ks_2samp(ref["ds"], r["ds"]).statistic) if label != ref_label else 0.0
        ks_fine = (
            max(float(ks_2samp(ref["fine"][:, d], r["fine"][:, d]).statistic) for d in range(3))
            if label != ref_label
            else 0.0
        )
        cell = r["L"] / r["R"]
        print(
            f"{label:32s} {cell:6.4f} {st['mean']:6.3f} {st['p50']:6.3f} {st['p90']:6.3f} "
            f"{st['p99']:7.2f} {st['p999']:8.2f} {st['max']:9.1f} "
            f"{st['tail2']:8.5f} {st['tail5']:8.5f} {st['tail10']:8.5f} {ks_ds:6.3f} {ks_fine:7.3f}"
        )

    # Plots: Δs on log-y (tails are the question) + fine_x overlay.
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    for label, r in results.items():
        ds = r["ds"]
        axes[0].hist(np.clip(ds, -2, 20), bins=110, density=True, histtype="step", label=label)
        axes[1].hist(np.clip(ds, -2, 20), bins=110, density=True, histtype="step", label=label, log=True)
        axes[2].hist(r["fine"][:, 0], bins=60, density=True, histtype="step", label=label)
    axes[0].set_title("Δs (clipped to [-2, 20])")
    axes[1].set_title("Δs — log scale (tail comparison)")
    axes[2].set_title("fine_x (cell-normalized)")
    for ax in axes:
        ax.legend(fontsize=7)
    plt.tight_layout()
    out = os.path.join(args.out_dir, "arc_delta_s_transfer.png")
    plt.savefig(out, dpi=140, bbox_inches="tight")
    plt.close()
    print(f"\nSaved {out}")

    # Verdict guidance: compare transfer-size tails to the train-size reference.
    print("\nVerdict guidance:")
    ref_t5 = tail_fraction(ref["ds"], 5.0)
    for label, r in results.items():
        if label == ref_label:
            continue
        t5 = tail_fraction(r["ds"], 5.0)
        ratio = t5 / max(ref_t5, 1e-12)
        flag = "OK" if ratio < 3.0 else "HEAVY TAIL (turn problem moved into Δs)"
        print(f"  {label:32s} P(Δs>5) = {t5:.5f} ({ratio:5.1f}x train ref) -> {flag}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_configs", type=int, default=500)
    ap.add_argument("--out_dir", type=str, default="reports/arc_repr_delta_s")
    ap.add_argument("--seed", type=int, default=42)
    main(ap.parse_args())
