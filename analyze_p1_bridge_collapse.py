"""P1 probe: does a bridge normalization collapse the absolute-anchor residual across sizes?

Theory: s-space order-statistic bridge std ~ sqrt(j(N-j)/N) cells, pushed through Hilbert
locality |dpos| ~ |ds|^(1/3), gives per-axis position residual std

    std_j ≈ C * (j(N-j)/N)^(1/6)        (at fixed density, independent of L and R)

Endpoint check against SIZE_TRANSFER_FINDINGS.md §5: pooled std ratios L5/L3, L10/L3
match N^(1/6) to 1-3%. This probe tests the per-index SHAPE: divide the measured
per-index std curves by the profile and check L3/L5/L10 overlap (pass ~<=10% spread).

Gate for action-plan Arm C' (normalized absolute anchor target).
"""
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from analyze_anchor_residual_perindex import SIZES, collect

OUT = "reports/multisize_arc/figs"
FRAC_GRID = np.linspace(0.02, 0.98, 97)  # avoid endpoint noise where profile -> 0


def bridge_profile(j: np.ndarray, N: int, alpha: float = 1.0 / 6.0) -> np.ndarray:
    return (j * (N - j) / N) ** alpha


def cross_size_spread(curves: dict[str, np.ndarray]) -> float:
    """Mean over the fractional grid of (max-min)/mean across sizes."""
    M = np.stack(list(curves.values()))  # [n_sizes, n_grid]
    return float(np.mean((M.max(0) - M.min(0)) / M.mean(0)))


def main():
    data = {tag: collect(tag) for tag in SIZES}

    raw, glob_norm, prof_norm, prev_raw = {}, {}, {}, {}
    for tag, (A, P, L) in data.items():
        N = A.shape[1] + 1  # residuals exist for j = 1..N-1
        j = np.arange(1, N)
        frac = j / N
        std_abs = A.std(0).mean(-1)  # per-index std, averaged over xyz axes
        std_prev = P.std(0).mean(-1)
        raw[tag] = np.interp(FRAC_GRID, frac, std_abs)
        glob_norm[tag] = np.interp(FRAC_GRID, frac, std_abs / float(N) ** (1.0 / 6.0))
        prof_norm[tag] = np.interp(FRAC_GRID, frac, std_abs / bridge_profile(j, N))
        prev_raw[tag] = np.interp(FRAC_GRID, frac, std_prev)

    # best-fit exponent per size: log std_j = log C + alpha * log(j(N-j)/N)
    print("=== per-size best-fit exponent alpha (theory: 1/6 = 0.167) ===")
    for tag, (A, P, L) in data.items():
        N = A.shape[1] + 1
        j = np.arange(1, N)
        keep = (j / N >= FRAC_GRID[0]) & (j / N <= FRAC_GRID[-1])
        x = np.log(j[keep] * (N - j[keep]) / N)
        y = np.log(A.std(0).mean(-1)[keep])
        alpha, logC = np.polyfit(x, y, 1)
        print(f"  {tag:10s} alpha={alpha:.3f}  C={np.exp(logC):.3f}")

    metrics = {
        "raw (no normalization)": cross_size_spread(raw),
        "global N^(1/6)": cross_size_spread(glob_norm),
        "per-index (j(N-j)/N)^(1/6)": cross_size_spread(prof_norm),
        "prev-particle control (known good)": cross_size_spread(prev_raw),
    }
    print("\n=== cross-size spread, mean (max-min)/mean over fractional index ===")
    for k, v in metrics.items():
        print(f"  {k:38s} {v:.3f}")

    fig, ax = plt.subplots(1, 4, figsize=(22, 4.5), sharex=True)
    panels = [
        (raw, "ABS anchor std (raw)"),
        (glob_norm, "ABS / N^(1/6) (global norm)"),
        (prof_norm, "ABS / (j(N-j)/N)^(1/6) (bridge profile)"),
        (prev_raw, "PREV-particle std (control)"),
    ]
    for a, (curves, title) in zip(ax, panels):
        for tag, c in curves.items():
            a.plot(FRAC_GRID, c, label=tag)
        spread = cross_size_spread(curves)
        a.set_title(f"{title}\nspread={spread:.3f}")
        a.set_xlabel("j / N")
        a.legend(fontsize=8)
    ax[0].set_ylabel("per-axis std")
    fig.suptitle("P1: bridge-normalization collapse of the absolute-anchor residual")
    fig.tight_layout()
    fig.savefig(f"{OUT}/p1_bridge_collapse.png", dpi=110)
    print(f"\nwrote {OUT}/p1_bridge_collapse.png")

    gate = metrics["per-index (j(N-j)/N)^(1/6)"]
    ctrl = metrics["prev-particle control (known good)"]
    verdict = "PASS" if gate <= max(0.10, 1.5 * ctrl) else "FAIL"
    print(f"\nP1 verdict: {verdict} (bridge-norm spread {gate:.3f} vs gate "
          f"max(0.10, 1.5*control={1.5*ctrl:.3f}))")


if __name__ == "__main__":
    main()
