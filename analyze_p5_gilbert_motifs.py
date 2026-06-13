"""P5 probe: are gilbert curve motifs / data jump statistics stationary across R?

Two measurements, kept separate per the action plan
(reports/2026-06-12-size-transfer-action-plan.md):

(a) DATA jump fraction: fraction of curve-sorted MCMC transitions with
    normalized Δs = (s_{j+1}-s_j)/X > 4.0 (model default jump_delta_s_threshold),
    for gilbert at every R in [8,20] plus production-scale even R, vs the
    resolution-matched pow2 Hilbert baselines (R=8,16).
(b) CURVE-intrinsic motif stats: straight-step fraction and multi-cell-step
    fraction along the bare curve path (no data), same R values.

Gate: if the gilbert jump fraction varies by more than 2x across R in [8,20],
restrict the ladder to R within 30% of the nearest pow2 baseline.
"""
import h5py
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from grid_transformer.data.curves import get_curve3d

OUT = "reports/multisize_arc/figs"
JUMP_THRESHOLD = 4.0  # GraphormerAR.jump_delta_s_threshold default
R_SCAN = list(range(8, 21))
DATA = {  # tag: (h5, N, L)
    "L4_N64": ("/mnt/ssd/mcmc/lj_mcmc_sweep_3d/lj3d_L4_rho1.0_N64_T1.0.h5", 64, 4.0),
    "L5_N125": ("/mnt/ssd/mcmc/lj_mcmc_sweep_3d/lj3d_L5_rho1.0_N125_T1.0.h5", 125, 5.0),
}
# production-scale even gilbert R for constant cell ~1/21.3 (cell 0.046875-ish)
PROD_R = {"L4_N64": 86, "L5_N125": 106}
N_FRAMES = 400


def curve_intrinsic_stats(ordering: str, R: int) -> dict:
    curve = get_curve3d(ordering, R)
    if ordering == "hilbert":
        path = curve.decode(np.arange(curve.ncells))
    else:
        path = curve.coords
    d = np.diff(path, axis=0)
    step_len = np.sqrt((d.astype(np.float64) ** 2).sum(-1))
    straight = np.all(d[1:] == d[:-1], axis=1)
    return {
        "straight_frac": float(straight.mean()),
        "multicell_frac": float((step_len > 1.0 + 1e-9).mean()),
        "max_step": float(step_len.max()),
    }


def data_jump_fraction(tag: str, ordering: str, R: int, traj: np.ndarray, L: float) -> dict:
    curve = get_curve3d(ordering, R)
    N = traj.shape[1]
    X = max(1, curve.ncells // N)
    cell = L / R
    jump_fracs, ds_all = [], []
    for pos in traj:
        g = np.clip(np.floor(np.mod(pos, L) / cell).astype(np.int64), 0, R - 1)
        codes = np.sort(curve.encode(g))
        ds = np.diff(curve.arc(codes)) / X
        jump_fracs.append((ds > JUMP_THRESHOLD).mean())
        ds_all.append(ds)
    ds_all = np.concatenate(ds_all)
    return {
        "jump_frac": float(np.mean(jump_fracs)),
        "ds_mean": float(ds_all.mean()),
        "ds_p99": float(np.quantile(ds_all, 0.99)),
    }


def main():
    # ---- (b) curve-intrinsic motifs ----
    print("=== curve-intrinsic motif stats ===")
    intrinsic = {}
    for R in R_SCAN + sorted(set(PROD_R.values())):
        intrinsic[("gilbert", R)] = curve_intrinsic_stats("gilbert", R)
    for R in (8, 16, 64):
        intrinsic[("hilbert", R)] = curve_intrinsic_stats("hilbert", R)
    for (fam, R), st in sorted(intrinsic.items()):
        print(f"  {fam:8s} R={R:3d}  straight={st['straight_frac']:.4f}  "
              f"multicell={st['multicell_frac']:.5f}  max_step={st['max_step']:.2f}")

    # ---- (a) data jump fraction ----
    results = {}
    for tag, (path, N, L) in DATA.items():
        with h5py.File(path, "r") as f:
            traj = f["traj"][:N_FRAMES].reshape(-1, N, 3)
        print(f"\n=== data jump fraction: {tag} ({traj.shape[0]} frames) ===")
        for R in R_SCAN + [PROD_R[tag]]:
            results[(tag, "gilbert", R)] = data_jump_fraction(tag, "gilbert", R, traj, L)
        for R in (8, 16):
            results[(tag, "hilbert", R)] = data_jump_fraction(tag, "hilbert", R, traj, L)
        for (t2, fam, R), st in sorted(results.items()):
            if t2 == tag:
                print(f"  {fam:8s} R={R:3d}  jump_frac={st['jump_frac']:.4f}  "
                      f"ds_mean={st['ds_mean']:.3f}  ds_p99={st['ds_p99']:.2f}")

    # ---- gate ----
    print("\n=== P5 gate: gilbert jump-fraction variation across R in [8,20] ===")
    verdicts = {}
    for tag in DATA:
        jf = {R: results[(tag, "gilbert", R)]["jump_frac"] for R in R_SCAN}
        ratio = max(jf.values()) / max(min(jf.values()), 1e-12)
        base = {R: results[(tag, "hilbert", R)]["jump_frac"] for R in (8, 16)}
        ok_R = [
            R for R in R_SCAN
            if abs(jf[R] - base[min(base, key=lambda b: abs(b - R))])
            <= 0.30 * base[min(base, key=lambda b: abs(b - R))]
        ]
        verdicts[tag] = (ratio, ok_R)
        status = "PASS" if ratio <= 2.0 else "FAIL"
        print(f"  {tag}: max/min ratio = {ratio:.2f} -> {status}"
              + ("" if ratio <= 2.0 else f"; R within 30% of pow2 baseline: {ok_R}"))

    # ---- figure ----
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.5))
    for tag in DATA:
        jfs = [results[(tag, "gilbert", R)]["jump_frac"] for R in R_SCAN]
        ax[0].plot(R_SCAN, jfs, "o-", label=f"gilbert {tag}")
        for R in (8, 16):
            ax[0].plot([R], [results[(tag, "hilbert", R)]["jump_frac"]], "k*", ms=12)
        pr = PROD_R[tag]
        ax[0].plot([pr], [results[(tag, "gilbert", pr)]["jump_frac"]], "s", ms=9)
    ax[0].set_title(f"data jump fraction (Δs > {JUMP_THRESHOLD})\nstars = pow2 hilbert baseline, squares = production R")
    ax[0].set_xlabel("R"); ax[0].legend(fontsize=8)
    Rs_g = R_SCAN + sorted(set(PROD_R.values()))
    ax[1].plot(Rs_g, [intrinsic[("gilbert", R)]["straight_frac"] for R in Rs_g], "o-", label="gilbert")
    for R in (8, 16, 64):
        ax[1].plot([R], [intrinsic[("hilbert", R)]["straight_frac"]], "k*", ms=12)
    ax[1].set_title("curve straight-step fraction"); ax[1].set_xlabel("R"); ax[1].legend(fontsize=8)
    ax[2].plot(Rs_g, [intrinsic[("gilbert", R)]["multicell_frac"] for R in Rs_g], "o-", label="gilbert")
    ax[2].set_title("curve multi-cell step fraction (odd-grid artifact)"); ax[2].set_xlabel("R"); ax[2].legend(fontsize=8)
    fig.suptitle("P5: gilbert motif/jump stationarity across R")
    fig.tight_layout()
    fig.savefig(f"{OUT}/p5_gilbert_motifs.png", dpi=110)
    print(f"\nwrote {OUT}/p5_gilbert_motifs.png")


if __name__ == "__main__":
    main()
