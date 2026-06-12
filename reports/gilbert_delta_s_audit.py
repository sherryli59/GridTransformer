"""Gilbert vs Hilbert Δs audit: constant-cell size-invariance across L3/L4/L5.

Hypothesis: gilbert constant-cell targets are MORE size-invariant (lower KS vs the L3
reference) than hilbert octave-rounded ones.  Previous hilbert-only audit found KS ≤ 0.044
(reports/arc_repr_delta_s/FINDINGS.md).

Production rules:
  hilbert: R = next_pow2(round(L / CELL)) — cell is constant only within each octave.
  gilbert: R = nearest even integer of L/CELL    — cell ≈ CELL to <1%.

Outputs audit table and verdict to stdout AND reports/gilbert_delta_s_audit.out.

CPU only — no GPU needed; runs in ~1 minute (gilbert LUT at R=86/106 builds once
and is disk-cached under ~/.cache/grid_transformer/gilbert).
"""
from __future__ import annotations

import sys
import os

import h5py
import numpy as np
from scipy.stats import ks_2samp

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CELL = 3.0 / 64          # training cell size: 0.046875 exactly
MAX_CONFIGS = 1000        # per file (= n_chains in the current h5 files; we take
                          # the last frame of each independent chain)

DATA_FILES = {
    "L3": "/mnt/ssd/mcmc/lj_mcmc_sweep_3d/lj3d_L3_rho1.0_N27_T1.0.h5",
    "L4": "/mnt/ssd/mcmc/lj_mcmc_sweep_3d/lj3d_L4_rho1.0_N64_T1.0.h5",
    "L5": "/mnt/ssd/mcmc/lj_mcmc_sweep_3d/lj3d_L5_rho1.0_N125_T1.0.h5",
}

OUTPUT_FILE = os.path.join(os.path.dirname(__file__), "gilbert_delta_s_audit.out")

# ---------------------------------------------------------------------------
# Imports from package (task constraint: no modifications to package code)
# ---------------------------------------------------------------------------
from grid_transformer.data.curves import get_curve3d
from grid_transformer.data.lj_transferable import (
    hilbert_arc_delta,
    resolution_for_box_rule,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_configs(path: str, N: int, max_configs: int) -> np.ndarray:
    """Load up to max_configs positions from the MCMC h5.

    Real files: traj shape [chains, steps, N, 3].  Read [:max_configs] from
    the last frame of each chain (well-decorrelated).
    """
    with h5py.File(path, "r") as f:
        traj = f["traj"]  # [chains, steps, N, 3]
        n_chains = traj.shape[0]
        n_take = min(max_configs, n_chains)
        configs = traj[:n_take, -1, :, :].astype(np.float64)
    return configs


def arc_targets_for_configs(configs: np.ndarray, L: float, R: int, ordering: str) -> np.ndarray:
    """Compute stacked [n*(N-1), 4] arc targets for all configs."""
    box = np.array([L, L, L], dtype=np.float64)
    curve = get_curve3d(ordering, R)
    cell = box / float(R)
    all_targets = []
    for pos in configs:
        wrapped = np.mod(pos.astype(np.float64), box)
        grid = np.clip(np.floor(wrapped / cell).astype(np.int64), 0, R - 1)
        codes = curve.encode(grid)
        order = np.argsort(codes, kind="stable")
        arc = hilbert_arc_delta(wrapped[order], codes[order], box, R, periodic=True, curve=curve)
        all_targets.append(arc)
    return np.concatenate(all_targets, axis=0)


def gilbert_diagonal_fraction(R: int) -> float:
    """Fraction of steps in the gilbert curve at R that are NOT face-adjacent (length != 1)."""
    from grid_transformer.data.curves import GilbertCurve3D
    curve = GilbertCurve3D(R)
    # cumlen differences give step lengths in cell units
    step_lens = np.diff(curve.cumlen)
    return float(np.mean(np.abs(step_lens - 1.0) > 1e-9))


def resolution_for(L: float, ordering: str) -> int:
    """Apply production resolution rule for a given box size and ordering."""
    box = np.array([L, L, L])
    # hilbert_resolution is the fallback; with cell_size set it uses the rule
    return resolution_for_box_rule(box, cell_size=CELL, hilbert_resolution=64, ordering=ordering)


def ks_stat(a: np.ndarray, b: np.ndarray) -> float:
    return float(ks_2samp(a, b).statistic)


# ---------------------------------------------------------------------------
# Main audit
# ---------------------------------------------------------------------------

def main():
    lines = []

    def out(s: str = ""):
        print(s)
        lines.append(s)

    out("=" * 78)
    out("Gilbert vs Hilbert Δs audit — constant-cell size-invariance")
    out(f"CELL = {CELL:.6f}  MAX_CONFIGS = {MAX_CONFIGS}")
    out("=" * 78)

    # Verify data files exist
    for label, path in DATA_FILES.items():
        if not os.path.exists(path):
            out(f"ERROR: {label} file not found: {path}")
            sys.exit(1)

    sizes = [
        ("L3", 3.0, 27),
        ("L4", 4.0, 64),
        ("L5", 5.0, 125),
    ]

    # Pre-compute resolutions for each ordering and size
    out("\n--- Resolution rules ---")
    out(f"{'size':4s}  {'L':5s}  {'N':5s}  {'R_hilbert':10s}  {'cell_hilbert':12s}  {'dev_h%':7s}  "
        f"{'R_gilbert':10s}  {'cell_gilbert':12s}  {'dev_g%':7s}")
    for label, L, N in sizes:
        R_h = resolution_for(L, "hilbert")
        R_g = resolution_for(L, "gilbert")
        ch = L / R_h
        cg = L / R_g
        dh = (ch - CELL) / CELL * 100
        dg = (cg - CELL) / CELL * 100
        out(f"{label:4s}  {L:5.1f}  {N:5d}  {R_h:10d}  {ch:12.6f}  {dh:+7.2f}%  "
            f"{R_g:10d}  {cg:12.6f}  {dg:+7.2f}%")

    # Gilbert diagonal-step fraction per R
    out("\n--- Gilbert diagonal-step fraction ---")
    out("(even R expected to be 0: all steps face-adjacent)")
    gilbert_Rs = {}
    for label, L, N in sizes:
        R_g = resolution_for(L, "gilbert")
        gilbert_Rs[label] = R_g
    for label, R in gilbert_Rs.items():
        diag_frac = gilbert_diagonal_fraction(R)
        out(f"  {label}: R={R}  diagonal-step fraction = {diag_frac:.6f}  "
            f"({'fully face-continuous' if diag_frac == 0.0 else f'{diag_frac*100:.3f}% non-unit steps'})")

    # Collect arc targets for all sizes and orderings
    out("\n--- Computing arc targets ---")
    results = {}  # (ordering, label) -> {ds, fine}
    for ordering in ("hilbert", "gilbert"):
        for label, L, N in sizes:
            R = resolution_for(L, ordering)
            configs = load_configs(DATA_FILES[label], N, MAX_CONFIGS)
            n_loaded = len(configs)
            out(f"  {ordering:7s}  {label}: L={L}, N={N}, R={R}, configs={n_loaded}")
            arc = arc_targets_for_configs(configs, L, R, ordering)
            results[(ordering, label)] = {
                "ds": arc[:, 0].astype(np.float64),
                "fine": arc[:, 1:4].astype(np.float64),
                "R": R,
                "L": L,
                "N": N,
            }

    # Build summary table
    out("\n--- KS statistics vs L3 reference (per ordering) ---")
    header = (f"{'ordering':8s}  {'size':4s}  {'R':6s}  {'cell':8s}  {'dev%':6s}  "
              f"{'mean_ds':8s}  {'p50_ds':8s}  {'p90_ds':8s}  "
              f"{'KS_ds':7s}  {'KS_fine':8s}  {'jump_frac':10s}")
    out(header)
    out("-" * len(header))

    ks_table = {}  # (ordering, label) -> KS_ds
    for ordering in ("hilbert", "gilbert"):
        ref = results[(ordering, "L3")]
        for label, L, N in sizes:
            r = results[(ordering, label)]
            ds = r["ds"]
            fine = r["fine"]
            R = r["R"]
            cell = L / R
            dev = (cell - CELL) / CELL * 100

            mean_ds = float(ds.mean())
            p50_ds = float(np.percentile(ds, 50))
            p90_ds = float(np.percentile(ds, 90))
            jump_frac = float(np.mean(np.abs(ds) > 4.0))

            if label == "L3":
                ks_ds = 0.0
                ks_fine = 0.0
            else:
                ks_ds = ks_stat(ref["ds"], ds)
                ks_fine = max(ks_stat(ref["fine"][:, d], fine[:, d]) for d in range(3))

            ks_table[(ordering, label)] = ks_ds
            out(f"{ordering:8s}  {label:4s}  {R:6d}  {cell:8.5f}  {dev:+6.2f}%  "
                f"{mean_ds:8.4f}  {p50_ds:8.4f}  {p90_ds:8.4f}  "
                f"{ks_ds:7.4f}  {ks_fine:8.4f}  {jump_frac:10.6f}")
        out("")

    # Verdict
    out("--- Verdict ---")
    h_l4 = ks_table[("hilbert", "L4")]
    h_l5 = ks_table[("hilbert", "L5")]
    g_l4 = ks_table[("gilbert", "L4")]
    g_l5 = ks_table[("gilbert", "L5")]

    gilbert_wins_l4 = g_l4 <= h_l4
    gilbert_wins_l5 = g_l5 <= h_l5
    both_small = all(v <= 0.10 for v in [h_l4, h_l5, g_l4, g_l5])

    out(f"KS(Δs) L4: hilbert={h_l4:.4f}  gilbert={g_l4:.4f}  -> {'gilbert ≤ hilbert' if gilbert_wins_l4 else 'hilbert < gilbert'}")
    out(f"KS(Δs) L5: hilbert={h_l5:.4f}  gilbert={g_l5:.4f}  -> {'gilbert ≤ hilbert' if gilbert_wins_l5 else 'hilbert < gilbert'}")

    # NOTE on what this audit can and cannot decide: it compares MARGINAL
    # (Δs, fine) distributions. Gilbert's hypothesized benefit — constant
    # physical cell size ℓ so the learned CONDITIONAL p(Δs, fine | context)
    # transfers across box sizes — is structurally invisible to a marginal
    # KS (the fine marginal is ~uniform for any ℓ << sigma). The audit can
    # only flag a marginal-level regression; it cannot confirm or refute the
    # conditional-level hypothesis. The decisive test is the model-level A/B:
    # held-out L4 NLL/OTgap with ORDERING=gilbert (matched cell) vs the
    # hilbert octave-cell baseline.
    if not both_small:
        out(f"\nVERDICT: NO-GO — KS values are large (L4_h={h_l4:.3f}, L4_g={g_l4:.3f}, "
            f"L5_h={h_l5:.3f}, L5_g={g_l5:.3f}). Both curves show marginal distribution "
            f"mismatch; investigate before any training.")
    else:
        better = "gilbert" if (gilbert_wins_l4 and gilbert_wins_l5) else "hilbert"
        out("\nVERDICT: MARGINALS PASS for BOTH orderings (all KS(Δs) < 0.10) — no "
            f"marginal-level blocker for either curve; {better} is marginally tighter.")
        out("This audit does NOT decide the gilbert hypothesis (constant-cell benefit "
            "lives in the conditional, invisible to marginal KS). GO/NO-GO for gilbert "
            "requires the model-level A/B: held-out-L4 NLL/OTgap at matched cell "
            "(ORDERING=gilbert) vs the octave-cell hilbert baseline.")

    out("\n(End of audit)")

    # Write to output file
    with open(OUTPUT_FILE, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\n[Saved to {OUTPUT_FILE}]")


if __name__ == "__main__":
    main()
