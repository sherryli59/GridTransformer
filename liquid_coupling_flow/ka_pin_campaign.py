"""Point-to-set campaign: run the T x c ladder (random pinning), gate each cell, extract xi_pin(T).
All learned parts N=100-trained, zero-shot at N>=256, behind exact Metropolis. Saves per-cell trajectories
incrementally and the final xi(T) figure."""
import os, torch, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from liquid_coupling_flow.ka_pin import pin_mask, constrained_run
from liquid_coupling_flow.ka_pin_extract import stretched_exp_fit, xi_threshold
from liquid_coupling_flow.ka_pin_gates import g_conv
from liquid_coupling_flow.ka_pin_overlap import q_rand

RHO = 1.2


def _plot_xi_of_T(agg, out_path, thresholds=(0.1, 0.2, 0.3)):
    """Plot xi_pin(T) per threshold. Handles all xi_threshold kinds: point (float) -> point+dxi bar;
    range (tuple) -> midpoint with half-range bar; none -> skipped. Returns warning strings for range/none."""
    import matplotlib.pyplot as plt
    warnings = []
    fig, ax = plt.subplots(figsize=(6, 4.4))
    for thr, mk in zip(thresholds, ("o", "s", "^")):
        Ts = sorted(agg[thr].keys())
        xs, ys, yerr = [], [], []
        for T in Ts:
            r = agg[thr][T]
            if r["kind"] == "point":
                xs.append(T); ys.append(r["xi"]); yerr.append(r["dxi"] or 0.0)
            elif r["kind"] == "range":
                lo, hi = r["xi"]; xs.append(T); ys.append(0.5 * (lo + hi)); yerr.append(0.5 * (hi - lo))
                warnings.append(f"T={T} thr={thr}: xi RANGE [{lo:.2f},{hi:.2f}] (non-monotone Qinf)")
            else:
                warnings.append(f"T={T} thr={thr}: NO crossing (xi undefined)")
        ax.errorbar(xs, ys, yerr=yerr, marker=mk, label=f"thr={thr}", capsize=3)
    ax.set_xlabel("T"); ax.set_ylabel(r"$\xi_{pin}$"); ax.legend(); ax.set_title("Point-to-set length vs T")
    fig.tight_layout(); fig.savefig(out_path, dpi=130); plt.close(fig)
    return warnings


def run_cell(T, c, refs, n_iter, table_fn, device, out_dir, n_real=16, dt=0.01, record_every=5):
    os.makedirs(out_dir, exist_ok=True)
    x = refs["x"][:n_real].to(device); s = refs["s"].to(device).long()   # species as int64 (dtype-safe kernels)
    if s.dim() == 1:
        s = s[None].expand(n_real, -1).contiguous()
    N = x.shape[1]; L = refs["L"]
    gen = torch.Generator(device=device); gen.manual_seed(hash((round(T, 3), round(c, 3))) % (2**31))
    mobile = pin_mask(n_real, N, c, device, generator=gen)
    torch.manual_seed(hash((round(T, 3), round(c, 3), "scr")) % (2**31))
    scr = torch.rand_like(x) * L
    ref_run = constrained_run(x, s, mobile, T, L, n_iter, table_fn, dt, record_every, arm="ref")
    scr_run = constrained_run(x, s, mobile, T, L, n_iter, table_fn, dt, record_every,
                              arm="scramble", scramble_x=scr)
    gc = g_conv(ref_run, scr_run)
    # per-realization Q_inf -> REAL bootstrap error over pinning realizations (not a stub), from the ref arm's
    # per-chain Q(t) (Q_chain). The ref arm decays, so stretched_exp_fit's default rising=False is correct.
    t = np.asarray(ref_run["t"], float)
    Qc = np.stack([q.numpy() for q in ref_run["Q_chain"]])              # [n_steps, n_real]
    qinf_b = []
    for b in range(Qc.shape[1]):
        f = stretched_exp_fit(t, Qc[:, b])
        qinf_b.append(f["Qinf"] if f["ok"] else float(Qc[-max(1, len(t) // 5):, b].mean()))
    qinf_b = np.asarray(qinf_b, float)
    Qinf = float(np.median(qinf_b))
    Qinf_err = float(qinf_b.std(ddof=1) / np.sqrt(len(qinf_b))) if len(qinf_b) > 1 else 0.02
    lc = (c * RHO) ** -0.5
    passed = bool(gc["passed"] and abs(Qinf - gc["q_ref"]) <= 0.02)   # gate + estimator-consistency
    cell = {"T": T, "c": c, "lc": lc, "Qinf": Qinf, "Qinf_err": Qinf_err, "passed": passed,
            "gconv": gc, "xB": float((s[mobile] == 1).float().mean()),
            "arms": {"ref": ref_run, "scramble": scr_run}}
    torch.save(cell, os.path.join(out_dir, f"cell_T{T}_c{c}.pt"))       # incremental save
    print(f"[pts] T={T} c={c} lc={lc:.2f} Qinf={Qinf:.3f}+/-{Qinf_err:.3f} "
          f"{'PASS' if passed else 'BOUND'} -> {os.path.join(out_dir, f'cell_T{T}_c{c}.pt')}", flush=True)
    return cell


def aggregate(cells, thresholds=(0.1, 0.2, 0.3), Qrand=None):
    """Per threshold, per T: fit xi_threshold ONLY from cells that passed G-conv (cell['passed']); cells that
    failed are recorded as BOUNDS (their lc), never fitted (spec: unconverged -> bound). Cells lacking a
    'passed' key default to passing (synthetic/manual use). Fewer than 2 passing points -> kind='none'."""
    Qrand = q_rand() if Qrand is None else Qrand
    by_T = {}
    for cl in cells:
        by_T.setdefault(cl["T"], []).append(cl)
    out = {}
    for thr in thresholds:
        out[thr] = {}
        for T, cs in by_T.items():
            passing = sorted([z for z in cs if z.get("passed", True)], key=lambda z: z["lc"])
            bound_lc = sorted([z["lc"] for z in cs if not z.get("passed", True)])
            if len(passing) < 2:
                out[thr][T] = {"xi": None, "dxi": None, "kind": "none",
                               "n_pass": len(passing), "n_bound": len(bound_lc), "bound_lc": bound_lc}
                continue
            lv = np.array([z["lc"] for z in passing]); qi = np.array([z["Qinf"] for z in passing])
            qe = np.array([z.get("Qinf_err", 0.01) for z in passing])
            r = xi_threshold(lv, qi, qe, thr, Qrand=Qrand)
            r.update({"n_pass": len(passing), "n_bound": len(bound_lc), "bound_lc": bound_lc})
            out[thr][T] = r
    return out


def resolvability_verdict(agg, thresholds=(0.1, 0.2, 0.3)):
    """For each threshold and each adjacent T-pair, decide if the xi difference is RESOLVED:
    |xi_lo - xi_hi| > dxi_lo + dxi_hi, and BOTH points are kind='point'. Returns a list of verdict dicts."""
    verdicts = []
    for thr in thresholds:
        Ts = sorted(agg[thr].keys())
        for Ta, Tb in zip(Ts[:-1], Ts[1:]):
            ra, rb = agg[thr][Ta], agg[thr][Tb]
            if ra["kind"] != "point" or rb["kind"] != "point":
                verdicts.append({"thr": thr, "T_pair": (Ta, Tb), "resolved": False,
                                 "reason": f"{ra['kind']}/{rb['kind']}"})
                continue
            dxi = (ra["dxi"] or 0.0) + (rb["dxi"] or 0.0)
            gap = abs(ra["xi"] - rb["xi"])
            verdicts.append({"thr": thr, "T_pair": (Ta, Tb), "resolved": bool(gap > dxi),
                             "gap": float(gap), "dxi_sum": float(dxi)})
    return verdicts


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = "liquid_coupling_flow/artifacts/pts"; os.makedirs(out_dir, exist_ok=True)
    from liquid_coupling_flow.ka_pin import load_geometry_table
    from liquid_coupling_flow.ka_pin_refs import get_references
    ladder = {256: [0.24, 0.16, 0.12, 0.08], 576: [0.06, 0.04]}
    temps = [0.8, 0.65, 0.5]
    cells = []
    tables = {}
    for T in temps:
        for N, cs in ladder.items():
            if N == 576 and T == 0.5:
                continue                                             # stretch goal (see spec §6)
            refs = get_references(T, N, 16, dev, out_dir)
            if N not in tables:
                nB = int((refs["s"] == 1).sum())
                tables[N] = load_geometry_table(N, refs["L"], nB,
                                                "liquid_coupling_flow/ipl44/data/jf_ka100tt_best.pt", dev)
            table_fn = tables[N]
            for c in cs:
                cells.append(run_cell(T, c, refs, n_iter=3000, table_fn=table_fn, device=dev, out_dir=out_dir))
    agg = aggregate(cells)
    verdicts = resolvability_verdict(agg)
    torch.save({"cells": cells, "agg": agg, "resolvability": verdicts},
               os.path.join(out_dir, "pts_summary.pt"))
    for v in verdicts:
        tag = "RESOLVED" if v["resolved"] else "UNRESOLVED"
        detail = v.get("reason") or f"gap {v.get('gap', 0):.2f} vs dxi_sum {v.get('dxi_sum', 0):.2f}"
        print(f"[pts] resolvability thr={v['thr']} T{v['T_pair']}: {tag} ({detail})", flush=True)
    p = os.path.join(out_dir, "xi_of_T.png")
    for w in _plot_xi_of_T(agg, p):
        print(f"[pts] WARN {w}", flush=True)
    print(f"[pts] SAVED {os.path.abspath(p)}", flush=True)


if __name__ == "__main__":
    main()
