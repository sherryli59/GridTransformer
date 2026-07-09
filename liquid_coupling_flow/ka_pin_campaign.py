"""Point-to-set campaign: run the T x c ladder (random pinning), gate each cell, extract xi_pin(T).
All learned parts N=100-trained, zero-shot at N>=256, behind exact Metropolis. Saves per-cell trajectories
incrementally and the final xi(T) figure."""
import os, math, torch, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from liquid_coupling_flow.ka_pin import pin_mask, constrained_run
from liquid_coupling_flow.ka_pin_extract import stretched_exp_fit, xi_threshold
from liquid_coupling_flow.ka_pin_gates import g_conv
from liquid_coupling_flow.ka_pin_overlap import q_rand

RHO = 1.2


def run_cell(T, c, refs, n_iter, table_fn, device, out_dir, n_real=16, dt=0.01, record_every=5):
    os.makedirs(out_dir, exist_ok=True)
    x = refs["x"][:n_real].to(device); s = refs["s"].to(device)
    if s.dim() == 1:
        s = s[None].expand(n_real, -1).contiguous()
    N = x.shape[1]; L = refs["L"]
    gen = torch.Generator(device=device); gen.manual_seed(hash((round(T, 3), round(c, 3))) % (2**31))
    mobile = pin_mask(n_real, N, c, device, generator=gen)
    torch.manual_seed(12345); scr = torch.rand_like(x) * L
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
    cell = {"T": T, "c": c, "lc": lc, "Qinf": Qinf, "Qinf_err": Qinf_err,
            "gconv": gc, "xB": float((s[mobile] == 1).float().mean()),
            "arms": {"ref": ref_run, "scramble": scr_run}}
    torch.save(cell, os.path.join(out_dir, f"cell_T{T}_c{c}.pt"))       # incremental save
    print(f"[pts] T={T} c={c} lc={lc:.2f} Qinf={Qinf:.3f}+/-{Qinf_err:.3f} "
          f"gconv={'PASS' if gc['passed'] else 'BOUND'} -> {os.path.join(out_dir, f'cell_T{T}_c{c}.pt')}",
          flush=True)
    return cell


def aggregate(cells, thresholds=(0.1, 0.2, 0.3), Qrand=None):
    Qrand = q_rand() if Qrand is None else Qrand
    by_T = {}
    for cl in cells:
        by_T.setdefault(cl["T"], []).append(cl)
    out = {}
    for thr in thresholds:
        out[thr] = {}
        for T, cs in by_T.items():
            cs = sorted(cs, key=lambda z: z["lc"])
            lv = np.array([z["lc"] for z in cs]); qi = np.array([z["Qinf"] for z in cs])
            qe = np.array([z.get("Qinf_err", 0.01) for z in cs])
            out[thr][T] = xi_threshold(lv, qi, qe, thr, Qrand=Qrand)
    return out


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = "liquid_coupling_flow/artifacts/pts"; os.makedirs(out_dir, exist_ok=True)
    from liquid_coupling_flow.ka_pin import load_geometry_table
    from liquid_coupling_flow.ka_pin_refs import get_references
    ladder = {256: [0.24, 0.16, 0.12, 0.08], 576: [0.06, 0.04]}
    temps = [0.8, 0.65, 0.5]
    cells = []
    for T in temps:
        for N, cs in ladder.items():
            if N == 576 and T == 0.5:
                continue                                             # stretch goal (see spec §6)
            refs = get_references(T, N, 16, dev, out_dir)
            nB = int((refs["s"] == 1).sum()) if refs["s"].dim() == 1 else int(refs["s"][0].sum())
            table_fn = load_geometry_table(N, refs["L"], nB,
                                           "liquid_coupling_flow/ipl44/data/jf_ka100tt_best.pt", dev)
            for c in cs:
                cells.append(run_cell(T, c, refs, n_iter=3000, table_fn=table_fn, device=dev, out_dir=out_dir))
    agg = aggregate(cells)
    torch.save({"cells": cells, "agg": agg}, os.path.join(out_dir, "pts_summary.pt"))
    fig, ax = plt.subplots(figsize=(6, 4.4))
    for thr, mk in zip((0.1, 0.2, 0.3), ("o", "s", "^")):
        Ts = sorted(agg[thr].keys())
        xis = [agg[thr][T]["xi"] if isinstance(agg[thr][T]["xi"], float) else np.nan for T in Ts]
        dxis = [agg[thr][T]["dxi"] or 0 for T in Ts]
        ax.errorbar(Ts, xis, yerr=dxis, marker=mk, label=f"thr={thr}", capsize=3)
    ax.set_xlabel("T"); ax.set_ylabel(r"$\xi_{pin}$"); ax.legend(); ax.set_title("Point-to-set length vs T")
    p = os.path.join(out_dir, "xi_of_T.png"); fig.tight_layout(); fig.savefig(p, dpi=130)
    print(f"[pts] SAVED {os.path.abspath(p)}", flush=True)


if __name__ == "__main__":
    main()
