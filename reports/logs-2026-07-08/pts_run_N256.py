"""PTS campaign launch (scoped v1): N=256, T in {0.8, 0.65} (feasible interior convergence; T=0.5 deferred to
v2 per the probe: scramble-arm tau~13000 at T=0.5). Full c-ladder, G-conv gates each cell (converged->xi,
unconverged->bound). Per-cell incremental saves (run_cell) + campaign-level partial saves. Reuses the reviewed
ka_pin_campaign functions unchanged. Run from repo root: python -m ... or python reports/logs-2026-07-08/pts_run_N256.py"""
import os, sys, time, torch
sys.path.insert(0, os.getcwd())
from liquid_coupling_flow.ka_pin_refs import get_references
from liquid_coupling_flow.ka_pin import load_geometry_table
from liquid_coupling_flow.ka_pin_campaign import run_cell, aggregate, resolvability_verdict, _plot_xi_of_T

dev = "cuda"
OUT = "liquid_coupling_flow/artifacts/pts"; os.makedirs(OUT, exist_ok=True)
CKPT = "liquid_coupling_flow/ipl44/data/jf_ka100tt_best.pt"
TEMPS = [0.8, 0.65]; CS = [0.24, 0.16, 0.12, 0.08]; N = 256; NITER = 12000; NREAL = 12

print(f"[campaign] N={N} temps={TEMPS} c={CS} n_iter={NITER} n_real={NREAL}", flush=True)
cells = []; tables = {}; t0 = time.time()
for T in TEMPS:
    refs = get_references(T, N, 16, dev, OUT)
    if N not in tables:
        nB = int((refs["s"] == 1).sum())
        tables[N] = load_geometry_table(N, refs["L"], nB, CKPT, dev)
    print(f"[campaign] T={T} refs ready xB={refs['xB']:.3f} ({time.time()-t0:.0f}s)", flush=True)
    for c in CS:
        tc = time.time()
        cell = run_cell(T, c, refs, n_iter=NITER, table_fn=tables[N], device=dev, out_dir=OUT, n_real=NREAL)
        cells.append(cell)
        print(f"[campaign] T={T} c={c} lc={cell['lc']:.2f} passed={cell['passed']} "
              f"run_ok={cell['gconv']['run_ok']} gap={cell['gconv']['gap']:.3f} tau={cell['gconv']['tau']:.0f} "
              f"Qinf={cell['Qinf']:.3f}+/-{cell['Qinf_err']:.3f} | cell {time.time()-tc:.0f}s total {time.time()-t0:.0f}s",
              flush=True)
        torch.save({"cells": cells, "meta": {"temps": TEMPS, "cs": CS, "N": N, "n_iter": NITER, "n_real": NREAL}},
                   f"{OUT}/pts_campaign_N{N}_partial.pt")

agg = aggregate(cells); verd = resolvability_verdict(agg)
torch.save({"cells": cells, "agg": agg, "resolvability": verd}, f"{OUT}/pts_summary_N{N}.pt")
for thr in (0.1, 0.2, 0.3):
    for Tk in sorted(agg[thr].keys()):
        r = agg[thr][Tk]
        print(f"[xi] thr={thr} T={Tk} kind={r['kind']} xi={r['xi']} dxi={r.get('dxi')} "
              f"n_pass={r.get('n_pass')} n_bound={r.get('n_bound')} bound_lc={r.get('bound_lc')}", flush=True)
for v in verd:
    print(f"[resolv] {v}", flush=True)
p = f"{OUT}/xi_of_T_N{N}.png"; w = _plot_xi_of_T(agg, p)
print(f"[campaign] SAVED {os.path.abspath(p)} warns={w}", flush=True)
print(f"[campaign] DONE total {time.time()-t0:.0f}s", flush=True)
