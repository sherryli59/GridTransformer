"""Joint (x,u) block-pair datasets from the FIXED banks (poly_bank_fixed_run{r}.pt, which pair
x[j] with sigs[j] per frame -- see poly_rebank.py for why the original bank is unusable).

Continuous transport: no permutation; make_joint_block_pair runs short block-local joint MC
(displacement + single-site u-moves), so both channels drift continuously (amendment spec
2026-07-17-joint-sigma-x-continuous-flow-amendment.md).

Usage:
  poly_joint_pairs.py --T 0.085 --k 8 --npairs 3000 [--runs 1,2] [--relax_sw 400] [--delta 0.5]
                      [--calibrate]
--calibrate: instead of writing a dataset, sweep (relax_sw x delta) on 150 pairs each and print
transport stats (mean|du|, p95|du|, rms|dx|, p95 max|dx|, u/x acceptance) to choose settings.
Differences are min-imaged (a naive x_new-x_old on centered coords shows ~L wrap artifacts).
Output: poly_jointpairs_T{T}_k{k}.pt (list of make_joint_block_pair dicts + a 'meta' entry).
"""
import argparse, sys, time
import numpy as np
import torch
sys.path.insert(0, "/mnt/ssd/GridTransformer")
from liquid_coupling_flow.poly.semigrand import make_joint_block_pair, u_of_sigma  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--T", type=float, required=True)
p.add_argument("--k", type=int, required=True)
p.add_argument("--npairs", type=int, default=3000)
p.add_argument("--runs", type=str, default="1,2", help="training runs (run 3 held out for gate)")
p.add_argument("--relax_sw", type=int, default=400)
p.add_argument("--delta", type=float, default=0.5)
p.add_argument("--calibrate", action="store_true")
p.add_argument("--out", type=str, default=None)
a = p.parse_args()
runs = [int(r) for r in a.runs.split(",")]

banks = {}
for r in runs:
    D = torch.load(f"reports/logs-2026-07-17/poly_bank_fixed_run{r}.pt", weights_only=False)
    key = (r, a.T)
    assert key in D, f"fixed bank run{r} has no T={a.T} yet (rebank still descending?)"
    rec = D[key]
    assert "sigs" in rec, "fixed bank must carry per-frame sigs"
    banks[r] = rec


def build(m, relax_sw, delta, seed0=9000):
    r = runs[m % len(runs)]
    rec = banks[r]
    j = m % rec["x"].shape[0]
    fr = rec["x"][j].astype(np.float64).copy()
    sig = rec["sigs"][j].astype(np.float64).copy()          # per-frame pairing (THE fix)
    u = u_of_sigma(sig)
    return make_joint_block_pair(fr, sig, u, float(rec["L"]), 1.0 / a.T, a.k,
                                 seed=seed0 + m, relax_sw=relax_sw, delta=delta)


def stats(pairs, L):
    du = np.concatenate([np.abs(q["u_new"] - q["u_old"]) for q in pairs])
    dmax, rms = [], []
    for q in pairs:
        d = q["x_new"] - q["x_old"]
        d -= L * np.round(d / L)                            # min-image the DIFFERENCE
        r = np.sqrt((d ** 2).sum(1))
        rms.append((r ** 2).mean()); dmax.append(r.max())
    return (du.mean(), np.percentile(du, 95),
            float(np.sqrt(np.mean(rms))), float(np.percentile(dmax, 95)))


if a.calibrate:
    L = float(banks[runs[0]]["L"])
    print(f"calibration at T={a.T} k={a.k} (150 pairs/cell, fixed banks, min-imaged diffs)")
    print(f"{'relax_sw':>8} {'delta':>6} | {'mean|du|':>8} {'p95|du|':>8} | {'rms|dx|':>8} {'p95max|dx|':>10}")
    for relax_sw in (50, 100, 200, 400):
        for delta in (0.15, 0.3, 0.5):
            ps = [build(m, relax_sw, delta, seed0=70000 + 1000 * relax_sw) for m in range(150)]
            mu, p95u, rx, p95x = stats(ps, L)
            print(f"{relax_sw:>8} {delta:>6} | {mu:8.3f} {p95u:8.3f} | {rx:8.3f} {p95x:10.3f}", flush=True)
    sys.exit(0)

out = a.out or f"reports/logs-2026-07-17/poly_jointpairs_T{a.T}_k{a.k}.pt"
pairs = []
t0 = time.time()
for m in range(a.npairs):
    pairs.append(build(m, a.relax_sw, a.delta))
    if (m + 1) % 500 == 0:
        torch.save({"pairs": pairs, "meta": vars(a)}, out)
        print(f"{m+1}/{a.npairs} ({time.time()-t0:.0f}s)", flush=True)
torch.save({"pairs": pairs, "meta": vars(a)}, out)
mu, p95u, rx, p95x = stats(pairs, float(banks[runs[0]]["L"]))
print(f"PAIRS DONE -> {out} ({len(pairs)}; mean|du| {mu:.3f}, rms|dx| {rx:.3f}, "
      f"p95max|dx| {p95x:.3f}; {time.time()-t0:.0f}s)", flush=True)
