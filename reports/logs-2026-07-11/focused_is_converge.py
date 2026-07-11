"""Converged IMPORTANCE-SAMPLING PTS estimate with the K=n full-regen proposal (exact logq).
Per cavity: draw M independent full-interior regens C_k ~ q(.|boundary), weight w_k = e^{-bU(C_k)-logq(C_k)}
(-> Boltzmann), estimate q_IS = sum_k w_k overlap(C_k,C0). Tracks q_IS(M) and ESS%(M) as M grows to test
(a) does the estimate CONVERGE, (b) is ESS% STABLE (healthy) or collapsing (heavy-tailed, single-sample
dominance). Averages over cavities for a PTS overlap vs R. IS is embarrassingly parallel + unbiased, so
low-but-stable ESS just costs more M (per user)."""
import argparse, math, statistics as st, time
from pathlib import Path
import torch
from liquid_coupling_flow.ka3d_scaffold_ebm import KA3DScaffoldEBM
from liquid_coupling_flow.ka3d_scaffold_ar import label_to_scaffold
from liquid_coupling_flow.ka3d_cavity_ar import _mic
from liquid_coupling_flow.ka3d_cavity_carve import carve
from liquid_coupling_flow.ka_energy import ka_energy

ART = "liquid_coupling_flow/artifacts"; dev = "cuda"


def energy(x, s, bnd, s_bnd):
    xx = torch.cat([x, bnd], 0); ss = torch.cat([s, s_bnd], 0)
    return float(ka_energy(xx[None], ss.long()[None], 100.0)[0])


def overlap(C, C0, a=0.3):
    return float((torch.cdist(C0, C).min(1).values < a).float().mean())


def cum(logw, overs, Ms):
    """cumulative self-normalised q_IS and ESS% using the first M samples, for M in Ms."""
    out = []
    for M in Ms:
        lw = logw[:M]; w = torch.softmax(lw, 0)
        ess = float(1.0 / (w ** 2).sum() / M)
        q = float((w * overs[:M]).sum())
        out.append((q, 100 * ess))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--radii", type=float, nargs="+", default=[1.6, 2.0, 2.4])
    ap.add_argument("--ncav", type=int, default=8); ap.add_argument("--M", type=int, default=768)
    ap.add_argument("--beta", type=float, default=2.0); ap.add_argument("--r-ctx", type=float, default=2.5)
    ap.add_argument("--ckpt", default=f"{ART}/ka3d_cavity_ebm3ax.pt")
    ap.add_argument("--out", default="reports/logs-2026-07-11/pts_is.pt")
    a = ap.parse_args()
    ck = torch.load(a.ckpt, map_location=dev, weights_only=False)
    m = KA3DScaffoldEBM(cat_bins=128, cat_range=2.5).to(dev)
    m.load_state_dict(ck["state_dict"], strict=False); m.eval(); m.use_frame = False
    d = torch.load(f"{ART}/ka3d_dataset_N512_T0.5.pt", map_location=dev, weights_only=False)
    X, S, L = d["x"].to(dev).float(), d["s"].to(dev).long(), float(d["L"])
    gen = torch.Generator(device=dev).manual_seed(0)
    Ms = [x for x in (48, 96, 192, 384, 768, 1536) if x <= a.M]
    results = {}
    for R in a.radii:
        percav, ncav = [], 0; t0 = time.time()
        for ci in range(900, 900 + 6 * a.ncav):
            if ncav >= a.ncav:
                break
            c = torch.rand(3, generator=gen, device=dev) * L; p = carve(X[ci], S[ci], c, R, L)
            if p["n_in"] < 10:
                continue
            xo, so, _ = label_to_scaffold(_mic(p["x_in"], c, L), p["s_in"], R)
            xout = _mic(p["x_out"], c, L); bm = xout.norm(dim=-1) < (R + a.r_ctx)
            bnd, s_bnd, n = xout[bm], p["s_out"][bm], xo.shape[0]
            C0 = xo.clone(); allmask = torch.ones(n, dtype=torch.bool, device=dev)
            logw = torch.empty(a.M, device=dev); overs = torch.empty(a.M, device=dev)
            for k in range(a.M):
                fx, fs, flq = m.sample_block(xo, so, allmask, bnd, s_bnd, R, gen=gen)
                logw[k] = -a.beta * energy(fx, fs, bnd, s_bnd) - float(flq); overs[k] = overlap(fx, C0)
            percav.append(cum(logw, overs, Ms)); ncav += 1
        # average across cavities at each M
        qM = [st.mean([pc[i][0] for pc in percav]) for i in range(len(Ms))]
        eM = [st.mean([pc[i][1] for pc in percav]) for i in range(len(Ms))]
        results[R] = {"Ms": Ms, "q": qM, "ess": eM, "ncav": ncav}
        print(f"R={R} (ncav={ncav}, {time.time()-t0:.0f}s):", flush=True)
        for i, M in enumerate(Ms):
            print(f"    M={M:5d}: q_IS={qM[i]:.3f}  ESS={eM[i]:.1f}%  (eff~{eM[i]/100*M:.0f} samples)", flush=True)
        Path(a.out).parent.mkdir(parents=True, exist_ok=True); torch.save(results, a.out)
    print("\n=== converged IS PTS overlap vs R ===", flush=True)
    for R in a.radii:
        r = results[R]
        print(f"  R={R}: q_IS={r['q'][-1]:.3f}  (ESS {r['ess'][-1]:.1f}% @ M={r['Ms'][-1]}, stable={'~yes' if abs(r['ess'][-1]-r['ess'][0])<2 else 'NO'})", flush=True)
    print(f"saved {a.out}", flush=True)


if __name__ == "__main__":
    main()
