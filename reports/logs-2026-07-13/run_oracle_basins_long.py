"""PROPER basin R-scan: oracle heat-bath to CONVERGENCE (150 sweeps + swaps), equilibration-guarded, on
N=4096. The 12-sweep version was under-equilibrated (uni +1.38 vs data 0; convergence check showed uni
reaches data only by ~100-150 sweeps). Here: per R in {2.0,3.0,4.0}, M=12 uniform seeds + 1 data seed,
150 sweeps, record energy every 25 sweeps (convergence proof), equilibration guard (uni vs data final),
then cluster by overlap for a TRUSTWORTHY basin count. Incremental save + print per R (durability)."""
import torch, statistics as st, time
from liquid_coupling_flow.ka3d_cavity_ar import _mic
from liquid_coupling_flow.ka3d_cavity_carve import carve
from liquid_coupling_flow.ka_cavity_3d import local_identity_swap
from liquid_coupling_flow.ka_energy import ka_energy, SIGMA, EPS, RCUT_FACTOR

dev = "cuda"; RCTX = 2.5; M = 12; BIGL = 100.0; BETA = 2.0
NCAND = 96; SWEEPS = 150; REC = 25; SIGLOC = 0.15; NCAV = 3; A_OVL = 0.3; QTHR = 0.5
T_SIG = torch.tensor(SIGMA, device=dev); T_EPS = torch.tensor(EPS, device=dev)
D = torch.load("liquid_coupling_flow/artifacts/ka3d_dataset_N4096_T0.5_rho1.15.pt", map_location=dev, weights_only=False)
X, S, L = D["x"].to(dev).float(), D["s"].to(dev).long(), float(D["L"])
OUT = "reports/logs-2026-07-13/run_oracle_basins_long.pt"; results = {}


def row_E(cand, s_i, ox, os_):
    sig = T_SIG[s_i[:, None], os_]; eps = T_EPS[s_i[:, None], os_]; rc2 = (RCUT_FACTOR * sig) ** 2
    r2 = (cand[:, :, None, :] - ox[:, None, :, :]).square().sum(-1)
    inv6 = (sig[:, None] ** 2 / r2) ** 3
    e = 4 * eps[:, None] * (inv6 ** 2 - inv6); src6 = (1.0 / RCUT_FACTOR) ** 6
    return torch.where(r2 < rc2[:, None], e - 4 * eps[:, None] * (src6 ** 2 - src6), torch.zeros_like(e)).sum(-1)


def full_E(x, s, bnd, sb):
    Mn = x.shape[0]
    return ka_energy(torch.cat([x, bnd[None].expand(Mn, bnd.shape[0], 3)], 1),
                     torch.cat([s, sb[None].expand(Mn, sb.shape[0])], 1).long(), BIGL)


def heatbath_swaps(x, s, bnd, sb, Rr, gen, sweeps, Ed):
    Mn, Nn = x.shape[0], x.shape[1]; mob = torch.ones(Mn, Nn, dtype=torch.bool, device=dev); traj = {}
    for sw in range(1, sweeps + 1):
        for i in torch.randperm(Nn, generator=gen, device=dev).tolist():
            oth = [j for j in range(Nn) if j != i]
            ox = torch.cat([x[:, oth], bnd[None].expand(Mn, bnd.shape[0], 3)], 1)
            os_ = torch.cat([s[:, oth], sb[None].expand(Mn, sb.shape[0])], 1)
            nt = int(0.6 * NCAND)
            u = torch.randn(Mn, nt, 3, generator=gen, device=dev)
            u = u / u.norm(dim=-1, keepdim=True) * (torch.rand(Mn, nt, 1, generator=gen, device=dev) ** (1 / 3)) * Rr
            loc = x[:, i:i + 1] + SIGLOC * torch.randn(Mn, NCAND - nt, 3, generator=gen, device=dev)
            cand = torch.cat([u, loc, x[:, i:i + 1]], 1)
            pick = torch.multinomial(torch.softmax(-BETA * row_E(cand, s[:, i], ox, os_), 1), 1, generator=gen).squeeze(1)
            x = x.clone(); x[:, i] = cand[torch.arange(Mn, device=dev), pick]
        for _ in range(max(1, Nn // 8)):                                    # a few identity swaps per sweep
            s, _, _ = local_identity_swap(x, s, torch.zeros(Mn, device=dev), mob, BETA, BIGL)
        if sw % REC == 0:
            traj[sw] = (full_E(x, s, bnd, sb).mean().item() - Ed) / Nn
    return x, s, traj


def overlap(xi, si, xj, sj):
    d = torch.cdist(xi, xj).masked_fill(si[:, None] != sj[None, :], 9.0)
    return float((d.min(1).values < A_OVL).float().mean())


gen = torch.Generator(device=dev).manual_seed(0)
for Rr in (2.0, 3.0, 4.0):
    t0 = time.time(); dEu, dEd, rearr, largest, ncomp, trajs = [], [], [], [], [], []
    ncav = 0
    for ci in range(16):
        c = torch.rand(3, generator=gen, device=dev) * L; p = carve(X[ci], S[ci], c, Rr, L)
        if p["n_in"] < 12:
            continue
        xin = _mic(p["x_in"], c, L); sin = p["s_in"].long(); n = xin.shape[0]
        xout = _mic(p["x_out"], c, L); bm = xout.norm(dim=-1) < (Rr + RCTX); bnd, sb = xout[bm], p["s_out"][bm].long()
        Ed = full_E(xin[None], sin[None], bnd, sb).item()
        u0 = torch.randn(M, n, 3, generator=gen, device=dev)
        xu = u0 / u0.norm(dim=-1, keepdim=True) * (torch.rand(M, n, 1, generator=gen, device=dev) ** (1 / 3)) * Rr
        su = sin[None].expand(M, n).contiguous().clone()
        xu_eq, su_eq, tr = heatbath_swaps(xu.contiguous(), su, bnd, sb, Rr, gen, SWEEPS, Ed)
        xd_eq, sd_eq, _ = heatbath_swaps(xin[None].clone(), sin[None].clone(), bnd, sb, Rr, gen, SWEEPS, Ed)
        dEu.append((full_E(xu_eq, su_eq, bnd, sb).mean().item() - Ed) / n)
        dEd.append((full_E(xd_eq, sd_eq, bnd, sb).item() - Ed) / n); trajs.append(tr)
        rearr += [float((torch.cdist(xu_eq[k], xin).min(1).values > 0.6).float().mean()) for k in range(M)]
        cfgs = [(xu_eq[k], su_eq[k]) for k in range(M)] + [(xd_eq[0], sd_eq[0])]
        Bn = len(cfgs); adj = [[overlap(*cfgs[i], *cfgs[j]) >= QTHR for j in range(Bn)] for i in range(Bn)]
        seen = [False] * Bn; comps = []
        for i in range(Bn):
            if seen[i]:
                continue
            stk = [i]; sz = 0
            while stk:
                v = stk.pop()
                if seen[v]:
                    continue
                seen[v] = True; sz += 1; stk += [w for w in range(Bn) if adj[v][w] and not seen[w]]
            comps.append(sz)
        largest.append(max(comps) / Bn); ncomp.append(len(comps))
        ncav += 1
        if ncav >= NCAV:
            break
    eqok = abs(st.mean(dEu) - st.mean(dEd)) < 0.3
    tr_med = {k: st.median([t[k] for t in trajs if k in t]) for k in sorted(trajs[0])}
    results[Rr] = {"dEu": dEu, "dEd": dEd, "rearr": rearr, "largest": largest, "ncomp": ncomp,
                   "traj_med": tr_med, "eqok": eqok}
    torch.save(results, OUT)                                                # incremental durability
    curve = " ".join(f"{k}:{v:+.2f}" for k, v in tr_med.items())
    print(f"R={Rr}: conv[{curve}] final uni {st.mean(dEu):+.2f} vs data {st.mean(dEd):+.2f} "
          f"[{'EQUILIBRATED' if eqok else 'NOT-EQ'}] | rearr {st.mean(rearr):.2f} largest-basin {st.mean(largest):.2f} "
          f"n_basins {st.mean(ncomp):.1f} -> {'PINNED' if st.mean(largest) > 0.8 else 'MULTI/unresolved'}  ({time.time()-t0:.0f}s)", flush=True)
print(f"saved -> {OUT}", flush=True)
