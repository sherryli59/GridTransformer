"""Proper cavity basin measurement with the ORACLE HEAT-BATH (teleport candidates cross single-particle
barriers, unlike local MC). On N=4096, R spanning xi~3.8. Per cavity: run M heat-bath chains from UNIFORM
seeds + 1 from DATA, equilibrate SWEEPS sweeps, then:
  (i) EQUILIBRATION CHECK: does uniform-init reach the same <U/particle> as data-init? (seed-independence
      -> the heat-bath actually mixed; if not, the basin count below is untrustworthy at that R.)
  (ii) BASIN COUNT: cluster the M equilibrated uniform configs + data by PTS overlap (q>=0.5 same basin).
       PINNED (R<xi): all reach the data basin -> largest-basin frac ~1, rearr~0.
       MULTI-BASIN (R>xi): configs spread -> largest-frac<1, rearr>0.
The heat-bath teleport (0.6 of candidates uniform-in-ball) is the basin-crossing move local MC lacked."""
import torch, statistics as st
from liquid_coupling_flow.ka3d_scaffold_ar import label_to_scaffold, fixed_ball_scaffold
from liquid_coupling_flow.ka3d_cavity_ar import _mic
from liquid_coupling_flow.ka3d_cavity_carve import carve
from liquid_coupling_flow.ka_energy import ka_energy, SIGMA, EPS, RCUT_FACTOR

dev = "cuda"; RCTX = 2.5; M = 12; BIGL = 100.0; BETA = 2.0
NCAND = 96; SWEEPS = 12; SIGLOC = 0.15; NCAV = 3; A_OVL = 0.3; QTHR = 0.5
T_SIG = torch.tensor(SIGMA, device=dev); T_EPS = torch.tensor(EPS, device=dev)
D = torch.load("liquid_coupling_flow/artifacts/ka3d_dataset_N4096_T0.5_rho1.15.pt", map_location=dev, weights_only=False)
X, S, L = D["x"].to(dev).float(), D["s"].to(dev).long(), float(D["L"])


def row_E(cand, s_i, ox, os_):
    sig = T_SIG[s_i[:, None], os_]; eps = T_EPS[s_i[:, None], os_]; rc2 = (RCUT_FACTOR * sig) ** 2
    r2 = (cand[:, :, None, :] - ox[:, None, :, :]).square().sum(-1)
    inv6 = (sig[:, None] ** 2 / r2) ** 3
    e = 4 * eps[:, None] * (inv6 ** 2 - inv6)
    src6 = (1.0 / RCUT_FACTOR) ** 6
    return torch.where(r2 < rc2[:, None], e - 4 * eps[:, None] * (src6 ** 2 - src6), torch.zeros_like(e)).sum(-1)


def full_E(x_int, s_int, bnd, sb):
    Mn = x_int.shape[0]
    allx = torch.cat([x_int, bnd[None].expand(Mn, bnd.shape[0], 3)], 1)
    alls = torch.cat([s_int, sb[None].expand(Mn, sb.shape[0])], 1)
    return ka_energy(allx, alls.long(), BIGL)


def heatbath(x_int, s_int, bnd, sb, Rr, gen, sweeps):
    Mn, Nn = x_int.shape[0], x_int.shape[1]
    for sw in range(sweeps):
        for i in torch.randperm(Nn, generator=gen, device=dev).tolist():
            oth = [j for j in range(Nn) if j != i]
            ox = torch.cat([x_int[:, oth], bnd[None].expand(Mn, bnd.shape[0], 3)], 1)
            os_ = torch.cat([s_int[:, oth], sb[None].expand(Mn, sb.shape[0])], 1)
            n_tel = int(0.6 * NCAND)
            u = torch.randn(Mn, n_tel, 3, generator=gen, device=dev)
            u = u / u.norm(dim=-1, keepdim=True) * (torch.rand(Mn, n_tel, 1, generator=gen, device=dev) ** (1 / 3)) * Rr
            loc = x_int[:, i:i + 1] + SIGLOC * torch.randn(Mn, NCAND - n_tel, 3, generator=gen, device=dev)
            cand = torch.cat([u, loc, x_int[:, i:i + 1]], 1)
            logits = -BETA * row_E(cand, s_int[:, i], ox, os_)
            pick = torch.multinomial(torch.softmax(logits, 1), 1, generator=gen).squeeze(1)
            x_int = x_int.clone(); x_int[:, i] = cand[torch.arange(Mn, device=dev), pick]
    return x_int


def overlap(xi, si, xj, sj):
    d = torch.cdist(xi, xj).masked_fill(si[:, None] != sj[None, :], 9.0)
    return float((d.min(1).values < A_OVL).float().mean())


print(f"=== ORACLE HEAT-BATH BASIN MEASUREMENT (N=4096, M={M} chains, {NCAND} cands, {SWEEPS} sweeps) ===", flush=True)
gen = torch.Generator(device=dev).manual_seed(0)
for Rr in (2.0, 3.0, 4.0):
    dE_uni, dE_dat, rearr, largest, ncomp = [], [], [], [], []
    ncav = 0
    for ci in range(16):
        c = torch.rand(3, generator=gen, device=dev) * Rr * 0 + torch.rand(3, generator=gen, device=dev) * L
        p = carve(X[ci], S[ci], c, Rr, L)
        if p["n_in"] < 12:
            continue
        xin = _mic(p["x_in"], c, L); sin = p["s_in"].long(); n = xin.shape[0]
        xout = _mic(p["x_out"], c, L); bm = xout.norm(dim=-1) < (Rr + RCTX); bnd, sb = xout[bm], p["s_out"][bm].long()
        E_data = full_E(xin[None], sin[None], bnd, sb).item()
        # uniform seeds (M) + data seed (1)
        u0 = torch.randn(M, n, 3, generator=gen, device=dev)
        xu = u0 / u0.norm(dim=-1, keepdim=True) * (torch.rand(M, n, 1, generator=gen, device=dev) ** (1 / 3)) * Rr
        su = sin[None].expand(M, n).contiguous()
        xu_eq = heatbath(xu.contiguous(), su.clone(), bnd, sb, Rr, gen, SWEEPS)
        xd_eq = heatbath(xin[None].clone(), sin[None].clone(), bnd, sb, Rr, gen, SWEEPS)
        dE_uni.append((full_E(xu_eq, su, bnd, sb).mean().item() - E_data) / n)
        dE_dat.append((full_E(xd_eq, sin[None], bnd, sb).item() - E_data) / n)
        rearr += [float((torch.cdist(xu_eq[k], xin).min(1).values > 0.6).float().mean()) for k in range(M)]
        # cluster M uniform-eq + data-eq by overlap
        cfgs = [(xu_eq[k], su[k]) for k in range(M)] + [(xd_eq[0], sin)]
        B = len(cfgs); Q = [[overlap(cfgs[i][0], cfgs[i][1], cfgs[j][0], cfgs[j][1]) for j in range(B)] for i in range(B)]
        adj = [[Q[i][j] >= QTHR for j in range(B)] for i in range(B)]; seen = [False] * B; comps = []
        for i in range(B):
            if seen[i]:
                continue
            stack = [i]; sz = 0
            while stack:
                v = stack.pop()
                if seen[v]:
                    continue
                seen[v] = True; sz += 1; stack += [w for w in range(B) if adj[v][w] and not seen[w]]
            comps.append(sz)
        largest.append(max(comps) / B); ncomp.append(len(comps))
        ncav += 1
        if ncav >= NCAV:
            break
    eq = "EQUILIBRATED" if abs(st.mean(dE_uni) - st.mean(dE_dat)) < 0.3 else "NOT-equilibrated (basin count untrustworthy)"
    verdict = "PINNED" if st.mean(largest) > 0.8 else "MULTI-BASIN"
    print(f"R={Rr}: dU/part uni {st.mean(dE_uni):+.2f} vs data {st.mean(dE_dat):+.2f} [{eq}] | "
          f"rearr {st.mean(rearr):.2f}  largest-basin {st.mean(largest):.2f}  n_basins {st.mean(ncomp):.1f} -> {verdict}", flush=True)
torch.save({"note": "oracle basins"}, "reports/logs-2026-07-13/diag_oracle_basins.pt")
print("saved", flush=True)
