"""How many basins does the Boltzmann cavity ensemble have, vs radius R? (Physics, model-free.) Relax
B diverse seeds (1 data + 11 uniform-random interiors) under frozen-boundary T=0.5 MC, then cluster by
PTS overlap. Pinned (R<xi~3.8): all seeds converge to the data basin (high overlap, 1 cluster) -> a
proposal need not cross basins. Unpinned (R>xi): seeds spread into multiple basins -> cross-basin
proposals become necessary. Confinement: radial clamp into |x|<R each sweep (diagnostic, not exact MC).
Overlap q(x,y) = mean over interior-x of [exists same-species interior-y within a=0.3]."""
import torch, statistics as st
from liquid_coupling_flow.ka_pmc_3d import parallel_mc_disp
from liquid_coupling_flow.ka_cavity_3d import local_identity_swap
from liquid_coupling_flow.ka3d_cavity_ar import _mic
from liquid_coupling_flow.ka3d_cavity_carve import carve

dev = "cuda"; RCTX = 2.5; BETA = 2.0; BIGL = 100.0; STEP = 0.05; N_SWEEP = 200; B = 12; A_OVL = 0.3; QTHR = 0.5
D = torch.load("liquid_coupling_flow/artifacts/ka3d_dataset_N4096_T0.5_rho1.15.pt", map_location=dev, weights_only=False)
X, S, L = D["x"].to(dev).float(), D["s"].to(dev).long(), float(D["L"])
gen = torch.Generator(device=dev).manual_seed(0)


def relax_seeds(interior_x, interior_s, bnd_x, bnd_s, R):
    """B mobile interiors (row 0 = data, 1..=uniform) + shared frozen boundary. Returns relaxed interiors."""
    ni, nb = interior_x.shape[1], bnd_x.shape[0]; off = BIGL / 2
    x = torch.cat([interior_x, bnd_x[None].expand(B, nb, 3)], 1).clone() + off
    s = torch.cat([interior_s, bnd_s[None].expand(B, nb)], 1).clone().long()
    mobile = torch.zeros(B, ni + nb, dtype=torch.bool, device=dev); mobile[:, :ni] = True
    ctr = torch.full((B, 3), off, device=dev)
    for sw in range(N_SWEEP):
        x = parallel_mc_disp(x, s, BIGL, BETA, STEP, mobile=mobile)
        if sw % 5 == 0:                                                  # swaps to let species rearrange
            s, _, _ = local_identity_swap(x, s, torch.zeros(B, device=dev), mobile, BETA, BIGL)
        rel = x[:, :ni] - ctr[:, None]; rr = rel.norm(dim=-1, keepdim=True)   # radial clamp: keep interior in |x|<R
        x[:, :ni] = ctr[:, None] + rel * (R * 0.999 / rr.clamp_min(1e-9)).clamp(max=1.0)
    return x[:, :ni] - off, s[:, :ni]


def overlap(xi, si, xj, sj):
    d = torch.cdist(xi, xj)                                              # [ni, nj]
    same = (si[:, None] == sj[None, :])
    d = d.masked_fill(~same, 9.0)
    return float((d.min(1).values < A_OVL).float().mean())


print("=== CAVITY BASIN STRUCTURE vs R (frozen-boundary T=0.5, B=12 seeds/cavity) ===", flush=True)
for R in (2.0, 3.0, 4.0):
    ov_data, largest_frac, ncomp_all = [], [], []
    for trial in range(3):                                              # 3 cavities per R
        while True:
            c = torch.rand(3, generator=gen, device=dev) * L; p = carve(X[trial], S[trial], c, R, L)
            if p["n_in"] >= 12:
                break
        xin = _mic(p["x_in"], c, L); sin = p["s_in"]; ni = xin.shape[0]
        xout = _mic(p["x_out"], c, L); bm = xout.norm(dim=-1) < (R + RCTX); bnd, sb = xout[bm], p["s_out"][bm]
        seeds = xin[None].expand(B, ni, 3).clone()
        seeds[1:] = (torch.rand(B - 1, ni, 3, generator=gen, device=dev) * 2 - 1)   # uniform in cube, clamp to ball
        rr = seeds[1:].norm(dim=-1, keepdim=True); seeds[1:] = seeds[1:] * (R * 0.9 / rr.clamp_min(1e-9)).clamp(max=1.0)
        sseed = sin[None].expand(B, ni).clone()
        rx, rs = relax_seeds(seeds, sseed, bnd, sb, R)
        # overlap of each relaxed seed (1..) with the relaxed data config (row 0)
        ov_data += [overlap(rx[k], rs[k], rx[0], rs[0]) for k in range(1, B)]
        # cluster all B by pairwise overlap >= QTHR (connected components)
        Q = torch.tensor([[overlap(rx[i], rs[i], rx[j], rs[j]) for j in range(B)] for i in range(B)])
        adj = Q >= QTHR
        seen = [False] * B; comps = []
        for i in range(B):
            if seen[i]:
                continue
            stack = [i]; comp = []
            while stack:
                u = stack.pop()
                if seen[u]:
                    continue
                seen[u] = True; comp.append(u)
                stack += [v for v in range(B) if adj[u, v] and not seen[v]]
            comps.append(len(comp))
        largest_frac.append(max(comps) / B); ncomp_all.append(len(comps))
    print(f"R={R}: overlap(uniform-relaxed, data) median {st.median(ov_data):.2f}  | "
          f"largest-basin frac {st.mean(largest_frac):.2f}  n_basins {st.mean(ncomp_all):.1f}  "
          f"-> {'PINNED (1 basin)' if st.mean(largest_frac) > 0.8 else 'MULTI-BASIN'}", flush=True)
torch.save({"note": "cavity basins"}, "reports/logs-2026-07-13/diag_cavity_basins.pt")
print("saved", flush=True)
