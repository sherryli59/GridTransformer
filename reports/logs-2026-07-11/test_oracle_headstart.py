"""GATE part 3 (final): AR-seed HEAD-START vs uniform at R=2.5 (the max radius the N=512 box allows:
2R+r_cut<=L caps R at 2.51 — this bounds the WHOLE PTS campaign on this dataset, not just this test).

Per-sweep median dE/particle trajectories of the whole-interior oracle heat-bath from AR seeds vs UNIFORM
seeds vs DATA control, 12 sweeps. The decision metric is SWEEPS-SAVED by AR seeding (the metric that
justified the glass pivot): if AR saves ~0 sweeps of a ~74-site heat-bath, the learned generator adds no
value to cavity sampling at any measurable radius, and option B has no payoff on this dataset."""
import torch, statistics as st
from liquid_coupling_flow.ka3d_ebm_batched import KA3DScaffoldEBMBatched
from liquid_coupling_flow.ka3d_scaffold_ar import label_to_scaffold
from liquid_coupling_flow.ka3d_cavity_ar import _mic
from liquid_coupling_flow.ka3d_cavity_carve import carve
from liquid_coupling_flow.ka_energy import ka_energy, SIGMA, EPS, RCUT_FACTOR

dev = "cuda"; RCTX = 2.5; M = 8; BIGL = 100.0; BETA = 2.0
NCAND = 192; SWEEPS = 12; SIGLOC = 0.15; NCAV = 5; Rr = 2.5
T_SIG = torch.tensor(SIGMA, device=dev); T_EPS = torch.tensor(EPS, device=dev)


def load_base():
    ck = torch.load("liquid_coupling_flow/artifacts/ka3d_cavity_ebm3ax.pt", map_location=dev, weights_only=False)
    m = KA3DScaffoldEBMBatched(cat_bins=128, cat_range=2.5).to(dev)
    m.load_state_dict(ck["state_dict"], strict=False); m.eval(); m.use_frame = False
    return m


def row_E(cand, s_i, ox, os_):
    sig = T_SIG[s_i[:, None], os_]; eps = T_EPS[s_i[:, None], os_]
    rc2 = (RCUT_FACTOR * sig) ** 2
    r2 = (cand[:, :, None, :] - ox[:, None, :, :]).square().sum(-1)
    inv6 = (sig[:, None] ** 2 / r2) ** 3
    e = 4 * eps[:, None] * (inv6 ** 2 - inv6)
    src6 = (1.0 / RCUT_FACTOR) ** 6
    e = torch.where(r2 < rc2[:, None], e - 4 * eps[:, None] * (src6 ** 2 - src6), torch.zeros_like(e))
    return e.sum(-1)


def full_E(x_int, s_int, bnd, sb):
    Mn = x_int.shape[0]
    allx = torch.cat([x_int, bnd[None].expand(Mn, bnd.shape[0], 3)], 1)
    alls = torch.cat([s_int, sb[None].expand(Mn, sb.shape[0])], 1)
    return ka_energy(allx, alls.long(), BIGL)


def heatbath_traj(x_int, s_int, bnd, sb, E_data, n, gen, sweeps=SWEEPS):
    Mn, Nn = x_int.shape[0], x_int.shape[1]; dEs = []
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
        dEs.append((full_E(x_int, s_int, bnd, sb).mean().item() - E_data) / n)
    return dEs, x_int


m = load_base()
d = torch.load("liquid_coupling_flow/artifacts/ka3d_dataset_N512_T0.5.pt", map_location=dev, weights_only=False)
X, S, L = d["x"].to(dev).float(), d["s"].to(dev).long(), float(d["L"]); gen = torch.Generator(device=dev).manual_seed(0)
tr = {"ar": [], "un": [], "ct": []}; e0 = {"ar": [], "un": []}; ncav = 0
for ci in range(900, 990):
    c = torch.rand(3, generator=gen, device=dev) * L; p = carve(X[ci], S[ci], c, Rr, L)
    if p["n_in"] < 10:
        continue
    xo, so, _ = label_to_scaffold(_mic(p["x_in"], c, L), p["s_in"], Rr)
    xout = _mic(p["x_out"], c, L); bm = xout.norm(dim=-1) < (Rr + RCTX); bnd, sb = xout[bm], p["s_out"][bm]
    n = xo.shape[0]
    E_data = full_E(xo[None], so[None], bnd, sb).item()
    blk_all = torch.ones(n, dtype=torch.bool, device=dev)
    xa, sa, _ = m.sample_block_b(xo[None].expand(M, n, 3), so[None].expand(M, n), blk_all, bnd, sb, Rr, gen=gen)
    u0 = torch.randn(M, n, 3, generator=gen, device=dev)
    xu = u0 / u0.norm(dim=-1, keepdim=True) * (torch.rand(M, n, 1, generator=gen, device=dev) ** (1 / 3)) * Rr
    e0["ar"].append((full_E(xa, sa, bnd, sb).mean().item() - E_data) / n)
    e0["un"].append((full_E(xu, so[None].expand(M, n), bnd, sb).mean().item() - E_data) / n)
    for key, x0, s0 in (("ar", xa.contiguous(), sa), ("un", xu, so[None].expand(M, n).contiguous()),
                        ("ct", xo[None].expand(M, n, 3).contiguous(), so[None].expand(M, n).contiguous())):
        dEs, _ = heatbath_traj(x0, s0, bnd, sb, E_data, n, gen)
        tr[key].append(dEs)
    ncav += 1
    print(f"  cav {ncav} (n={n}) done", flush=True)
    if ncav >= NCAV:
        break

med = lambda k, sw: st.median([t[sw] for t in tr[k]])
print(f"\nR={Rr} whole-interior oracle heat-bath, {ncav} cavities, {SWEEPS} sweeps  (median dE/particle)", flush=True)
print(f"  seed energies: AR {st.median(e0['ar']):+9.1f}   UNIF {st.median(e0['un']):+9.1f}", flush=True)
print("  sweep:" + "".join(f"{sw+1:>8d}" for sw in range(SWEEPS)), flush=True)
for key, name in (("ar", "AR   "), ("un", "UNIF "), ("ct", "DATA ")):
    print(f"  {name}:" + "".join(f"{med(key, sw):+8.2f}" for sw in range(SWEEPS)), flush=True)
saved = None
for thr in (3.0, 2.0, 1.5):
    t_ar = next((sw + 1 for sw in range(SWEEPS) if med("ar", sw) < thr), SWEEPS + 1)
    t_un = next((sw + 1 for sw in range(SWEEPS) if med("un", sw) < thr), SWEEPS + 1)
    print(f"  sweeps-to dE<{thr}: AR {t_ar}  UNIF {t_un}  -> AR saves {t_un - t_ar}", flush=True)
    if thr == 2.0:
        saved = t_un - t_ar
print("READOUT:", ("AR head-start = " + str(saved) + " sweeps (of a ~74-site oracle heat-bath) at the max "
      "measurable radius").format(), flush=True)
