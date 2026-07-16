"""G0 — BASIN-DISCOVERY CONTROL (user challenge 2026-07-16: "random start probably also genuinely finds
basins"). The p~1e-3 basin-mass measurement (diag_basin_mass 07-14) quenched AR draws ONLY — no uniform
control ever ran, and the recorded p spans 1e-3 (strict attraction basin) to ~1 (loose metabasin, FL
clustering) depending on criterion. This measures, under ONE identical pipeline:

  pools:  AR (KA3DScaffoldEBMBatched, exact-logq draws)  |  FLOW (cavity eRSI, RK4-30)  |  UNIFORM (in-sphere)
  per cavity: quench (a) top-K prescreened-by-ref-overlap and (b) RANDOM-K unscreened, from each pool
  membership vs the QUENCHED reference (never thermal-vs-quenched):
     STRICT (attraction basin): dU_IS/n < 0.15  AND  IS-box-overlap > 0.4
     META   (FL metabasin)    : dU_IS/n < 0.50  AND  IS-box-overlap > 0.3
  quench = displacement-capped steepest descent, f64 (Adam DIVERGES on r^-12 — recorded lesson).

Decision (scoping doc G0): p_q0 >= ~5x p_uniform at a useful granularity => learned discovery earns its keep;
p_q0 ~ p_uniform => discovery is classical, T2 needs no model, premise shrinks to T4-only.
Full distributions saved (record-simulation-data directive). Usage: g0_discovery_control.py [R] [NCAV]"""
import sys, math, time, statistics as st
from pathlib import Path
import torch

REPO = Path("/mnt/ssd/GridTransformer")
sys.path.insert(0, str(REPO))
from liquid_coupling_flow.ka3d_cavity_carve import carve
from liquid_coupling_flow.ka3d_cavity_ar import _mic
from liquid_coupling_flow.ka3d_scaffold_ar import label_to_scaffold
from liquid_coupling_flow.ka3d_ebm_batched import KA3DScaffoldEBMBatched
from liquid_coupling_flow.ka3d_cavity_egnn import CavityBlockFlow
from liquid_coupling_flow.ka_energy import ka_energy
torch.set_grad_enabled(False)

dev = "cuda"; RCTX = 2.5; BIGL = 100.0
R = float(sys.argv[1]) if len(sys.argv) > 1 else 2.0
NCAV = int(sys.argv[2]) if len(sys.argv) > 2 else 3
M_DRAW = 2048          # draws per pool (cheap)
K_PRE = 48             # quenched from prescreen top-K
K_RND = 64             # quenched from unscreened random-K  -> raw p detectable down to ~1.5%
QSTEPS = 700; QBATCH = 224
STRICT_DU, STRICT_OV = 0.15, 0.40
META_DU, META_OV = 0.50, 0.30
RHO = 1.2; L_H = (0.06 / RHO) ** (1.0 / 3.0)
DUMMY0, DUMMY_D = 50.0, 5.0
OUT = REPO / f"reports/logs-2026-07-16/g0_discovery_control_R{R}.pt"

D = torch.load(REPO / "liquid_coupling_flow/artifacts/ka3d_dataset_N512_T0.5.pt", map_location="cpu", weights_only=False)
Xds, Sds, L = D["x"].float(), D["s"].long(), float(D["L"])
n_train = int(Xds.shape[0] * 0.9)
ar = KA3DScaffoldEBMBatched(cat_bins=128, cat_range=2.5, knn_pot=24, use_demand=True).to(dev)
ar.load_state_dict(torch.load(REPO / "liquid_coupling_flow/artifacts/ka3d_cavity_ebm3ax_rho115_rl_Rext_knn24_lam05_best.pt",
                              map_location=dev, weights_only=False)["state_dict"], strict=False)
ar.eval(); ar.use_frame = False
fck = torch.load(REPO / "liquid_coupling_flow/artifacts/ka3d_cavity_ersi_rho12_best.pt", map_location=dev, weights_only=False)
FA = fck["args"]; K_MAX = FA["K_MAX"]; N_CAGE = FA["N_CAGE"]
flow = CavityBlockFlow(n_cage=N_CAGE, k=K_MAX, r_c=2.5, hidden_nf=FA["hidden"], n_layers=FA["layers"],
                       n_species=2, max_neighbors=16).to(dev)
flow.load_state_dict(fck["state_dict"]); flow.eval()
print(f"G0 discovery control: R={R} NCAV={NCAV} | pools AR/FLOW/UNIFORM x {M_DRAW} draws | "
      f"quench {K_PRE} prescreened + {K_RND} random each | strict dU<{STRICT_DU} ov>{STRICT_OV}, "
      f"meta dU<{META_DU} ov>{META_OV}", flush=True)


def box_set(x, R):
    mm = x.norm(dim=-1) < R
    ijk = torch.floor(x[mm] / L_H).long()
    h = int(R / L_H) + 2; side = 2 * h + 1
    return set(((ijk[:, 0] + h) * side * side + (ijk[:, 1] + h) * side + (ijk[:, 2] + h)).tolist())


def box_ov(a, b, R):
    A, B = box_set(a, R), box_set(b, R)
    return len(A & B) / max(1, min(len(A), len(B)))


def quench_batch(Xb, Sb, bnd, sb, steps=QSTEPS, dmax=0.05):
    """Displacement-capped steepest descent, f64 (recorded: Adam diverges on r^-12)."""
    Mn, n, _ = Xb.shape; mm = bnd.shape[0]
    xi = Xb.double().clone().to(dev); bndd = bnd.double().to(dev)
    sfull = torch.cat([Sb.to(dev), sb[None].expand(Mn, mm).to(dev)], 1).long()
    with torch.enable_grad():
        for t in range(steps):
            xi = xi.detach().requires_grad_(True)
            U = ka_energy(torch.cat([xi, bndd[None].expand(Mn, mm, 3)], 1), sfull, BIGL)
            (gr,) = torch.autograd.grad(U.sum(), xi)
            step = dmax * (0.2 + 0.8 * (1 - t / steps))
            gn = gr.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            xi = (xi - gr / gn * torch.minimum(gn * 1e-3, torch.full_like(gn, step))).detach()
    Uf = ka_energy(torch.cat([xi, bndd[None].expand(Mn, mm, 3)], 1), sfull, BIGL).float()
    return Uf.cpu(), xi.float().cpu()


def dummies(m):
    d = torch.zeros(m, 3); d[:, 0] = DUMMY0 + DUMMY_D * torch.arange(m, dtype=torch.float32); return d


def sample_rk4(x0, cage_x, spb, spc, nsteps=30):
    k = flow.k; B = x0.shape[0]; cl = x0.clone(); sp = torch.cat([spb, spc], 1); dt = 1.0 / nsteps
    def vel(x, tv):
        tt = torch.full((B,), tv, device=x.device, dtype=x.dtype)
        v, _ = flow.ce.vel_div(torch.cat([x, cage_x], 1), tt, sp, k); return v[:, :k]
    for s in range(nsteps):
        t = s * dt
        k1 = vel(cl, t); k2 = vel(cl + 0.5 * dt * k1, t + 0.5 * dt)
        k3 = vel(cl + 0.5 * dt * k2, t + 0.5 * dt); k4 = vel(cl + dt * k3, t + dt)
        cl = cl + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    return cl


results = {"meta": {"R": R, "M_DRAW": M_DRAW, "K_PRE": K_PRE, "K_RND": K_RND,
                    "strict": (STRICT_DU, STRICT_OV), "metab": (META_DU, META_OV)}}
gen = torch.Generator().manual_seed(1616)
gdev = torch.Generator(device=dev).manual_seed(1616)
for cav_i in range(NCAV):
    cav = None
    for _ in range(80):
        ci = int(torch.randint(n_train, Xds.shape[0], (1,), generator=gen))
        c = torch.rand(3, generator=gen) * L
        pr = carve(Xds[ci], Sds[ci], c, R, L)
        if pr["n_in"] < 14 or pr["n_in"] > K_MAX:
            continue
        xin = _mic(pr["x_in"], c, L); sin = pr["s_in"].long()
        xout = _mic(pr["x_out"], c, L); bm = xout.norm(dim=-1) < (R + RCTX)
        bx, bs = xout[bm], pr["s_out"][bm].long()
        if bx.shape[0] >= 8:
            cav = (ci, xin, sin, bx, bs); break
    if cav is None:
        continue
    ci, xin, sin, bx, bs = cav; n = xin.shape[0]
    t0 = time.time()
    # reference quench
    Uref, Xref_is = quench_batch(xin[None], sin[None], bx, bs)
    uref = float(Uref[0]) / n
    # ---- pools ----
    pools = {}
    # AR
    xo, so, _ = label_to_scaffold(xin.to(dev), sin.to(dev), R)
    allm = torch.ones(n, dtype=torch.bool, device=dev)
    Xa = []
    for b0 in range(0, M_DRAW, 512):
        nb = min(512, M_DRAW - b0)
        xa, sa, _ = ar.sample_block_b(xo[None].expand(nb, n, 3).clone(), so[None].expand(nb, n).clone(),
                                      allm, bx.to(dev), bs.to(dev), R, gen=gdev)
        Xa.append(xa.cpu())
    pools["AR"] = torch.cat(Xa)
    # FLOW
    idx = bx.norm(dim=-1).argsort()[:N_CAGE]; cage_x = bx[idx]; sp_cage = bs[idx]; nc = cage_x.shape[0]
    if nc < N_CAGE:
        cage_x = torch.cat([cage_x, dummies(N_CAGE - nc)]); sp_cage = torch.cat([sp_cage, torch.zeros(N_CAGE - nc, dtype=torch.long)])
    Xf = []
    for b0 in range(0, M_DRAW, 64):
        nb = min(64, M_DRAW - b0)
        x0 = torch.zeros(nb, K_MAX, 3); spb = torch.zeros(nb, K_MAX, dtype=torch.long)
        dpad = dummies(K_MAX - n) if K_MAX > n else None
        for s in range(nb):
            u = torch.randn(n, 3, generator=gen); u = u / u.norm(dim=-1, keepdim=True)
            rr = R * torch.rand(n, 1, generator=gen) ** (1.0 / 3.0)
            x0[s, :n] = u * rr; spb[s, :n] = sin
            if dpad is not None:
                x0[s, n:] = dpad
        xf = sample_rk4(x0.to(dev), cage_x[None].expand(nb, -1, -1).to(dev),
                        spb.to(dev), sp_cage[None].expand(nb, -1).to(dev))
        Xf.append(xf[:, :n].cpu())
    pools["FLOW"] = torch.cat(Xf)
    # UNIFORM
    u = torch.randn(M_DRAW, n, 3, generator=gen); u = u / u.norm(dim=-1, keepdim=True)
    rr = R * torch.rand(M_DRAW, n, 1, generator=gen) ** (1.0 / 3.0)
    pools["UNIFORM"] = u * rr
    print(f"=== cav {ci} (n={n}) U_IS(ref)/n = {uref:+.3f} | pools drawn ({time.time()-t0:.0f}s) ===", flush=True)
    # ---- prescreen + quench + membership, identical for each pool ----
    for tag, X in pools.items():
        pre = torch.tensor([box_ov(X[m_], xin, R) for m_ in range(M_DRAW)])
        top = pre.argsort(descending=True)[:K_PRE]
        rnd = torch.randperm(M_DRAW, generator=gen)[:K_RND]
        sel = torch.cat([top, rnd])
        Uq, Xq = quench_batch(X[sel], sin[None].expand(len(sel), n).clone(), bx, bs)
        du = Uq / n - uref
        ov = torch.tensor([box_ov(Xq[j], Xref_is[0], R) for j in range(len(sel))])
        strict = (du.abs() < STRICT_DU) & (ov > STRICT_OV)
        metab = (du.abs() < META_DU) & (ov > META_OV)
        s_pre, s_rnd = int(strict[:K_PRE].sum()), int(strict[K_PRE:].sum())
        m_pre, m_rnd = int(metab[:K_PRE].sum()), int(metab[K_PRE:].sum())
        results[(ci, tag)] = {"pre_scores": pre, "sel": sel, "U_IS_n": (Uq / n), "du": du, "ov": ov,
                              "strict": strict, "metab": metab, "uref": uref}
        torch.save(results, OUT)
        q = torch.quantile(Uq / n, torch.tensor([0.1, 0.5, 0.9]))
        print(f"  {tag:>8}: U_IS/n [{float(q[0]):+.2f} {float(q[1]):+.2f} {float(q[2]):+.2f}] "
              f"| STRICT pre {s_pre}/{K_PRE} rnd {s_rnd}/{K_RND} | META pre {m_pre}/{K_PRE} rnd {m_rnd}/{K_RND} "
              f"| prescreen-top ov {float(pre[top].mean()):.2f}", flush=True)
print(f"saved -> {OUT}", flush=True)
