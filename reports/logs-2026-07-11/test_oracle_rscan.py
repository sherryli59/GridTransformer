"""GATE part 2: R-SCAN of the oracle full-cage heat-bath with WHOLE-INTERIOR resampling (the PTS move).

At R=2.0 the cavity is PTS-pinned (rearr 2%, single basin) so basin-crossing was unmeasurable. Here we scan
R in {2.5, 3.0, 3.5} (approaching/entering the multi-basin regime, xi_PTS~3.8 at this T) and resample the
ENTIRE interior (cage = frozen boundary only), asking:
  1. Does oracle heat-bath + guard still reach near-equilibrium?  (dE/particle < ~2)
  2. Does it find genuinely DIFFERENT arrangements at larger R?   (rearr >> data-start intrinsic)
  3. Do AR seeds BEAT uniform seeds (sweeps-to-basin, arrangement diversity)? — the learned model's
     value-add; if uniform matches AR, option B collapses into plain constrained MC (LJ-liquid lesson).
Arms per cavity: AR-init / UNIFORM-init / DATA-init (control). Species: AR arm uses AR species; uniform arm
uses the DATA species (uniform has no species model); control uses data species."""
import torch, statistics as st
from liquid_coupling_flow.ka3d_ebm_batched import KA3DScaffoldEBMBatched
from liquid_coupling_flow.ka3d_scaffold_ar import label_to_scaffold, fixed_ball_scaffold
from liquid_coupling_flow.ka3d_cavity_ar import _mic
from liquid_coupling_flow.ka3d_cavity_carve import carve
from liquid_coupling_flow.ka_energy import ka_energy, SIGMA, EPS, RCUT_FACTOR

dev = "cuda"; RCTX = 2.5; M = 8; BIGL = 100.0; BETA = 2.0
NCAND = 192; SWEEPS = 6; SIGLOC = 0.15; NCAV = 5
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


def heatbath(x_int, s_int, bnd, sb, Rr, gen, sweeps=SWEEPS):
    """Whole-interior per-site heat-bath (cage = boundary only). Returns dE-trajectory snapshots + final."""
    Mn, Nn = x_int.shape[0], x_int.shape[1]; traj = {}
    for sw in range(1, sweeps + 1):
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
        traj[sw] = x_int.clone()
    return traj


def rearr_frac(x_int, data_x, thr=0.6):
    d = torch.cdist(x_int, data_x[None].expand(x_int.shape[0], *data_x.shape))
    return float((d.min(2).values > thr).float().mean())


m = load_base()
d = torch.load("liquid_coupling_flow/artifacts/ka3d_dataset_N512_T0.5.pt", map_location=dev, weights_only=False)
X, S, L = d["x"].to(dev).float(), d["s"].to(dev).long(), float(d["L"])
print(f"whole-interior oracle heat-bath R-scan (M={M}, {NCAND} cands, {SWEEPS} sweeps, beta={BETA})", flush=True)
for Rr in (2.5, 3.0, 3.5):
    gen = torch.Generator(device=dev).manual_seed(0)
    agg = {k: [] for k in ["arE", "unE", "ctE", "arR", "unR", "ctR", "arT", "unT", "n_in"]}
    ncav = 0
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
        arms = {"ar": (xa.contiguous(), sa), "un": (xu, so[None].expand(M, n).contiguous()),
                "ct": (xo[None].expand(M, n, 3).contiguous(), so[None].expand(M, n).contiguous())}
        for key, (x0, s0) in arms.items():
            traj = heatbath(x0, s0, bnd, sb, Rr, gen)
            dEs = [(full_E(traj[sw], s0, bnd, sb).mean().item() - E_data) / n for sw in range(1, SWEEPS + 1)]
            tt = next((sw for sw, e in zip(range(1, SWEEPS + 1), dEs) if e < 2.0), SWEEPS + 1)
            agg[key + "E"].append(dEs[-1]); agg[key + "R"].append(rearr_frac(traj[SWEEPS], xo))
            if key != "ct":
                agg[key + "T"].append(tt)
        agg["n_in"].append(n); ncav += 1
        if ncav >= NCAV:
            break
    md = lambda k: st.median(agg[k])
    print(f"\nR={Rr}  n_in~{int(st.mean(agg['n_in']))}  ({ncav} cavities)", flush=True)
    print(f"  dE/part @sweep{SWEEPS}:  AR {md('arE'):+6.2f}   UNIF {md('unE'):+6.2f}   DATA-ctl {md('ctE'):+6.2f}", flush=True)
    print(f"  sweeps-to-basin(<2): AR {md('arT'):.0f}   UNIF {md('unT'):.0f}", flush=True)
    print(f"  rearr vs data:       AR {100*st.mean(agg['arR']):3.0f}%  UNIF {100*st.mean(agg['unR']):3.0f}%  DATA-ctl {100*st.mean(agg['ctR']):3.0f}%", flush=True)
print("\nread-out: basin-crossing exists where AR/UNIF rearr >> DATA-ctl with dE<2; AR value-add = faster "
      "sweeps-to-basin and/or higher clean diversity than UNIF.", flush=True)
