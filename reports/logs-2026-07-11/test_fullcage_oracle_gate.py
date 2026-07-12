"""CHEAP GATE for the full-cage direction (Option B): can an ORACLE full-cage conditional + energy guard turn
AR half-cage K=12 blocks into near-basin, genuinely-rearranged configurations?

Oracle = per-site heat-bath under the TRUE KA energy: for each block particle in turn, resample its position
from ~exp(-beta*U(x_i | all others)) via a dense candidate cloud (uniform-in-cavity teleports + local Gaussian
+ keep-current). This is the CEILING of any learned full-cage position model (Option B amortizes exactly this
conditional). No training. If even the oracle cannot reach <~+2/particle with real rearrangement, options A/B
are dead. If it can, B is GO (train a flow to amortize it).

Measures per cavity (M=8 chains, K=12, beta=2):
  dE/particle vs the equilibrium data block  (raw AR -> after 1,3,6 heat-bath sweeps)
  guard-survival: fraction of chains with dE < 3/particle (what an energy guard would keep)
  rearrangement: fraction of block particles farther than 0.6 from EVERY data-block particle (new arrangement,
                 not a copy) + cross-chain diversity as before
  CONTROL: chains started from the DATA block (heat-bath must hold equilibrium: dE~0; its rearrangement rate
           shows the intrinsic mobility allowed by this cavity => the basin-crossing yardstick)."""
import torch, statistics as st
from liquid_coupling_flow.ka3d_ebm_batched import KA3DScaffoldEBMBatched
from liquid_coupling_flow.ka3d_scaffold_ar import label_to_scaffold, fixed_ball_scaffold
from liquid_coupling_flow.ka3d_cavity_ar import _mic
from liquid_coupling_flow.ka3d_cavity_carve import carve
from liquid_coupling_flow.ka_energy import ka_energy, SIGMA, EPS, RCUT_FACTOR

dev = "cuda"; R = 2.0; RCTX = 2.5; K = 12; M = 8; BIGL = 100.0; BETA = 2.0
NCAND = 256; SWEEPS = 6; SIGLOC = 0.15
T_SIG = torch.tensor(SIGMA, device=dev); T_EPS = torch.tensor(EPS, device=dev)


def load_base():
    ck = torch.load("liquid_coupling_flow/artifacts/ka3d_cavity_ebm3ax.pt", map_location=dev, weights_only=False)
    m = KA3DScaffoldEBMBatched(cat_bins=128, cat_range=2.5).to(dev)
    m.load_state_dict(ck["state_dict"], strict=False); m.eval(); m.use_frame = False
    return m


def row_E(cand, s_i, ox, os_):
    """Shifted/cutoff LJ row energy of candidates vs others; same convention as ka_energy (no MIC needed:
    frame is center-relative, all |x|<R+RCTX << BIGL). cand [M,C,3], s_i [M], ox [M,P,3], os_ [M,P] -> [M,C]."""
    sig = T_SIG[s_i[:, None], os_]                                                    # [M,P]
    eps = T_EPS[s_i[:, None], os_]
    rc2 = (RCUT_FACTOR * sig) ** 2
    r2 = (cand[:, :, None, :] - ox[:, None, :, :]).square().sum(-1)                   # [M,C,P]
    inv6 = (sig[:, None] ** 2 / r2) ** 3
    e = 4 * eps[:, None] * (inv6 ** 2 - inv6)
    src6 = (1.0 / RCUT_FACTOR) ** 6                                                   # (sig/rc)^6 constant
    e = torch.where(r2 < rc2[:, None], e - 4 * eps[:, None] * (src6 ** 2 - src6), torch.zeros_like(e))
    return e.sum(-1)                                                                  # [M,C]


def verify_rowE():
    g = torch.Generator(device=dev).manual_seed(3)
    x = torch.rand(2, 9, 3, generator=g, device=dev) * 2 - 1; s = torch.randint(0, 2, (2, 9), generator=g, device=dev)
    tot = ka_energy(x, s.long(), BIGL)
    acc = torch.zeros(2, device=dev)
    for i in range(9):
        oth = [j for j in range(9) if j != i]
        acc += 0.5 * row_E(x[:, i:i + 1], s[:, i], x[:, oth], s[:, oth])[:, 0]
    assert torch.allclose(tot, acc, atol=1e-4), (tot, acc)


def block_E(xblk, sblk, cage_x, cage_s):
    allx = torch.cat([xblk, cage_x], 1); alls = torch.cat([sblk, cage_s], 1)
    return ka_energy(allx, alls.long(), BIGL)


def heatbath_sweeps(xb, sblk, cage_x, cage_s, gen, sweeps=SWEEPS, snap_at=(1, 3, 6)):
    """Oracle full-cage per-site heat-bath over the K block particles. Returns snapshots at snap_at sweeps."""
    Mn, Kk = xb.shape[0], xb.shape[1]; snaps = {}
    for sw in range(1, sweeps + 1):
        for i in torch.randperm(Kk, generator=gen, device=dev).tolist():
            oth = [j for j in range(Kk) if j != i]
            ox = torch.cat([xb[:, oth], cage_x], 1); os_ = torch.cat([sblk[:, oth], cage_s], 1)
            # candidate cloud: 60% uniform-in-ball teleports, 40% local Gaussian, + keep-current
            n_tel = int(0.6 * NCAND)
            u = torch.randn(Mn, n_tel, 3, generator=gen, device=dev)
            u = u / u.norm(dim=-1, keepdim=True) * (torch.rand(Mn, n_tel, 1, generator=gen, device=dev) ** (1 / 3)) * R
            loc = xb[:, i:i + 1] + SIGLOC * torch.randn(Mn, NCAND - n_tel, 3, generator=gen, device=dev)
            cand = torch.cat([u, loc, xb[:, i:i + 1]], 1)                              # [M,C+1,3]
            cand = cand.clamp(-R - 0.001, R + 0.001)                                   # stay in frame (ball approx)
            logits = -BETA * row_E(cand, sblk[:, i], ox, os_)
            pick = torch.multinomial(torch.softmax(logits, 1), 1, generator=gen).squeeze(1)
            xb = xb.clone(); xb[:, i] = cand[torch.arange(Mn, device=dev), pick]
        if sw in snap_at:
            snaps[sw] = xb.clone()
    return snaps


def rearrange_frac(xblk, data_blk, thr=0.6):
    """fraction of block particles farther than thr from EVERY data-block particle (new arrangement)."""
    d = torch.cdist(xblk, data_blk[0:1].expand(xblk.shape[0], K, 3))                   # [M,K,K]
    return float((d.min(2).values > thr).float().mean())


def cross_div(xblk):
    return st.mean([float((torch.cdist(xblk[i], xblk[j]).min(1).values < 0.3).float().mean())
                    for i in range(min(xblk.shape[0], 8)) for j in range(i + 1, min(xblk.shape[0], 8))])


verify_rowE(); print("row_E == ka_energy convention: verified", flush=True)
m = load_base()
d = torch.load("liquid_coupling_flow/artifacts/ka3d_dataset_N512_T0.5.pt", map_location=dev, weights_only=False)
X, S, L = d["x"].to(dev).float(), d["s"].to(dev).long(), float(d["L"]); gen = torch.Generator(device=dev).manual_seed(0)
res = {k: [] for k in ["raw", "s1", "s3", "s6", "ctl6", "guard", "rearr", "rearr_ctl", "div"]}
ncav = 0
for ci in range(900, 970):
    c = torch.rand(3, generator=gen, device=dev) * L; p = carve(X[ci], S[ci], c, R, L)
    if p["n_in"] < K + 6:
        continue
    xo, so, _ = label_to_scaffold(_mic(p["x_in"], c, L), p["s_in"], R)
    xout = _mic(p["x_out"], c, L); bm = xout.norm(dim=-1) < (R + RCTX); bnd, sb = xout[bm], p["s_out"][bm]
    n = xo.shape[0]; a = fixed_ball_scaffold(n, R, dev); seed = int(torch.randint(n, (), generator=gen, device=dev))
    blk = torch.zeros(n, dtype=torch.bool, device=dev); blk[(a - a[seed]).norm(dim=-1).topk(K, largest=False).indices] = True
    order = torch.argsort(blk.to(torch.uint8), stable=True); n_ret = int((~blk).sum()); blk_re = blk[order]
    xo_re, so_re = xo[order], so[order]
    cage_x = torch.cat([bnd[None].expand(M, bnd.shape[0], 3), xo_re[None, :n_ret].expand(M, n_ret, 3)], 1)
    cage_s = torch.cat([sb[None].expand(M, bnd.shape[0]), so_re[None, :n_ret].expand(M, n_ret)], 1)
    data_blk = xo_re[None, n_ret:].expand(M, K, 3).contiguous(); data_bs = so_re[None, n_ret:].expand(M, K)
    E_data = block_E(data_blk, data_bs, cage_x, cage_s).mean().item()
    x0, s0, _ = m.sample_block_b(xo_re[None].expand(M, n, 3), so_re[None].expand(M, n), blk_re, bnd, sb, R, gen=gen)
    ar_blk = x0[:, n_ret:].contiguous(); ar_bs = s0[:, n_ret:]
    res["raw"].append((block_E(ar_blk, ar_bs, cage_x, cage_s).mean().item() - E_data) / K)
    snaps = heatbath_sweeps(ar_blk, ar_bs, cage_x, cage_s, gen)
    for sw, key in ((1, "s1"), (3, "s3"), (6, "s6")):
        res[key].append((block_E(snaps[sw], ar_bs, cage_x, cage_s).mean().item() - E_data) / K)
    dE6 = (block_E(snaps[6], ar_bs, cage_x, cage_s) - E_data) / K
    res["guard"].append(float((dE6 < 3.0).float().mean()))
    res["rearr"].append(rearrange_frac(snaps[6], data_blk)); res["div"].append(cross_div(snaps[6]))
    ctl = heatbath_sweeps(data_blk.clone(), data_bs, cage_x, cage_s, gen, snap_at=(6,))
    res["ctl6"].append((block_E(ctl[6], data_bs, cage_x, cage_s).mean().item() - E_data) / K)
    res["rearr_ctl"].append(rearrange_frac(ctl[6], data_blk))
    ncav += 1
    print(f"  cav {ncav}: raw {res['raw'][-1]:+9.1f} -> hb1 {res['s1'][-1]:+7.2f} hb3 {res['s3'][-1]:+6.2f} hb6 {res['s6'][-1]:+6.2f}"
          f" | guard {100*res['guard'][-1]:3.0f}% rearr {100*res['rearr'][-1]:3.0f}% (ctl {100*res['rearr_ctl'][-1]:3.0f}%) ctlE {res['ctl6'][-1]:+5.2f}", flush=True)
    if ncav >= 10:
        break

md = lambda k: st.median(res[k])
print(f"\nK={K}, {ncav} cavities, ORACLE full-cage heat-bath (beta={BETA}, {NCAND} cands, {SWEEPS} sweeps)", flush=True)
print(f"  dE/particle: raw {md('raw'):+9.1f} -> sweep1 {md('s1'):+6.2f} -> sweep3 {md('s3'):+6.2f} -> sweep6 {md('s6'):+6.2f}   (target < +2)", flush=True)
print(f"  CONTROL (data-start, 6 sweeps): {md('ctl6'):+6.2f}   (must stay ~0 => heat-bath holds equilibrium)", flush=True)
print(f"  guard-survival (dE<3/part): {100*st.mean(res['guard']):.0f}%   cross-chain div: {100*st.mean(res['div']):.0f}%", flush=True)
print(f"  rearrangement vs data block: AR-start {100*st.mean(res['rearr']):.0f}%  vs  data-start intrinsic {100*st.mean(res['rearr_ctl']):.0f}%", flush=True)
ok = md("s6") < 2.0 and md("ctl6") < 2.0 and st.mean(res["guard"]) > 0.3
print("GATE:", "PASS -> full-cage placement + guard reaches the basin; Option B GO (train the flow to amortize this)"
      if ok else "FAIL -> even the oracle can't; A/B dead, fall to D/C", flush=True)
