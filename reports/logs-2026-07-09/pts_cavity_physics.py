"""CAVITY point-to-set gate (vs random pinning). The decisive test for whether the learned interior sampler
has ANY load-bearing niche: does PHYSICS (displacement + in-place swap) equilibrate a WALLED cavity interior at
T=0.5, and how does that degrade with cavity radius R?

Cavity = freeze every particle OUTSIDE radius R of a fixed center; mobile = the compact interior blob. Unlike
random pinning there are NO percolating mobile channels, so displacement can only rearrange within the blob.
Two-arm init-independence gate on the DEEP interior (cells within R-shell of center, so the wall doesn't
trivially pin the measured cells):
  ref arm      -> interior starts at reference (Q decays from 1)
  scramble arm -> interior hot-melted among the frozen wall (Q rises), VALID (Metropolis melt, not jammed)
Convergence gap = |Q_inf(ref) - Q_inf(scramble)|; need <=0.05.

Outcomes:
  (1) all R converge          -> cavity is ALSO physics; NN has no niche in cavity PTS either.
  (2) small-R conv, large-R stall (gap grows with R) -> NICHE: walled interior collective rearrangement.
  (3) stalls but so would every learned mover (collective-move wall) -> kinetic, still not load-bearing.
N=256, T=0.5 (PT2 cold ref). Reuses masked_swap + parallel-Metropolis disp (consistent with pts_T05_physics)."""
import time, torch, numpy as np
from liquid_coupling_flow.ka_energy import SIGMA, EPS, RCUT_FACTOR, ka_energy
from liquid_coupling_flow.ka_pin import masked_swap
from liquid_coupling_flow.ipl44.ipl_swap_smc import uniform_weight_fn
from liquid_coupling_flow.ka_pin_overlap import cell_occupancy, overlap_Q, q_rand
from liquid_coupling_flow.ka_cavity import assert_mobile_inside, cavity_inside

dev = "cuda"; N = 256; T = 0.5; beta = 1.0 / T; B = 12; STEP = 0.08; SHELL = 1.0; QR = q_rand()
R_LIST = (3.0, 4.0, 5.0)
d = torch.load("liquid_coupling_flow/artifacts/pt_ladder_hb_N256.pt", map_location=dev, weights_only=False)
x0 = d["configs_per_rung"][0][:B].to(dev); s0 = d["s"].to(dev).long()[None].expand(B, -1).contiguous(); L = d["L"]
center = torch.tensor([L / 2, L / 2], device=dev)
t_sig = torch.tensor(SIGMA, device=dev); t_eps = torch.tensor(EPS, device=dev)
eye = torch.eye(N, device=dev, dtype=torch.bool)[None]
g = max(1, int(round(L / 0.3)))


def disp(x, s, mob, bet, step, R, stats=None):
    a = s[:, :, None].expand(-1, -1, N); b = s[:, None, :].expand(-1, N, -1)
    sig = t_sig[a, b]; eps = t_eps[a, b]; rc = RCUT_FACTOR * sig
    prop = torch.remainder(torch.where(mob[..., None], x + step * torch.randn_like(x), x), L)
    # A frozen exterior alone is not a strict cavity: a mobile labelled particle
    # can in principle thread through it.  Reject all proposals beyond R, as in
    # the hard spherical wall used in the cavity-PTS literature.
    inside = cavity_inside(prop, center, R, L)
    wall_reject = mob & ~inside
    if stats is not None:
        stats["wall_reject"] += int(wall_reject.sum())
        stats["mobile_proposals"] += int(mob.sum())

    def cross(xa):
        diff = xa[:, :, None, :] - x[:, None, :, :]; diff = diff - L * torch.round(diff / L)
        r2 = (diff ** 2).sum(-1).masked_fill(eye, 1e12); inv6 = (sig ** 2 / r2) ** 3
        e = 4 * eps * (inv6 ** 2 - inv6); src6 = (sig / rc) ** 6
        return torch.where(r2 < rc ** 2, e - 4 * eps * (src6 ** 2 - src6), torch.zeros_like(e)).sum(-1)
    dE = cross(prop) - cross(x)
    acc = (torch.log(torch.rand_like(dE)) < (-bet * dE)) & mob & inside
    return torch.where(acc[..., None], prop, x)


def melt(mob, R, T_hot=1.5, n=400, stats=None):
    x = x0.clone()
    for _ in range(n):
        x = disp(x, s0, mob, 1.0 / T_hot, 0.12, R, stats)
    assert_mobile_inside(x, mob, center, R, L)
    return x


def cavity_masks(R):
    dp = x0 - center[None, None]; dp = dp - L * torch.round(dp / L)
    mob = dp.norm(dim=-1) < R                                            # [B,N] mobile = interior
    gi = torch.arange(g, device=dev); cc = (gi + 0.5) * L / g
    ii = gi[:, None].expand(g, g).reshape(-1); jj = gi[None, :].expand(g, g).reshape(-1)
    cd = torch.stack([cc[ii], cc[jj]], -1) - center[None]; cd = cd - L * torch.round(cd / L)
    deep = cd.norm(dim=-1) < (R - SHELL)                                 # deep-interior cells only
    excl = (~deep)[None].expand(B, -1).contiguous()
    return mob, excl


def two_arm(mob, R, occ_ref, excl, n_iter=3000, ndisp=30, nswap=8, rec=300):
    efn = lambda a, bb: ka_energy(a, bb, L)
    out = {}
    for arm in ("ref", "scramble"):
        stats = {"wall_reject": 0, "mobile_proposals": 0}
        x = x0.clone() if arm == "ref" else melt(mob, R, stats=stats)
        assert_mobile_inside(x, mob, center, R, L)
        s = s0.clone(); U = efn(x, s); Q = []
        for it in range(n_iter + 1):
            if it % rec == 0:
                Q.append(overlap_Q(cell_occupancy(x, L), occ_ref, excl))
            if it == n_iter:
                break
            for _ in range(ndisp):
                x = disp(x, s, mob, beta, STEP, R, stats)
            U = efn(x, s)
            for _ in range(nswap):
                s, U, _ = masked_swap(x, s, U, mob, beta, efn, uniform_weight_fn)
        assert_mobile_inside(x, mob, center, R, L)
        out[arm] = {"Qinf": float(np.mean(Q[-3:])), **stats}
    return out


torch.manual_seed(0)
occ_ref0 = cell_occupancy(x0, L)
print(f"[cav] N={N} T={T} step={STEP} shell={SHELL} Q_rand={QR:.3f}; walled cavity interior physics gate", flush=True)
t0 = time.time()
for R in R_LIST:
    mob, excl = cavity_masks(R)
    n_mob = float(mob.float().sum(1).mean()); n_deepcell = int((~excl[0]).sum())
    melt_stats = {"wall_reject": 0, "mobile_proposals": 0}
    scr0 = melt(mob, R, stats=melt_stats)
    scr_q = overlap_Q(cell_occupancy(scr0, L), occ_ref0, excl)
    scr_U = float((ka_energy(scr0, s0, L) / N).median())
    o = two_arm(mob, R, occ_ref0, excl)
    gap = abs(o["ref"]["Qinf"] - o["scramble"]["Qinf"])
    wall_rej = sum(v["wall_reject"] for v in o.values()) + melt_stats["wall_reject"]
    wall_prop = sum(v["mobile_proposals"] for v in o.values()) + melt_stats["mobile_proposals"]
    print(f"[cav R={R:.1f}] n_mob~{n_mob:.0f} deepcells={n_deepcell} | melt-start Q={scr_q:.3f} U/N={scr_U:+.3f} "
          f"| ref {o['ref']['Qinf']:.3f} scr {o['scramble']['Qinf']:.3f} GAP={gap:.3f} "
          f"excess(ref)~{o['ref']['Qinf']-QR:.3f} | hard-wall rejects {wall_rej}/{wall_prop} "
          f"-> {'CONVERGED' if gap <= 0.05 else 'STALL'}  ({time.time()-t0:.0f}s)", flush=True)
print("[cav] DONE", flush=True)
