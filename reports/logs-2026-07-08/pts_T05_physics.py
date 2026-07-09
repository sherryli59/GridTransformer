"""(b)-gate: does PHYSICS converge the two-arm PTS gap at T=0.5 (deep glass)? Compares displacement-ONLY vs
displacement+SWAP-MC (per-chain species), both from a VALID melt init. If disp+swap converges -> physics wins,
no NN niche (kill b). If both plateau apart -> real gap for a learned collective POSITION mover (b has a target;
swap is species-only, position-blind for occupancy Q). N=256, T=0.5 (PT2 cold ref)."""
import time, torch, numpy as np
from liquid_coupling_flow.ka_energy import SIGMA, EPS, RCUT_FACTOR, ka_energy
from liquid_coupling_flow.ka_pin import masked_swap
from liquid_coupling_flow.ipl44.ipl_swap_smc import uniform_weight_fn
from liquid_coupling_flow.ka_pin_overlap import cell_occupancy, pinned_cells, overlap_Q, q_rand
from liquid_coupling_flow.ka_pin_extract import stretched_exp_fit

dev = "cuda"; N = 256; T = 0.5; beta = 1.0 / T; B = 12; c = 0.16; STEP = 0.08; QR = q_rand()
d = torch.load("liquid_coupling_flow/artifacts/pt_ladder_hb_N256.pt", map_location=dev, weights_only=False)
x0 = d["configs_per_rung"][0][:B].to(dev); s0 = d["s"].to(dev).long()[None].expand(B, -1).contiguous(); L = d["L"]
t_sig = torch.tensor(SIGMA, device=dev); t_eps = torch.tensor(EPS, device=dev)
eye = torch.eye(N, device=dev, dtype=torch.bool)[None]


def disp(x, s, mob, bet, step):
    a = s[:, :, None].expand(-1, -1, N); b = s[:, None, :].expand(-1, N, -1)
    sig = t_sig[a, b]; eps = t_eps[a, b]; rc = RCUT_FACTOR * sig
    prop = torch.remainder(torch.where(mob[..., None], x + step * torch.randn_like(x), x), L)

    def cross(xa):
        diff = xa[:, :, None, :] - x[:, None, :, :]; diff = diff - L * torch.round(diff / L)
        r2 = (diff ** 2).sum(-1).masked_fill(eye, 1e12); inv6 = (sig ** 2 / r2) ** 3
        e = 4 * eps * (inv6 ** 2 - inv6); src6 = (sig / rc) ** 6
        return torch.where(r2 < rc ** 2, e - 4 * eps * (src6 ** 2 - src6), torch.zeros_like(e)).sum(-1)
    dE = cross(prop) - cross(x)
    acc = (torch.log(torch.rand_like(dE)) < (-bet * dE)) & mob
    return torch.where(acc[..., None], prop, x)


def melt(mob, T_hot=1.5, n=400):
    x = x0.clone(); s = s0.clone()
    for _ in range(n):
        x = disp(x, s, mob, 1.0 / T_hot, 0.12)
    return x


def two_arm(mob, occ_ref, excl, use_swap, n_iter=2000, ndisp=20, nswap=8, rec=200):
    efn = lambda a, bb: ka_energy(a, bb, L)
    out = {}
    for arm in ("ref", "scramble"):
        x = x0.clone() if arm == "ref" else melt(mob)
        s = s0.clone(); U = efn(x, s); Q = []; t = []
        for it in range(n_iter + 1):
            if it % rec == 0:
                t.append(it); Q.append(overlap_Q(cell_occupancy(x, L), occ_ref, excl))
            if it == n_iter:
                break
            for _ in range(ndisp):
                x = disp(x, s, mob, beta, STEP)
            if use_swap:
                U = efn(x, s)
                for _ in range(nswap):
                    s, U, _ = masked_swap(x, s, U, mob, beta, efn, uniform_weight_fn)
        out[arm] = np.mean(Q[-3:])
    return out


torch.manual_seed(0); n_pin = int(np.ceil(c * N))
mob = torch.ones(B, N, dtype=torch.bool, device=dev)
for bch in range(B):
    mob[bch, torch.randperm(N, device=dev)[:n_pin]] = False
occ_ref = cell_occupancy(x0, L); excl = pinned_cells(x0, mob, L)
print(f"[T05] N={N} T={T} c={c} step={STEP} Q_rand={QR:.3f}; deep glass — physics gate for direction (b)", flush=True)
t0 = time.time()
for use_swap in (False, True):
    o = two_arm(mob, occ_ref, excl, use_swap)
    gap = abs(o["ref"] - o["scramble"])
    tag = "disp+SWAP" if use_swap else "disp-only"
    print(f"[T05 {tag}] ref {o['ref']:.3f} scr {o['scramble']:.3f} GAP={gap:.3f} excess(ref)~{o['ref']-QR:.3f} "
          f"-> {'CONVERGED' if gap<=0.05 else 'NOT-CONV'}  ({time.time()-t0:.0f}s)", flush=True)
print("[T05] DONE", flush=True)
