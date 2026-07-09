"""Physics-baseline scramble-arm test: can TUNED per-particle displacement Metropolis (masked to mobile,
NOT OOD-limited, no learned proposal) close the two-arm gap at T=0.8 that MALA and the learned cluster moves
couldn't? Decides whether PTS-by-pinning is feasible with the physics baseline doing the interior transport.
Displacement only (no swaps) so species stay shared [N] and _u_matrix (parallel-Metropolis dE) applies."""
import time, torch, numpy as np
from liquid_coupling_flow.ka_mcmc_fast import _u_matrix
from liquid_coupling_flow.ka_pin_overlap import cell_occupancy, pinned_cells, overlap_Q
from liquid_coupling_flow.ka_energy import ka_energy
from liquid_coupling_flow.ka_pin_extract import stretched_exp_fit

dev = "cuda"; N = 256; T = 0.8; beta = 1.0 / T; B = 12; c = 0.16
refs = torch.load("liquid_coupling_flow/artifacts/pts/refs_T0.8_N256.pt", map_location=dev, weights_only=False)
x0 = refs["x"][:B].to(dev); s = refs["s"].to(dev).long(); L = refs["L"]      # s shared [N]
torch.manual_seed(0); n_pin = int(np.ceil(c * N))
mob = torch.ones(B, N, dtype=torch.bool, device=dev)
for b in range(B):
    mob[b, torch.randperm(N, device=dev)[:n_pin]] = False
occ_ref = cell_occupancy(x0, L); excl = pinned_cells(x0, mob, L)
mobf = mob[..., None].to(x0.dtype)


def masked_disp(x, step):
    prop = torch.remainder(torch.where(mob[..., None], x + step * torch.randn_like(x), x), L)
    dE = (_u_matrix(prop, x, s, s, L, True) - _u_matrix(x, x, s, s, L, True)).sum(-1)   # [B,N]
    acc = (torch.log(torch.rand_like(dE)) < (-beta * dE)) & mob
    return torch.where(acc[..., None], prop, x), float(acc.float().sum() / mob.float().sum())


# quick step tune -> ~0.4 acceptance
xt = x0.clone()
for st in (0.06, 0.10, 0.15, 0.20):
    _, a = masked_disp(xt, st)
    print(f"[phys] step {st}: per-particle accept {a:.2f}", flush=True)
STEP = 0.12


def run(arm, n_iter=1500, ndisp=30, rec=100):
    x = x0.clone() if arm == "ref" else torch.where(mob[..., None], torch.remainder(torch.rand_like(x0) * L, L), x0)
    t = []; Q = []; da = 0.0; t0 = time.time()
    for it in range(n_iter + 1):
        if it % rec == 0:
            t.append(it); Q.append(overlap_Q(cell_occupancy(x, L), occ_ref, excl))
            print(f"  [{arm}] it {it:4d} Q {Q[-1]:.3f} ({time.time()-t0:.0f}s)", flush=True)
        if it == n_iter:
            break
        for _ in range(ndisp):
            x, a = masked_disp(x, STEP); da += a
    return t, Q, da / (n_iter * ndisp)


print(f"[phys] N={N} T={T} c={c} B={B}, displacement-only step={STEP}, {30} sweeps/iter", flush=True)
tr, Qr, ar = run("ref"); ts, Qs, as_ = run("scramble")
fr = stretched_exp_fit(tr, Qr); fs = stretched_exp_fit(ts, Qs)
print(f"[phys] ref Q {Qr[0]:.3f}->{Qr[-1]:.3f} (Qinf {fr['Qinf']:.3f}) accept {ar:.2f}", flush=True)
print(f"[phys] scr Q {Qs[0]:.3f}->{Qs[-1]:.3f} (Qinf {fs['Qinf']:.3f}) accept {as_:.2f}", flush=True)
print(f"[phys] TWO-ARM GAP = {abs(fr['Qinf']-fs['Qinf']):.3f}  (MALA 0.23, cluster-MTM 0.36; need <=0.02)", flush=True)
print("[phys] DONE", flush=True)
