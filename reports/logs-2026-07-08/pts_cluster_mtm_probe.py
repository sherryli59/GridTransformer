"""Decisive test: does LEARNED CLUSTER-MTM (collective, k=7, M=32) + MALA close the two-arm init-independence
gap at T=0.8 that MALA-alone left at 0.23? Pins in slot-space (occupancy Q is order-invariant); moves only
all-mobile clusters per chain (frozen = fixed boundary; exact per chain). This is the user-directed collective
learned-proposal interior sampler."""
import time, torch, numpy as np
from liquid_coupling_flow import ka_cluster as KC
from liquid_coupling_flow.ka_cluster_flow import slot_order, _scaffold, _load, ART
from liquid_coupling_flow.ka_cluster_mtm import mtm_move
from liquid_coupling_flow.ka_pin import masked_mala
from liquid_coupling_flow.ka_pin_overlap import cell_occupancy, pinned_cells, overlap_Q
from liquid_coupling_flow.ka_energy import ka_energy, ka_forces
from liquid_coupling_flow.ka_pin_extract import stretched_exp_fit

dev = "cuda"; N = 256; T = 0.8; beta = 1.0 / T; B = 12; c = 0.16
sc, L, geo = _scaffold(N, dev)
P = _load(torch.load(f"{ART}/ka_cluster_flow_full_N100.pt", map_location=dev, weights_only=False), dev)
d = torch.load(f"{ART}/pt_ladder_hb_N256.pt", map_location=dev, weights_only=False)
pos0, s = slot_order(d["configs_per_rung"][0][:B].to(dev), d["s"].to(dev).long(), geo, N)
# pin random slot-columns (physically = frozen particles; Q order-invariant)
torch.manual_seed(0); n_pin = int(np.ceil(c * N))
mob = torch.ones(B, N, dtype=torch.bool, device=dev)
for b in range(B):
    mob[b, torch.randperm(N, device=dev)[:n_pin]] = False
occ_ref = cell_occupancy(pos0, L); excl = pinned_cells(pos0, mob, L)
efn = lambda a, b: ka_energy(a, b, L); ffn = lambda a, b: ka_forces(a, b, L)
print(f"[clmix] N={N} T={T} c={c} B={B}, cluster-MTM(k=3,M=64)+MALA, mobile frac {float(mob.float().mean()):.2f}", flush=True)


_gcpu = torch.Generator().manual_seed(0)
def masked_mtm(pos, M, k, n_seeds, gen):
    probs = []; seeds = torch.randperm(N, generator=_gcpu)[:n_seeds].tolist()   # CPU gen for randperm
    for seed in seeds:
        cl = KC.cluster_slots(seed, sc, k, L)
        allmob = mob[:, cl].all(1)
        if not bool(allmob.any()):
            continue
        pn, moved, info = mtm_move(P, pos, s, cl, sc, L, M=M, beta=beta, gen=gen)
        pos = pos.clone(); pos[:, cl] = torch.where(allmob[:, None, None], pn[:, cl], pos[:, cl])
        probs.append(float((moved & allmob).float().mean()))
    return pos, (sum(probs) / len(probs) if probs else 0.0)


def run(arm, n_iter=200, mala=8, M=64, k=3, n_seeds=15, rec=20):
    gen = torch.Generator(device=dev).manual_seed(3 if arm == "ref" else 4)
    x = pos0.clone() if arm == "ref" else torch.where(mob[..., None], torch.remainder(torch.rand_like(pos0) * L, L), pos0)
    U = efn(x, s); t = []; Q = []; ca = 0.0; t0 = time.time()
    for it in range(n_iter + 1):
        if it % rec == 0:
            t.append(it); Q.append(overlap_Q(cell_occupancy(x, L), occ_ref, excl))
            print(f"  [{arm}] it {it:4d} Q {Q[-1]:.3f} ({time.time()-t0:.0f}s)", flush=True)
        if it == n_iter:
            break
        for _ in range(mala):
            x, U, _ = masked_mala(x, s, U, mob, beta, L, 0.01, efn, ffn)
        x, a = masked_mtm(x, M, k, n_seeds, gen); ca += a; U = efn(x, s)
    return t, Q, ca / n_iter


for (k, M, ns, nit, tag) in [(3, 64, 15, 200, "k3M64"), (7, 256, 6, 150, "k7M256")]:
    print(f"\n[clmix] === config {tag} (k={k}, M={M}, {ns} seeds/iter, {nit} iters) ===", flush=True)
    tr, Qr, ar = run("ref", n_iter=nit, M=M, k=k, n_seeds=ns)
    ts, Qs, as_ = run("scramble", n_iter=nit, M=M, k=k, n_seeds=ns)
    fr = stretched_exp_fit(tr, Qr); fs = stretched_exp_fit(ts, Qs)
    print(f"[clmix {tag}] ref Q {Qr[0]:.3f}->{Qr[-1]:.3f} (Qinf {fr['Qinf']:.3f}) cluster-accept {ar:.3f}", flush=True)
    print(f"[clmix {tag}] scr Q {Qs[0]:.3f}->{Qs[-1]:.3f} (Qinf {fs['Qinf']:.3f}) cluster-accept {as_:.3f}", flush=True)
    print(f"[clmix {tag}] TWO-ARM GAP = {abs(fr['Qinf']-fs['Qinf']):.3f}  (MALA-alone 0.23; need <=0.02)", flush=True)
print("\n[clmix] DONE ALL", flush=True)
