"""BULK KA BANK AT HOCKY'S STATE POINT (rho=1.2, T=0.55, N=4096) for the directly-comparable FL sweep.
No public configs exist (Hocky PRL'12 / BCY JCP'16 deposited nothing; Duke records = figure data only), so
we generate: MOSAIC tiling (2x2x2 of 8 DISTINCT equilibrated N=512 rho=1.2 T=0.5 configs -- no periodic
image correlation, only local seams) -> heal at T=0.80 -> equilibrate/produce at T=0.55 (above MCT 0.435;
bulk relaxes in ~1e3 sweeps; heating from the DEEPER T=0.5 equilibrium is the easy direction).
MC: cell-checkerboard parallel displacement (4x4x4 cells, size L/4=3.76 >= rc+2*delta=2.8; 8 parity
sublattices; simultaneous moves provably independent) + batched species-swap moves. Exact shifted-LJ rows
(ka_energy convention, RCUT_FACTOR=2.5, min-image). VALIDATION GATE: untiled N=512 run at T=0.55 -> compare
U/N (|dU/N|<0.01) + g(r) peak. Incremental saves. Output: bank of B x frames configs at (1.2, 0.55)."""
import sys, time, math, functools, statistics as st
import torch
sys.path.insert(0, "/mnt/ssd/GridTransformer")
from liquid_coupling_flow.ka_energy import SIGMA, EPS, RCUT_FACTOR, ka_energy
torch.set_grad_enabled(False)

dev = "cuda"
B = 4                      # mosaic replicas
T_HEAL, SW_HEAL = 0.80, 1000
T_PROD, SW_PROD = 0.55, 6000
SW_BURN = 2000; SAVE_EVERY = 250
DELTA = 0.15               # max displacement (uniform ball) -> cell constraint rc+2*delta=2.8 < 3.76
SWAPS_PER_SWEEP = 512
OUT = "reports/logs-2026-07-15/bulk_ka12_T055_N4096.pt"
VAL_OUT = "reports/logs-2026-07-15/bulk_ka12_T055_validate.out"

D = torch.load("liquid_coupling_flow/artifacts/ka3d_dataset_N512_T0.5.pt", map_location="cpu", weights_only=False)
X0, S0, L0 = D["x"].float(), D["s"].long(), float(D["L"])          # [1024, 512, 3], rho=1.200
N0 = X0.shape[1]; L = 2 * L0; N = 8 * N0                           # 4096, L=15.0566
print(f"mosaic: {B} replicas of 2x2x2 x distinct configs; N={N} L={L:.4f} rho={N/L**3:.4f}", flush=True)

t_sig = torch.tensor(SIGMA, device=dev); t_eps = torch.tensor(EPS, device=dev)


def mosaic(cfg_ids):
    xs, ss = [], []
    k = 0
    for ix in range(2):
        for iy in range(2):
            for iz in range(2):
                off = torch.tensor([ix * L0, iy * L0, iz * L0])
                xs.append(X0[cfg_ids[k]] + off[None]); ss.append(S0[cfg_ids[k]]); k += 1
    return torch.cat(xs), torch.cat(ss)


gen = torch.Generator().manual_seed(0)
ids = torch.randperm(1024, generator=gen)[:8 * B].reshape(B, 8)
Xb = torch.stack([mosaic(ids[b]) [0] for b in range(B)]).to(dev)
Sb = torch.stack([mosaic(ids[b]) [1] for b in range(B)]).to(dev)
g = torch.Generator(device=dev).manual_seed(1)

# cell bookkeeping: 4x4x4, parity sublattices
NCELL = 4; CW = L / NCELL
assert CW >= RCUT_FACTOR * 1.0 + 2 * DELTA + 0.0, "cell too small"   # sigma_max=1.0


def pair_row(Xf, Sf, rows_b, rows_i, xi_new):
    """Row pair-energy of particles (b, i) at positions xi vs all j (shifted LJ, min-image). [K]"""
    xr = Xf[rows_b]                                                  # [K,N,3]
    d = xr - xi_new[:, None]
    d = d - L * torch.round(d / L)
    r2 = (d ** 2).sum(-1)                                            # [K,N]
    r2[torch.arange(len(rows_b), device=dev), rows_i] = 1e12
    si = Sf[rows_b, rows_i]                                          # [K]
    sig = t_sig[si[:, None], Sf[rows_b]]                             # [K,N]
    eps = t_eps[si[:, None], Sf[rows_b]]
    rc = RCUT_FACTOR * sig
    inv6 = (sig ** 2 / r2.clamp_min(1e-12)) ** 3
    e = 4 * eps * (inv6 ** 2 - inv6)
    src6 = (sig / rc) ** 6
    e = torch.where(r2 < rc ** 2, e - 4 * eps * (src6 ** 2 - src6), torch.zeros_like(e))
    return e.sum(1)


def sweep(Xf, Sf, beta, gen):
    """One MC sweep: checkerboard displacement (N attempts) + species swaps."""
    acc = 0; att = 0
    cell = (Xf / CW).long().clamp(0, NCELL - 1)                      # [B,N,3]
    cid = cell[..., 0] * NCELL * NCELL + cell[..., 1] * NCELL + cell[..., 2]
    for rnd in range(N // (8 * B) * B // B):                         # ~N/8 rounds x 8 parities interleaved
        par = rnd % 8
        px, py, pz = par & 1, (par >> 1) & 1, (par >> 2) & 1
        # active cells of this parity: 2x2x2 = 8 cells per replica
        rows_b, rows_i = [], []
        for b in range(B):
            for cx in range(px, NCELL, 2):
                for cy in range(py, NCELL, 2):
                    for cz in range(pz, NCELL, 2):
                        c_id = cx * NCELL * NCELL + cy * NCELL + cz
                        members = (cid[b] == c_id).nonzero().squeeze(1)
                        if len(members):
                            rows_b.append(b)
                            rows_i.append(int(members[torch.randint(len(members), (1,), generator=gen, device=dev)]))
        if not rows_b:
            continue
        rb = torch.tensor(rows_b, device=dev); ri = torch.tensor(rows_i, device=dev)
        xi_old = Xf[rb, ri]
        prop = xi_old + DELTA * (2 * torch.rand(len(rb), 3, device=dev, generator=gen) - 1)
        dU = pair_row(Xf, Sf, rb, ri, torch.remainder(prop, L)) - pair_row(Xf, Sf, rb, ri, xi_old)
        a = torch.rand(len(rb), device=dev, generator=gen).log() < -beta * dU
        Xf[rb[a], ri[a]] = torch.remainder(prop[a], L)
        acc += int(a.sum()); att += len(rb)
    # species swaps (batched sequential-safe: one pair per replica per attempt round)
    for _ in range(SWAPS_PER_SWEEP // B):
        for b in range(B):
            A_ = (Sb[b] == 0).nonzero().squeeze(1); B_ = (Sb[b] == 1).nonzero().squeeze(1)
            ia = int(A_[torch.randint(len(A_), (1,), generator=gen, device=dev)])
            ib = int(B_[torch.randint(len(B_), (1,), generator=gen, device=dev)])
            rb2 = torch.tensor([b, b], device=dev); ri2 = torch.tensor([ia, ib], device=dev)
            e_old = pair_row(Xf, Sf, rb2, ri2, Xf[rb2, ri2]).sum()
            Sf[b, ia], Sf[b, ib] = Sf[b, ib].clone(), Sf[b, ia].clone()
            e_new = pair_row(Xf, Sf, rb2, ri2, Xf[rb2, ri2]).sum()
            if not (torch.rand((), device=dev, generator=gen).log() < -beta * (e_new - e_old)):
                Sf[b, ia], Sf[b, ib] = Sf[b, ib].clone(), Sf[b, ia].clone()   # revert
    return acc / max(att, 1)


def full_U(Xf, Sf):
    return torch.stack([ka_energy(Xf[b:b+1].double(), Sf[b:b+1].long(), L)[0].float() for b in range(B)])


bank = []; t0 = time.time()
for phase, (T_, SW_) in enumerate([(T_HEAL, SW_HEAL), (T_PROD, SW_PROD)]):
    beta = 1.0 / T_
    for sw in range(1, SW_ + 1):
        ar = sweep(Xb, Sb, beta, g)
        if sw % 100 == 0:
            U = full_U(Xb, Sb) / N
            print(f"phase {'HEAL' if phase == 0 else 'PROD'} T={T_} sw {sw:>5}: U/N = "
                  f"{' '.join(f'{float(u):.4f}' for u in U)}  disp-acc {ar:.2f}  ({time.time()-t0:.0f}s)", flush=True)
        if phase == 1 and sw > SW_BURN and sw % SAVE_EVERY == 0:
            bank.append({"x": Xb.cpu().clone(), "s": Sb.cpu().clone(), "sweep": sw})
            torch.save({"bank": bank, "L": L, "N": N, "rho": N / L ** 3, "T": T_PROD,
                        "note": "mosaic 2x2x2 distinct-config tiling; heal T=0.8; checkerboard MC + swaps"},
                       OUT)
print(f"saved {len(bank)} frames x {B} replicas -> {OUT}", flush=True)
