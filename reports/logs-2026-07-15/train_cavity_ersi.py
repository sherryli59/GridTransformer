"""CAVITY-CONDITIONED eRSI: uniform-in-sphere base -> equivariant flow -> equilibrium cavity interior,
conditioned on the frozen boundary. The mW-eRSI pattern (UniformParticles base -> traceable EGNN flow)
transplanted to the KA cavity, at rho=1.2 T=0.5 where our validated truth lives.

WHY (measured, 2026-07-15):
  - PT shrinkage: round-trip starved (TRIPS=0); needs 5e7 sweeps/cavity  -> unaffordable [[pt-shrinkage-infeasible]]
  - FL weights: untrusted (discovery completeness + tether F)             -> user rejected
  - random restart: p_success 0.04 @R=2.0, 0.00 @R=2.3                    -> dead (relax_sweep.out)
  - AR base: generates +36..+1e11 per-particle CLASH energies             -> relax_race.out
A flow maps a SMOOTH base onto the data manifold instead of placing particles sequentially into clashes,
so it is the natural fix for the last failure -- and it carries an EXACT log-likelihood (analytic
divergence) => exact-MH basin-crossing moves, no FL weights anywhere.

DIFFERS from reports/logs-2026-07-12/train_cavity_egnn_flow.py (which this reuses machinery from):
  - that trains a K=8 BLOCK CORRECTOR with an AR-block base at rho=1.15; this trains a FULL-CAVITY
    SAMPLER with a uniform-in-sphere base at rho=1.2/T=0.5.
  - movers = ALL n interior particles (not a K-subset); cage = boundary shell only.

FIXED-SIZE SCHEME (CavityCondEGNN bakes P=k+n_cage at construction; cavities vary n~36-44 @R=2.0):
  - k = K_MAX movers. Unused slots padded with INERT dummies at (50+5j, 0, 0) -- SPREAD, not stacked
    (coincident dummies would be each other's r=0 nearest neighbours -> NaN). Mutual gap 5 > r_c=2.5 and
    ~45 away from every real particle (which live within ~R+RCTX of the origin) => never enter any real
    particle's k-NN neighbourhood. Dummies are identical in x0 AND x1 => target_v = 0 => identity map,
    zero velocity, zero divergence contribution. Exactness untouched.
  - n_cage = N_CAGE nearest boundary particles to the cavity centre (origin), same dummy padding.
OT: per-species Hungarian (fm_data._match_species) between base and target -- without it, regressing a
velocity from uniform noise to an equilibrium packing is unlearnable label noise.
TRAINED == DEPLOYED velocity: loss regresses flow.ce.vel_div (the pairwise-only field the ODE integrates),
never .egnn.forward (which adds a com_pot term vel_div omits). See ka3d_cavity_egnn.py CAVEAT.
Usage: train_cavity_ersi.py [--steps N] [--smoke]"""
from __future__ import annotations
import argparse, math, sys, time
from pathlib import Path
import numpy as np
import torch

REPO = Path("/mnt/ssd/GridTransformer")
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "reports/logs-2026-07-12"))
from liquid_coupling_flow.ka3d_cavity_egnn import CavityBlockFlow
from liquid_coupling_flow.ka3d_cavity_carve import carve
from liquid_coupling_flow.ka3d_cavity_ar import _mic
from liquid_coupling_flow.ka_energy import ka_energy
from fm_data import _match_species

p = argparse.ArgumentParser()
p.add_argument("--steps", type=int, default=30000)
p.add_argument("--smoke", action="store_true")
p.add_argument("--R", type=float, default=2.0)
p.add_argument("--K_MAX", type=int, default=52)
p.add_argument("--N_CAGE", type=int, default=64)
p.add_argument("--hidden", type=int, default=128)
p.add_argument("--layers", type=int, default=5)
p.add_argument("--ncav", type=int, default=4)      # cavities per step
p.add_argument("--M", type=int, default=4)         # base draws per cavity
p.add_argument("--lr", type=float, default=3e-4)
p.add_argument("--out", type=str, default="ka3d_cavity_ersi_rho12")
a = p.parse_args()
if a.smoke:
    a.steps = 30
dev = "cuda"
RCTX = 2.5
DUMMY0, DUMMY_D = 50.0, 5.0
OUT = REPO / f"liquid_coupling_flow/artifacts/{a.out}.pt"
BEST = REPO / f"liquid_coupling_flow/artifacts/{a.out}_best.pt"

D = torch.load(REPO / "liquid_coupling_flow/artifacts/ka3d_dataset_N512_T0.5.pt", map_location="cpu", weights_only=False)
Xds, Sds, L = D["x"].float(), D["s"].long(), float(D["L"])
NCFG = Xds.shape[0]
n_train = int(NCFG * 0.9)                     # config-level split (zero leakage)
print(f"cavity eRSI: rho=1.2 T=0.5 R={a.R} | K_MAX={a.K_MAX} N_CAGE={a.N_CAGE} hidden={a.hidden} layers={a.layers} "
      f"| train cfgs {n_train}/{NCFG} | steps {a.steps}", flush=True)


def dummies(m, device):
    d = torch.zeros(m, 3, device=device)
    d[:, 0] = DUMMY0 + DUMMY_D * torch.arange(m, device=device, dtype=torch.float32)
    return d


def carve_full(gen, train=True):
    """Carve one cavity: movers = ALL interior, cage = boundary shell. Returns padded fixed-size tensors."""
    for _ in range(60):
        lo, hi = (0, n_train) if train else (n_train, NCFG)
        ci = int(torch.randint(lo, hi, (1,), generator=gen))
        c = torch.rand(3, generator=gen) * L
        pr = carve(Xds[ci], Sds[ci], c, a.R, L)
        n = pr["n_in"]
        if n < 14 or n > a.K_MAX:
            continue
        xin = _mic(pr["x_in"], c, L); sin = pr["s_in"].long()
        xout = _mic(pr["x_out"], c, L); bm = xout.norm(dim=-1) < (a.R + RCTX)
        bx, bs = xout[bm], pr["s_out"][bm].long()
        if bx.shape[0] < 8:
            continue
        # cage: N_CAGE nearest to origin, dummy-padded
        idx = bx.norm(dim=-1).argsort()[: a.N_CAGE]
        cage_x = bx[idx]; sp_cage = bs[idx]
        nc = cage_x.shape[0]
        if nc < a.N_CAGE:
            cage_x = torch.cat([cage_x, dummies(a.N_CAGE - nc, cage_x.device)])
            sp_cage = torch.cat([sp_cage, torch.zeros(a.N_CAGE - nc, dtype=torch.long)])
        return {"x1": xin, "sp": sin, "n": n, "cage_x": cage_x, "sp_cage": sp_cage}
    return None


def build_rows(cav, gen, M):
    """uniform-in-sphere base -> per-species Hungarian match to data -> FM interpolant. Padded to K_MAX."""
    n = cav["n"]; K = a.K_MAX
    x1r, spr = cav["x1"], cav["sp"]
    x0 = torch.zeros(M, K, 3); x1 = torch.zeros(M, K, 3)
    spb = torch.zeros(M, K, dtype=torch.long)
    real = torch.zeros(M, K, dtype=torch.bool); real[:, :n] = True
    dpad = dummies(K - n, torch.device("cpu")) if K > n else None
    for mi in range(M):
        u = torch.randn(n, 3, generator=gen); u = u / u.norm(dim=-1, keepdim=True)
        rr = a.R * torch.rand(n, 1, generator=gen) ** (1.0 / 3.0)
        x0n = u * rr                                          # uniform in sphere(R)
        perm = _match_species(x0n, spr, x1r, spr)             # per-species Hungarian (base->data)
        x0[mi, :n] = x0n; x1[mi, :n] = x1r[perm]; spb[mi, :n] = spr
        if dpad is not None:
            x0[mi, n:] = dpad; x1[mi, n:] = dpad              # identity on dummies -> target_v = 0
    t = torch.rand(M, generator=gen)
    x_t = (1 - t)[:, None, None] * x0 + t[:, None, None] * x1
    return {"x_t": x_t, "t": t, "target_v": x1 - x0, "sp_block": spb, "real": real,
            "cage_x": cav["cage_x"][None].expand(M, -1, -1).clone(),
            "sp_cage": cav["sp_cage"][None].expand(M, -1).clone()}


def rand_rot(B, gen):
    A = torch.randn(B, 3, 3, generator=gen)
    Q, R_ = torch.linalg.qr(A)
    d = torch.sign(torch.diagonal(R_, dim1=1, dim2=2)); Q = Q * d[:, None, :]
    return Q


flow = CavityBlockFlow(n_cage=a.N_CAGE, k=a.K_MAX, r_c=2.5, hidden_nf=a.hidden, n_layers=a.layers,
                       n_species=2, max_neighbors=16).to(dev)
opt = torch.optim.Adam(flow.parameters(), lr=a.lr)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.steps)
gen = torch.Generator().manual_seed(0)
best = float("inf"); hist = []
t0 = time.time()
for step in range(1, a.steps + 1):
    rows = []
    for _ in range(a.ncav):
        cav = carve_full(gen, train=True)
        if cav is None:
            continue
        rows.append(build_rows(cav, gen, a.M))
    if not rows:
        continue
    b = {k: torch.cat([r[k] for r in rows]) for k in rows[0]}
    B = b["x_t"].shape[0]
    Q = rand_rot(B, gen)                                       # rotate movers+cage TOGETHER (equivariance aug)
    xt = torch.einsum("bij,bkj->bki", Q, b["x_t"]).to(dev)
    tv = torch.einsum("bij,bkj->bki", Q, b["target_v"]).to(dev)
    cg = torch.einsum("bij,bkj->bki", Q, b["cage_x"]).to(dev)
    sp = torch.cat([b["sp_block"], b["sp_cage"]], 1).to(dev)
    tt = b["t"].to(dev); real = b["real"].to(dev)
    cloud = torch.cat([xt, cg], 1)
    vel, _ = flow.ce.vel_div(cloud, tt, sp, a.K_MAX)            # the EXACT field the ODE integrates
    loss = (((vel - tv) ** 2).sum(-1) * real).sum() / real.sum().clamp(min=1)
    opt.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_(flow.parameters(), 5.0)
    opt.step(); sched.step()
    hist.append(float(loss))
    if step % 250 == 0 or step == a.steps:
        m = float(np.mean(hist[-250:]))
        torch.save({"state_dict": flow.state_dict(), "args": vars(a), "step": step, "fm": m}, OUT)
        if m < best:
            best = m
            torch.save({"state_dict": flow.state_dict(), "args": vars(a), "step": step, "fm": m}, BEST)
        print(f"  step {step:>6}: FM loss {m:.4f} (best {best:.4f}) lr {sched.get_last_lr()[0]:.2e} "
              f"({time.time()-t0:.0f}s)", flush=True)
print(f"done. best FM {best:.4f} -> {BEST}", flush=True)
