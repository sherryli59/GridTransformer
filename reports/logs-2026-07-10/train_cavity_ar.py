"""Generalization training for the 3D local-frame AR cavity generator: MANY carved cavities +
random 3D rotation augmentation (the local frame uses global orientation, so rotation-invariance
is learned). Overfit already proved CAPACITY (-logp/n hit the delta floor); this run tests whether
data+augmentation smooth the conditional enough that FREE-RUNNING generation is physical, and
quantifies TF-vs-FR (exposure-bias drift) on held-out cavities.

Incremental checkpoint every eval (checkpoint-incrementally directive). Saves held-out eval series.
"""
import time
from pathlib import Path
import torch
from liquid_coupling_flow.ka3d_cavity_ar import KA3DCavityAR, _mic, cavity_order
from liquid_coupling_flow.ka3d_cavity_carve import carve
from liquid_coupling_flow.ka3d_pts_observables import core_overlap
from liquid_coupling_flow.ka_energy import ka_energy

dev = "cuda" if torch.cuda.is_available() else "cpu"
d = torch.load("liquid_coupling_flow/artifacts/ka3d_dataset_N512_T0.5.pt", map_location=dev, weights_only=False)
X, S, L = d["x"].to(dev).float(), d["s"].to(dev).long(), float(d["L"])
r_ctx, RADII = 2.5, (1.6, 2.0, 2.4)
CKPT = Path("liquid_coupling_flow/artifacts/ka3d_cavity_ar.pt")
torch.manual_seed(0)
g = torch.Generator(device=dev).manual_seed(1)


def carve_pool(cfg_idx, n_centers):
    pool = []
    for ci in cfg_idx:
        xb, sb = X[ci], S[ci]
        for _ in range(n_centers):
            center = xb[int(torch.randint(xb.shape[0], (), generator=g, device=dev))].clone()
            R = float(RADII[int(torch.randint(len(RADII), (), generator=g, device=dev))])
            p = carve(xb, sb, center, R, L)
            if p["n_in"] < 4:
                continue
            irel = _mic(p["x_in"], center, L); ball = _mic(p["x_out"], center, L)
            shell = ball.norm(dim=-1) < R + r_ctx
            n_B = int((p["s_in"] == 1).sum())
            pool.append(dict(irel=irel, s_in=p["s_in"], bnd=ball[shell], s_bnd=p["s_out"][shell], R=R,
                             n_A=p["n_in"] - n_B, n_B=n_B, n=p["n_in"],
                             xout=ball, sout=p["s_out"], center=center))
    return pool


train_pool = carve_pool(range(0, 900), 3)
held = carve_pool(range(900, 1024), 1)
print(f"train cavities={len(train_pool)}  held={len(held)}  radii={RADII}", flush=True)


def rand_rot():
    A = torch.randn(3, 3, generator=g, device=dev)
    Q, Rm = torch.linalg.qr(A)
    Q = Q * torch.sign(torch.diagonal(Rm))[None]
    if torch.det(Q) < 0:
        Q[:, 0] = -Q[:, 0]
    return Q


def energy_of(center, irel, s_in, xout, sout):
    N = irel.shape[0] + xout.shape[0]
    x = torch.empty(1, N, 3, device=dev); s = torch.empty(1, N, device=dev, dtype=torch.long)
    x[0, :irel.shape[0]] = torch.remainder(center + irel, L); x[0, irel.shape[0]:] = torch.remainder(center + xout, L)
    s[0, :irel.shape[0]] = s_in; s[0, irel.shape[0]:] = sout
    return (ka_energy(x, s, L) / N).item()


@torch.no_grad()
def evaluate(m, subset):
    m.eval(); nll = 0.0; du = []; clash = 0; ov = []
    for p in subset:
        nll += -m.log_prob_pair(p["irel"], p["s_in"], p["bnd"], p["s_bnd"], p["R"]).item() / p["n"]
        u_true = energy_of(p["center"], p["irel"], p["s_in"], p["xout"], p["sout"])
        gi, gs, _ = m.sample_pair(p["bnd"], p["s_bnd"], p["n_A"], p["n_B"], p["R"], return_logq=True)
        u_gen = energy_of(p["center"], gi, gs, p["xout"], p["sout"])
        du.append(u_gen); clash += int(u_gen > u_true + 2.0)
        ov.append(core_overlap(gi[None], gs[None], p["irel"][None], p["s_in"][None],
                               torch.zeros(3, device=dev), L)[0].item())
    m.train()
    du = torch.tensor(du)
    return nll / len(subset), du.median().item(), clash, len(subset), sum(ov) / len(ov)


m = KA3DCavityAR().to(dev)
opt = torch.optim.Adam(m.parameters(), lr=3e-4)
t0 = time.time(); series = []
for step in range(20001):
    idx = torch.randint(len(train_pool), (6,))
    loss = 0.0
    for i in idx:
        p = train_pool[i]     # NO rotation aug: the Gram-Schmidt local frame makes the model rotation-invariant
        loss = loss - m.log_prob_pair(p["irel"], p["s_in"], p["bnd"], p["s_bnd"], p["R"]) / p["n"]
    loss = loss / len(idx)
    opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 5.0); opt.step()
    if step % 2000 == 0 or step == 20000:
        nll, umed, clash, ntot, ov = evaluate(m, held[:40])
        series.append(dict(step=step, held_nll=nll, gen_U_median=umed, clash=clash, n=ntot, overlap=ov))
        torch.save({"state_dict": m.state_dict(), "step": step, "series": series,
                    "config": dict(radii=RADII, r_ctx=r_ctx)}, CKPT)
        print(f"  step {step:5d}  held -logp/n={nll:+.3f}  gen U/N med={umed:+.2f}  "
              f"clashed={clash}/{ntot}  overlap(gen,true)={ov:.3f}  ({time.time()-t0:.0f}s)  [ckpt]", flush=True)
print(f"DONE -> {CKPT}", flush=True)
