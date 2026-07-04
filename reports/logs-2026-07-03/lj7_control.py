"""POSITIVE CONTROL: can the production EGNN-CNF core (same backend, same OT-CFM loop, same RK4 sampling)
learn an isolated 2D LJ7 Boltzmann distribution with clean hard cores? Literature (DW4/LJ13 EGNN flows) says yes.
FAIL here -> bug in the core pipeline. PASS -> core fine; the glass-task residual is task/capacity/training.
Mirrors production exactly: ConditionalEGNN (same EGNN_dynamics), E.ot_assign, same interpolant/loss/optimizer,
same RK4 velocity-only sampling, float32 training. Only the scaffold/cage plumbing is bypassed (n_cage=0, P=7)."""
import math, time, torch
from liquid_coupling_flow import ka_cluster_egnn as E
from liquid_coupling_flow.ka_energy import ka_energy
from liquid_coupling_flow.ka_gridformer import _wrap_pm

DEV = "cuda"; L = 9.128709291752768; T = 0.2; BETA = 1.0 / T; k = 7
torch.manual_seed(0)

# ---------------- 1) exact Boltzmann data: vectorized single-particle Metropolis, 512 chains ----------------
nch = 512
s = torch.zeros(k, dtype=torch.long, device=DEV)                     # all species A (sigma=1, eps=1)
a0 = 2 ** (1 / 6)                                                     # LJ pair minimum
pts = [[0.0, 0.0]] + [[a0 * math.cos(2 * math.pi * i / 6), a0 * math.sin(2 * math.pi * i / 6)] for i in range(6)]
x = torch.tensor(pts, device=DEV, dtype=torch.float64)[None].repeat(nch, 1, 1) + L / 2
x = torch.remainder(x + 0.03 * torch.randn_like(x), L)
U = ka_energy(x, s, L)
sig_mv = 0.07; n_acc = 0; n_try = 0
frames = []
t0 = time.time()
for sweep in range(4000):
    for i in range(k):
        prop = x.clone()
        prop[:, i] = torch.remainder(prop[:, i] + sig_mv * torch.randn(nch, 2, device=DEV, dtype=x.dtype), L)
        Up = ka_energy(prop, s, L)
        acc = torch.rand(nch, device=DEV) < torch.exp(-BETA * (Up - U))
        x = torch.where(acc[:, None, None], prop, x); U = torch.where(acc, Up, U)
        n_acc += int(acc.sum()); n_try += nch
    if sweep >= 2000 and sweep % 10 == 0:
        frames.append(x.clone())
data = torch.cat(frames, 0).float()                                   # [~102k, 7, 2]
d = _wrap_pm(data[:, :, None] - data[:, None], L).norm(dim=-1) + torch.eye(k, device=DEV)[None] * 1e3
minr_data = d.min(-1).values.reshape(-1)
print(f"DATA: {data.shape[0]} configs  U/N {float(U.mean())/k:.3f}  MH-acc {100*n_acc/n_try:.0f}%  "
      f"min-r mean {minr_data.mean():.3f}  P(<0.85) {(minr_data<0.85).float().mean()*100:.2f}%  "
      f"P(<0.8) {(minr_data<0.8).float().mean()*100:.2f}%  ({time.time()-t0:.0f}s)", flush=True)

def centroid(xx):                                                     # min-image centroid about particle 0
    return torch.remainder(xx[:, 0] + _wrap_pm(xx - xx[:, :1], L).mean(1), L)

# sigma_b from data, exactly like compute_sigma_b (spread about the cluster centroid)
c_all = centroid(data[:2000])
sigma_b = float(_wrap_pm(data[:2000] - c_all[:, None], L).reshape(-1, 2).std().item())
print(f"sigma_b = {sigma_b:.3f}", flush=True)

# ---------------- 2) train: IDENTICAL core (backend, OT, interpolant, loss, optimizer) ----------------
ce = E.ConditionalEGNN(n_cage=0, k=k, r_c=3.0, L=L, hidden_nf=64, n_layers=4, max_neighbors=None).to(DEV)
with torch.no_grad():
    ce.egnn.pot_model[-1].weight.zero_(); ce.egnn.pot_model[-1].bias.zero_()
opt = torch.optim.AdamW(ce.parameters(), lr=3e-4, weight_decay=1e-4)
Bsz = 256; steps = 15000
spb = torch.zeros(Bsz, k, dtype=torch.long, device=DEV)
t0 = time.time()
for step in range(steps):
    idx = torch.randint(0, data.shape[0], (Bsz,), device=DEV)
    x1 = data[idx]
    c = centroid(x1)
    z = E.sample_base(c, k, sigma_b)
    perm = E.ot_assign(z, x1, spb, L)
    x1p = torch.gather(x1, 1, perm[..., None].expand(-1, -1, 2))
    target = _wrap_pm(x1p - z, L)
    t = torch.rand(Bsz, device=DEV)
    xt = torch.remainder(z + t[:, None, None] * target, L)
    v = ce.egnn.forward(t, xt, spb)
    loss = ((v - target) ** 2).sum(-1).mean()
    opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(ce.parameters(), 5.0); opt.step()
    if step % 1000 == 0:
        print(f"  step {step:5d} fm-loss {loss.item():.4f} {time.time()-t0:.0f}s", flush=True)


# ---------------- 3) sample: same RK4 velocity-only, n_steps sweep ----------------
spg = torch.zeros(1024, k, dtype=torch.long, device=DEV)
with torch.no_grad():
    for nst in (8, 32, 64):
        gen_minr = []
        for rep in range(8):
            idx = torch.randint(0, data.shape[0], (1024,), device=DEV)
            c = centroid(data[idx])
            cl = E.sample_base(c, k, sigma_b); dt = 1.0 / nst
            for i in range(nst):
                tt = i * dt
                v1 = ce.egnn.forward(torch.full((1024,), tt, device=DEV), cl, spg)
                v2 = ce.egnn.forward(torch.full((1024,), tt + 0.5 * dt, device=DEV), torch.remainder(cl + 0.5 * dt * v1, L), spg)
                v3 = ce.egnn.forward(torch.full((1024,), tt + 0.5 * dt, device=DEV), torch.remainder(cl + 0.5 * dt * v2, L), spg)
                v4 = ce.egnn.forward(torch.full((1024,), tt + dt, device=DEV), torch.remainder(cl + dt * v3, L), spg)
                cl = torch.remainder(cl + (dt / 6.0) * (v1 + 2 * v2 + 2 * v3 + v4), L)
            dg = _wrap_pm(cl[:, :, None] - cl[:, None], L).norm(dim=-1) + torch.eye(k, device=DEV)[None] * 1e3
            gen_minr.append(dg.min(-1).values.reshape(-1))
        g = torch.cat(gen_minr)
        print(f"GEN n_steps {nst:2d}: min-r mean {g.mean():.3f}  P(<0.85) {(g<0.85).float().mean()*100:.2f}%  "
              f"P(<0.8) {(g<0.8).float().mean()*100:.2f}%   [DATA: mean {minr_data.mean():.3f}, "
              f"P(<0.85) {(minr_data<0.85).float().mean()*100:.2f}%]", flush=True)
torch.save({"state_dict": ce.state_dict(), "sigma_b": sigma_b}, f"{E.ART}/lj7_control.pt")
print("saved", f"{E.ART}/lj7_control.pt", flush=True)
