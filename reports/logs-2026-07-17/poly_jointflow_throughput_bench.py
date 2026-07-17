"""Poly Task J2 gate-speed check: propose_batch throughput at B=64 (GPU) + a latency breakdown
isolating vmap(jacrev(...)) (u-divergence) cost vs the existing position-field EGNN cost, per the
coordinator's request. Saved per CLAUDE.md (harness scripts -> reports/logs-<date>/, committed)."""
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path("/mnt/ssd/GridTransformer")
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from liquid_coupling_flow.poly.joint_flow import JointBlockFlow, sigma_of_u_t, sigma_to_bin_t


def _perturb(f):
    torch.manual_seed(42)
    for p in f.parameters():
        p.data.add_(torch.randn_like(p) * 0.1)
    f._cbf.ce.egnn.pot_model[-1].weight.data.add_(torch.randn_like(f._cbf.ce.egnn.pot_model[-1].weight) * 0.1)
    f._cbf.ce.egnn.pot_model[-1].bias.data.add_(torch.randn_like(f._cbf.ce.egnn.pot_model[-1].bias) * 0.1)
    f.u_head.out_mlp[-1].weight.data.add_(torch.randn_like(f.u_head.out_mlp[-1].weight) * 0.1)
    f.u_head.out_mlp[-1].bias.data.add_(torch.randn_like(f.u_head.out_mlp[-1].bias) * 0.1)


device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device)

k_max, m_env = 16, 32
torch.manual_seed(0)
f = JointBlockFlow(k_max=k_max, m_env=m_env, hidden_nf=64, n_layers=3, u_knn=8).to(device)
_perturb(f)

B, k, m = 64, k_max, m_env
rng = np.random.default_rng(0)
x_old = rng.standard_normal((B, k, 3)).astype(np.float32) * 0.4
u_old = rng.standard_normal((B, k)).astype(np.float32) * 0.4
env_x = rng.standard_normal((B, m, 3)).astype(np.float32) * 2 + 3.0
env_u = rng.standard_normal((B, m)).astype(np.float32) * 0.4
gen = torch.Generator().manual_seed(1)

# ---- full propose_batch throughput ----------------------------------------------------------------
N_WARM, N_ITERS = 1, 3
for _ in range(N_WARM):
    f.propose_batch(x_old, u_old, env_x, env_u, gen)
if device == "cuda":
    torch.cuda.synchronize()
t0 = time.time()
for _ in range(N_ITERS):
    x_new, u_new, logq = f.propose_batch(x_old, u_old, env_x, env_u, gen)
if device == "cuda":
    torch.cuda.synchronize()
dt = (time.time() - t0) / N_ITERS
print(f"propose_batch: {dt:.3f} s/call @ B={B}  ->  {B/dt:.2f} proposals/sec")

# ---- latency breakdown: one _field() call, position-channel only vs +u-channel(vmap+jacrev) --------
x_old_t = torch.as_tensor(x_old, device=device)
u_old_t = torch.as_tensor(u_old, device=device)
env_x_t = torch.as_tensor(env_x, device=device)
env_u_t = torch.as_tensor(env_u, device=device)
movers_x, movers_u, n_real = f._prep_movers(x_old_t, u_old_t)
env_xp, env_up, env_bin, n_env_real = f._prep_env_bin(env_x_t, env_u_t)


def time_it(fn, n=5):
    for _ in range(2):
        fn()
    if device == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(n):
        fn()
    if device == "cuda":
        torch.cuda.synchronize()
    return (time.time() - t0) / n


def pos_only():
    sp_movers = sigma_to_bin_t(sigma_of_u_t(movers_u))
    sp = torch.cat([sp_movers, env_bin], dim=1)
    cloud = torch.cat([movers_x, env_xp], dim=1)
    with torch.no_grad():
        f._vel_div_masked(cloud, 0.3, sp, n_real)


def full_field():
    with torch.no_grad():
        f._field(movers_x, movers_u, env_xp, env_bin, env_up, 0.3, n_real, n_env_real)


t_pos = time_it(pos_only)
t_full = time_it(full_field)
t_u = t_full - t_pos
print(f"one _field() eval @ B={B}: position-only {t_pos*1e3:.2f} ms | full (pos+u/vmap-jacrev) {t_full*1e3:.2f} ms")
print(f"  -> u-channel (vmap+jacrev) overhead: {t_u*1e3:.2f} ms/eval ({100*t_u/t_full:.1f}% of one field eval)")
print(f"  -> RK4 does 24 steps * 4 stages = 96 field evals per direction (propose+logq_of=2 dirs => 192 evals);"
      f" 192 * {t_full*1e3:.2f} ms = {192*t_full:.2f} s predicted vs measured {dt:.2f} s/call")
