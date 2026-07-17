import numpy as np
import torch
from liquid_coupling_flow.poly.joint_flow import JointBlockFlow, BASE_W

def _perturb(f):
    torch.manual_seed(42)
    for p in f.parameters():
        p.data.add_(torch.randn_like(p) * 0.1)
    f._cbf.ce.egnn.pot_model[-1].weight.data.add_(torch.randn_like(f._cbf.ce.egnn.pot_model[-1].weight) * 0.1)
    f._cbf.ce.egnn.pot_model[-1].bias.data.add_(torch.randn_like(f._cbf.ce.egnn.pot_model[-1].bias) * 0.1)
    f.u_head.out_mlp[-1].weight.data.add_(torch.randn_like(f.u_head.out_mlp[-1].weight) * 0.1)
    f.u_head.out_mlp[-1].bias.data.add_(torch.randn_like(f.u_head.out_mlp[-1].bias) * 0.1)

def _rand_block(rng, k, m):
    x_old = rng.standard_normal((k, 3)) * 0.4
    u_old = rng.standard_normal(k) * 0.4
    env_x = rng.standard_normal((m, 3)) * 2 + 3.0
    env_u = rng.standard_normal(m) * 0.4
    return x_old, u_old, env_x, env_u

k, m = 5, 14
rng = np.random.default_rng(7)
x_old, u_old, env_x, env_u = _rand_block(rng, k, m)

torch.manual_seed(3)
f_a = JointBlockFlow(k_max=k, m_env=16, hidden_nf=16, n_layers=2, u_knn=3)
_perturb(f_a)
torch.manual_seed(3)
f_b = JointBlockFlow(k_max=k + 4, m_env=16, hidden_nf=16, n_layers=2, u_knn=3)
_perturb(f_b)
f_a = f_a.double(); f_b = f_b.double()

# reuse a fixed base sample (bypass gen/float32 path): same noise for both
rng2 = np.random.default_rng(123)
noise_x = rng2.standard_normal((1,k,3))
noise_u = rng2.standard_normal((1,k))
x_old_t = torch.tensor(x_old[None], dtype=torch.float64)
u_old_t = torch.tensor(u_old[None], dtype=torch.float64)
z_x = x_old_t + BASE_W*torch.tensor(noise_x, dtype=torch.float64)
z_u = u_old_t + f_a.base_w_u*torch.tensor(noise_u, dtype=torch.float64)
env_x_t = torch.tensor(env_x[None], dtype=torch.float64)
env_u_t = torch.tensor(env_u[None], dtype=torch.float64)

def run(f):
    movers_x, movers_u, n_real = f._prep_movers(z_x, z_u)
    env_xp, env_up, env_bin, n_env_real = f._prep_env_bin(env_x_t, env_u_t)
    x1,u1,_ = f._integrate(movers_x, movers_u, env_xp, env_bin, env_up, reverse=False, n_real=n_real, n_env_real=n_env_real)
    x_new = x1[:,:k].detach().numpy()
    u_new = u1[:,:k].detach().numpy()
    movers_x2, movers_u2, n_real2 = f._prep_movers(x1[:,:k].clone(), u1[:,:k].clone())
    env_xp2, env_up2, env_bin2, n_env_real2 = f._prep_env_bin(env_x_t, env_u_t)
    zx, zu, path_sum = f._integrate(movers_x2, movers_u2, env_xp2, env_bin2, env_up2, reverse=True, n_real=n_real2, n_env_real=n_env_real2)
    zx_real = zx[:,:k]; zu_real = zu[:,:k]
    dx = zx_real - x_old_t; du = zu_real - u_old_t
    import math
    from liquid_coupling_flow.poly.joint_flow import _LOG_2PI
    log_nx = (-0.5*(dx*dx).sum(dim=(1,2))/(BASE_W**2) - k*3*0.5*(_LOG_2PI+2.0*math.log(BASE_W)))
    log_nu = (-0.5*(du*du).sum(dim=1)/(f.base_w_u**2) - k*1*0.5*(_LOG_2PI+2.0*math.log(f.base_w_u)))
    logq = (log_nx+log_nu+path_sum).detach().numpy()
    return x_new, u_new, logq

xa,ua,qa = run(f_a)
xb,ub,qb = run(f_b)
print("logq_a", qa, "logq_b", qb, "diff", abs(qa[0]-qb[0]))
print("x diff", np.abs(xa-xb).max())
print("u diff", np.abs(ua-ub).max())
