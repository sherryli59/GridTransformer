"""Decisive single-site exactness test: freeze all particles but one; the heat-bath kernel's stationary
distribution of x_i MUST equal the exact Boltzmann conditional exp(-beta*U_full(x_i)) (any proposal, if the
MH ratio is right). Compares the kernel histogram <U> to a grid-integrated <U> of the true conditional."""
import torch, numpy as np
from liquid_coupling_flow.ka_heatbath_gate import _env
from liquid_coupling_flow.ka_heatbath import single_site_mh
from liquid_coupling_flow.ka_energy import ka_pair_row

DEV = "cuda" if torch.cuda.is_available() else "cpu"
sc, L, geo, pos, s, HB, N = _env(B=1)
from liquid_coupling_flow.ka_cluster_flow import slot_order
site = 5
beta = 2.0
# fix all others; sweep the kernel on site i only, cage frozen
x0 = pos.clone()
# TRUE conditional <U_row(x_i)> by grid over the box near x_i (exp(-beta*row(x_i)))
xi0 = x0[:, site]                                                  # [1,2]
g = 60
span = 1.6                                                        # +/- around current (local well)
gx = torch.linspace(-span, span, g, device=DEV)
XX, YY = torch.meshgrid(gx, gx, indexing="ij")
cand = xi0[0][None, :] + torch.stack([XX.reshape(-1), YY.reshape(-1)], -1)   # [g*g,2]
cand = torch.remainder(cand, L)
rows = ka_pair_row(x0.expand(g*g, N, 2).contiguous(), s.expand(g*g, N), site, cand, L)  # [g*g]
logw = -beta * rows
w = torch.softmax(logw, 0)
U_true = (w * rows).sum().item()                                  # <U_row> under the true conditional
# KERNEL: many single-site moves on site i, cage frozen (all others fixed). histogram <U_row>.
xi = xi0.clone()
gen = torch.Generator(device=DEV).manual_seed(0)
Us = []
p = x0.clone()
for step in range(4000):
    p2, acc = single_site_mh(HB, p, s, site, sc, L, beta, gen=gen)
    p = p2
    if step > 500:
        Us.append(ka_pair_row(p, s, site, p[:, site], L).item())
U_kernel = float(np.mean(Us))
print(f"site {site}: <U_row> true(grid) {U_true:.4f}  kernel(MH) {U_kernel:.4f}  diff {U_kernel-U_true:+.4f}")
print(f"  (if diff ~0 -> kernel EXACT; if kernel << true -> biased-low = BUG)")
