"""Thin adapters reusing Grenioux et al.'s released IPL energy + g(r) (vendored in ./learndiffeq, unmodified).
Do NOT reimplement the energy. N=44 IPL: 2D, 50:50, r^-12 BHHP, sigma=[[1,1.2],[1.2,1.4]], eps=1, rcut=2.5sigma,
rho=0.5 -> L=sqrt(88), T=0.1."""
from __future__ import annotations
import os, sys, math, torch

_HERE = os.path.dirname(__file__)
_REPO = os.path.join(_HERE, "learndiffeq")
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)                              # import their package
N_IPL = 44
L_IPL = math.sqrt(N_IPL / 0.5)                             # rho = N/L^2 = 0.5
BETA_IPL = 10.0                                            # T = 0.1


def ipl_box():
    return N_IPL, L_IPL


def _energy_module():
    """Their SoftSphere instance (default sigma/eps/rcut match the IPL spec)."""
    from learndiffeq.particles.distributions.soft_spheres import SoftSphere
    return SoftSphere(n_particles=N_IPL, dim_phys=2, L=L_IPL)


_SS = None
def ipl_energy(positions, species):
    """Total IPL potential U (their SoftSphere.U). positions [B,44,2], species [B,44] int. -> U [B]."""
    global _SS
    if _SS is None:
        _SS = _energy_module()
    dev = positions.device
    return _SS.to(dev).U(species.long().to(dev), positions)


def ipl_gr(positions, species, L=None, bins=100):
    """Their radial distribution function. Returns (r_centers, g) where each has shape (bins,).
    radial_distribution_function returns gr of shape (num_bins, 2): gr[:,0]=r, gr[:,1]=g(r)."""
    from learndiffeq.particles.callbacks.utils import radial_distribution_function
    gr = radial_distribution_function(positions, L or L_IPL, num_bins=bins)
    return gr[:, 0], gr[:, 1]


def load_ipl_reference(device="cpu"):
    pos = torch.load(os.path.join(_HERE, "data", "ipl44_T0.1_positions.pt"), weights_only=False).to(device).float()
    sp = torch.load(os.path.join(_HERE, "data", "ipl44_T0.1_species.pt"), weights_only=False).to(device).long()
    return pos, sp
