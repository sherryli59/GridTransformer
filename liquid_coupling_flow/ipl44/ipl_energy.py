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


_SS_CACHE: dict = {}  # keyed by str(device)


def _get_ss(dev):
    """Return a SoftSphere instance resident on *dev*, building and caching it on first use.

    SoftSphere registers sigma_matrix, epsilon_matrix, rcut_sq_matrix, row_indices,
    col_indices as buffers (moved by .to()), but stores mask and mask_local as plain
    tensor attributes that .to() does NOT move.  We move them explicitly so that
    grad_U / hessian_U (which reference mask/mask_local) also work on CUDA.
    """
    key = str(dev)
    if key not in _SS_CACHE:
        m = _energy_module().to(dev)
        if hasattr(m, 'mask') and m.mask is not None:
            m.mask = m.mask.to(dev)
        if hasattr(m, 'mask_local') and m.mask_local is not None:
            m.mask_local = m.mask_local.to(dev)
        _SS_CACHE[key] = m
    return _SS_CACHE[key]


def ipl_energy(positions, species):
    """Total IPL potential U (their SoftSphere.U). positions [B,44,2], species [B,44] int. -> U [B]."""
    dev = positions.device
    ss = _get_ss(dev)
    return ss.U(species.long().to(dev), positions.to(dev))


def ipl_gr(positions, species, L=None, bins=120):
    """Total (species-agnostic) radial distribution function g(r). Returns (r_centers, g), shapes (bins,).

    NOTE: we do NOT reuse their `radial_distribution_function` here — for dim>1 it histograms the
    `gram_torus` DIFFERENCE VECTORS (shape [B, pairs, dim]) component-wise, not the scalar pair
    distances, which produces an unphysical g(r->0) spike (g~54 at r->0 even on equilibrium data whose
    minimum pair distance is ~1.0). We compute the standard g(r): scalar minimum-image distances over
    i<j pairs, histogrammed on [0, L/2], normalized by the ideal-gas shell count
    B*(N-1)/2 * rho * shell_area so g(r)->1 at large r. (Energy reuse is unaffected — that path is
    correct and is what the discard/ESS numbers depend on.) `species` is accepted for API stability but
    unused (total g(r); species-resolved g_AA/g_AB/g_BB is a separate concern)."""
    L = float(L or L_IPL)
    B, N, dim = positions.shape
    iu, ju = torch.triu_indices(N, N, offset=1, device=positions.device)
    edges = torch.linspace(0.0, L / 2, bins + 1, device=positions.device)
    H = torch.zeros(bins, device=positions.device)
    for i in range(0, B, 2048):                                  # chunk: pairwise tensor is O(chunk*N^2)
        p = positions[i:i + 2048]
        d = p[:, iu, :] - p[:, ju, :]
        d = d - L * torch.round(d / L)                           # minimum image
        dist = (d * d).sum(-1).clamp_min(1e-12).sqrt()
        H += torch.histc(dist, bins=bins, min=0.0, max=L / 2)
    shell = math.pi * (edges[1:] ** 2 - edges[:-1] ** 2)         # 2D annulus area per bin
    rho = N / L ** dim
    ideal = B * (N - 1) / 2 * rho * shell                        # ideal-gas i<j pair count per shell
    return 0.5 * (edges[:-1] + edges[1:]), H / ideal


def load_ipl_reference(device="cpu"):
    """Released Grenioux IPL configs. CRITICAL: the Zenodo data stores UNWRAPPED MD coordinates
    (particles drift across periodic images; raw range ~[-29, 35] for L=9.38, ~50% of coords
    outside [0,L)). The energy is wrap-invariant (min-image gram_torus), so this is harmless for
    U/g(r) — but the AR model's curve-order/scaffold pipeline assumes positions in [0,L) and
    silently mis-registers unwrapped coords. We wrap into [0,L) here so every consumer (support
    coverage, training, benchmark) sees the same in-box frame the model is defined on."""
    pos = torch.load(os.path.join(_HERE, "data", "ipl44_T0.1_positions.pt"), weights_only=False).to(device).float()
    sp = torch.load(os.path.join(_HERE, "data", "ipl44_T0.1_species.pt"), weights_only=False).to(device).long()
    return torch.remainder(pos, L_IPL), sp
