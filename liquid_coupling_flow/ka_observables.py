"""Structural observables for 2D KA: partial g(r) (per pair-type), hexatic psi6."""
from __future__ import annotations
import math, torch


def _minimg(x, L):
    diff = x[:, :, None, :] - x[:, None, :, :]
    diff = diff - L * torch.round(diff / L)
    r = torch.sqrt((diff ** 2).sum(-1) + 1e-12)
    return diff, r


def partial_gr(x, s, L, rmax, nbins, pair):
    """g_{ab}(r). x:[B,N,2], s:[N], pair=(a,b). Plateau normalized to ~1."""
    B, N = x.shape[0], x.shape[1]
    a, b = pair
    _, r = _minimg(x, L)                                  # [B,N,N]
    ia = (s == a).nonzero().squeeze(-1)
    ib = (s == b).nonzero().squeeze(-1)
    sub = r[:, ia][:, :, ib]                              # [B,na,nb]
    if a == b:
        m = ~torch.eye(len(ia), dtype=torch.bool, device=x.device)[None]
        d = sub[m.expand_as(sub)].reshape(-1)
        rho_b = (len(ia) - 1) / (L ** 2)
    else:
        d = sub.reshape(-1)
        rho_b = len(ib) / (L ** 2)
    d = d[d < rmax].cpu()
    counts = torch.histc(d, bins=nbins, min=0.0, max=rmax)
    edges = torch.linspace(0.0, rmax, nbins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    shell = math.pi * (edges[1:] ** 2 - edges[:-1] ** 2)
    ideal = B * len(ia) * rho_b * shell                  # expected counts per a-ref over B frames
    return centers.numpy(), (counts / ideal).numpy()


def psi6(x, L, cutoff=1.5):
    """Mean over particles of |(1/n) sum_j e^{i6 theta_ij}| using neighbours within cutoff."""
    diff, r = _minimg(x, L)                               # [B,N,N,2], [B,N,N]
    ang = torch.atan2(diff[..., 1], diff[..., 0])        # [B,N,N]
    nb = (r < cutoff) & (r > 1e-6)
    w = nb.float()
    z = (torch.exp(1j * 6 * ang) * w).sum(-1)            # [B,N]
    n = w.sum(-1).clamp_min(1)
    return (z.abs() / n).mean(-1)                         # [B]
