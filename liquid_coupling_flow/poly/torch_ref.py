"""Independent f64 torch reference for the exactness gate. Must NOT import poly.model."""
import torch

XC = 1.25
C0 = -28.0 / XC ** 12
C2 = 48.0 / XC ** 14
C4 = -21.0 / XC ** 16


def total_U_torch(x, sig, L):
    x = torch.as_tensor(x, dtype=torch.float64)
    s = torch.as_tensor(sig, dtype=torch.float64)
    d = x[:, None, :] - x[None, :, :]
    d -= L * torch.round(d / L)
    r2 = (d ** 2).sum(-1)
    sij = 0.5 * (s[:, None] + s[None, :]) * (1.0 - 0.2 * (s[:, None] - s[None, :]).abs())
    n = x.shape[0]
    iu = torch.triu_indices(n, n, offset=1)
    r2u = r2[iu[0], iu[1]]
    su = sij[iu[0], iu[1]]
    inside = r2u < (XC * su) ** 2
    inv6 = (su ** 2 / r2u.clamp_min(1e-18)) ** 6
    x2 = r2u / su ** 2
    v = inv6 + C0 + C2 * x2 + C4 * x2 * x2
    return float(v[inside].sum())
