"""BCY core overlap q_c (Eqs 5-8): Gaussian-NN w(z)=exp(-(z/0.2)^2), field integrated over
rc=0.5 core ball at cavity center, symmetrised. Verbatim port of bcy_qc from
reports/logs-2026-07-15/bcy_gpts_v2.py (constants B_OV=0.2, RC_CORE=0.5, P_MC=1500, K_NN=6),
generalized to accept numpy arrays or tensors. CPU only."""
import torch

B_OV = 0.2
RC_CORE = 0.5
P_MC = 1500
K_NN = 6


def bcy_qc(X, Y, gen):
    """BCY core overlap (Eqs 5-8) between two centered interior configs, symmetrised. CPU."""
    X = torch.as_tensor(X, dtype=torch.float32).cpu()
    Y = torch.as_tensor(Y, dtype=torch.float32).cpu()

    def field_core(vpos, vq):
        u = torch.randn(P_MC, 3, generator=gen); u = u / u.norm(dim=-1, keepdim=True)
        r = RC_CORE * torch.rand(P_MC, generator=gen) ** (1.0 / 3.0); pmc = u * r[:, None]
        d2 = ((pmc[:, None] - vpos[None]) ** 2).sum(-1); w = 1.0 / (d2 + 1e-6)
        kk = min(K_NN, vpos.shape[0]); topw, idx = w.topk(kk, dim=1)
        return float(((topw * vq[idx]).sum(1) / topw.sum(1)).mean())

    qX = torch.exp(-(torch.cdist(X, Y).min(1).values / B_OV) ** 2)
    qY = torch.exp(-(torch.cdist(Y, X).min(1).values / B_OV) ** 2)
    return 0.5 * (field_core(X, qX) + field_core(Y, qY))
