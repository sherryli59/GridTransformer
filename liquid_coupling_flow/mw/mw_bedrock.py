"""Grid-quadrature bedrock <U> at N=2 (two-body only) and N=3 (the only analytic 3-body exercise).
Midpoint rule on the periodic box (periodic integrand => midpoint converges fast); particle 1 fixed
at the origin (translation invariance). Halving error = |value(n) - value(n//2)|."""
from __future__ import annotations
import os, torch
from liquid_coupling_flow.mw.mw_energy import mw_energy

ART = os.path.join(os.path.dirname(__file__), "artifacts")
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def _grid(L, n, device):
    g = (torch.arange(n, device=device, dtype=torch.float64) + 0.5) * (L / n)
    return torch.stack(torch.meshgrid(g, g, g, indexing="ij"), -1).reshape(-1, 3)   # [n^3,3]


def quad_N2(L, beta, n):
    def run(nn):
        p2 = _grid(L, nn, DEV)
        x = torch.zeros(p2.shape[0], 2, 3, dtype=torch.float64, device=DEV)
        x[:, 1] = p2
        U = mw_energy(x, L)
        w = torch.softmax(-beta * U, 0)
        return float((w * U).sum())
    v, vh = run(n), run(n // 2)
    return v, abs(v - vh)


def quad_N3(L, beta, n, chunk=2_000_000, two_body_only=False):
    def run(nn):
        p = _grid(L, nn, DEV)                              # [M,3]
        M = p.shape[0]
        block = max(1, chunk // M)                          # keep block*M <= chunk configs/energy call
        num = torch.tensor(0.0, dtype=torch.float64); den = torch.tensor(0.0, dtype=torch.float64)
        m_shift = None                                      # streaming logsumexp-style stabilization
        for i in range(0, M, block):
            p2 = p[i:i + block]                             # [m,3] second particle block
            m = p2.shape[0]
            x = torch.zeros(m * M, 3, 3, dtype=torch.float64, device=DEV)
            x[:, 1] = p2.repeat_interleave(M, 0)
            x[:, 2] = p.repeat(m, 1)
            if two_body_only:
                from liquid_coupling_flow.mw.mw_energy import _pair, _phi2, A_CUT
                d, r = _pair(x, L)
                eye = torch.eye(3, dtype=torch.bool, device=x.device)
                U = _phi2(r.masked_fill(eye[None], A_CUT + 1.0)).sum((1, 2)) / 2
            else:
                U = mw_energy(x, L)
            if m_shift is None:                             # fix the shift on the FIRST batch only;
                m_shift = float(U.min())                     # re-shifting per batch corrupts the weights
            w = torch.exp(-beta * (U - m_shift))
            num += (w * U).sum().cpu(); den += w.sum().cpu()
        return float(num / den)
    v, vh = run(n), run(n // 2)
    return v, abs(v - vh)


if __name__ == "__main__":
    n2 = quad_N2(4.0, 2.0, 96); n3 = quad_N3(4.0, 2.0, 20)
    print(f"BEDROCK N=2: <U> {n2[0]:.6f} (halving err {n2[1]:.2e})", flush=True)
    print(f"BEDROCK N=3: <U> {n3[0]:.6f} (halving err {n3[1]:.2e})", flush=True)
    os.makedirs(ART, exist_ok=True)
    torch.save({"N2": n2, "N3": n3, "L": 4.0, "beta": 2.0}, os.path.join(ART, "mw_bedrock.pt"))
