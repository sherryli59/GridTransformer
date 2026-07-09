import math, torch, numpy as np, pytest
from liquid_coupling_flow.mw.mw_energy import (mw_energy, mw_energy_chunked, du_move,
                                               A_SW, B_SW, A_CUT, GAMMA, LAMBDA3, COS0)

def _phi2_py(r):
    if r >= A_CUT: return 0.0
    return A_SW * (B_SW * r**-4 - 1.0) * math.exp(1.0 / (r - A_CUT))

def _phi3_py(rij, rik, cos):
    if rij >= A_CUT or rik >= A_CUT: return 0.0
    return LAMBDA3 * (cos - COS0)**2 * math.exp(GAMMA/(rij-A_CUT)) * math.exp(GAMMA/(rik-A_CUT))

def _energy_np(x, L):
    """Independent numpy oracle: brute-force pairs + all triplets. Never imported by main code."""
    N = x.shape[0]; U = 0.0
    def mi(d): return d - L*np.round(d/L)
    for i in range(N):
        for j in range(i+1, N):
            U += _phi2_py(float(np.linalg.norm(mi(x[j]-x[i]))))
    for i in range(N):
        for j in range(N):
            for k in range(j+1, N):
                if j == i or k == i: continue
                dj, dk = mi(x[j]-x[i]), mi(x[k]-x[i])
                rj, rk = float(np.linalg.norm(dj)), float(np.linalg.norm(dk))
                if rj < A_CUT and rk < A_CUT:
                    U += _phi3_py(rj, rk, float(np.dot(dj, dk)/(rj*rk)))
    return U

def test_dimer_analytic():
    x = torch.tensor([[[0.,0,0],[1.,0,0]]], dtype=torch.float64); U = mw_energy(x, 20.0)
    expect = A_SW*(B_SW-1.0)*math.exp(1.0/(1.0-A_CUT))
    assert abs(float(U) - expect) < 1e-10 and abs(expect - (-0.80340)) < 1e-4

def test_trimer_analytic():
    th = math.radians(100.0); r = 1.1
    x = torch.tensor([[[0.,0,0],[r,0,0],[r*math.cos(th), r*math.sin(th),0]]], dtype=torch.float64)
    expect = _energy_np(x[0].numpy(), 20.0)                # exercises phi3 at ALL THREE centers
    assert abs(float(mw_energy(x, 20.0)) - expect) < 1e-10
    assert abs(expect - sum(_phi2_py(d) for d in (r, r, 2*r*math.sin(th/2)))) > 0.01  # 3-body nonzero

def test_vs_numpy_oracle_random():
    g = torch.Generator().manual_seed(0); L = 5.198
    x = torch.rand(4, 64, 3, generator=g, dtype=torch.float64) * L
    U = mw_energy(x, L)
    for b in range(4):
        assert abs(float(U[b]) - _energy_np(x[b].numpy(), L)) < 1e-8 * max(1, abs(float(U[b])))

def test_invariances():
    g = torch.Generator().manual_seed(1); L = 5.198
    x = torch.rand(2, 64, 3, generator=g, dtype=torch.float64) * L
    U0 = mw_energy(x, L)
    assert torch.allclose(mw_energy(torch.remainder(x + 1.234, L), L), U0, atol=1e-9)   # translation+wrap
    perm = torch.randperm(64, generator=g)
    assert torch.allclose(mw_energy(x[:, perm], L), U0, atol=1e-9)                       # permutation

def test_cutoff_smooth():
    for r in (A_CUT - 1e-6, A_CUT - 1e-3):
        x = torch.tensor([[[0.,0,0],[r,0,0]]], dtype=torch.float64)
        assert abs(float(mw_energy(x, 20.0))) < 1e-3                                     # -> 0 at cutoff

def test_chunked_and_du_move():
    g = torch.Generator().manual_seed(2); L = 5.198
    x = torch.rand(9, 64, 3, generator=g, dtype=torch.float64) * L
    assert torch.allclose(mw_energy_chunked(x, L, chunk=4), mw_energy(x, L), atol=1e-10)
    xi = torch.rand(9, 3, generator=g, dtype=torch.float64) * L
    x2 = x.clone(); x2[:, 7] = xi
    assert torch.allclose(du_move(x, 7, xi, L), mw_energy(x2, L) - mw_energy(x, L), atol=1e-8)
