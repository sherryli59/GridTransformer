import torch
from liquid_coupling_flow.ka3d_shrink import lambda_tilde, shrink_particle_energy, shrink_pair_row
from liquid_coupling_flow.ka_pmc_3d import particle_energies
from liquid_coupling_flow.ka_energy import ka_energy, SIGMA, EPS, RCUT_FACTOR


def _ref_particle_energy(x, s, L, lam, mobile):
    """Independent, explicit-loop shifted-LJ per-particle energy with sigma manually scaled by
    lambda_tilde_pair (lam mobile-mobile, (1+lam)/2 mobile-pinned, 1.0 pinned-pinned). Deliberately does
    NOT call the module under test's _lt_values/lambda_tilde, so it is a genuine cross-check."""
    B, N, _ = x.shape
    dtype = x.dtype
    si = s.long()
    t_sig = torch.tensor(SIGMA, dtype=dtype); t_eps = torch.tensor(EPS, dtype=dtype)
    pe = torch.zeros(B, N, dtype=dtype)
    for bi in range(B):
        for i in range(N):
            for j in range(N):
                if j == i:
                    continue
                sig = t_sig[si[bi, i], si[bi, j]]; eps = t_eps[si[bi, i], si[bi, j]]
                mi = bool(mobile[bi, i]); mj = bool(mobile[bi, j])
                lt = lam if (mi and mj) else ((1.0 + lam) / 2.0 if (mi or mj) else 1.0)
                sig_s = lt * sig; rc = RCUT_FACTOR * sig_s
                d = x[bi, i] - x[bi, j]; d = d - L * torch.round(d / L)
                r2 = (d ** 2).sum()
                if r2 < rc ** 2:
                    inv6 = (sig_s ** 2 / r2) ** 3
                    src6 = (sig_s / rc) ** 6
                    pe[bi, i] = pe[bi, i] + 4 * eps * (inv6 ** 2 - inv6) - 4 * eps * (src6 ** 2 - src6)
    return pe

def test_lambda_one_matches_standard():
    torch.manual_seed(0); B,N,L=2,30,3.4
    x=torch.rand(B,N,3)*L; s=torch.tensor([[0]*24+[1]*6]*B); mob=torch.ones(B,N,dtype=torch.bool)
    assert torch.allclose(shrink_particle_energy(x,s,L,1.0,mob), particle_energies(x,s,L), atol=1e-5)

def test_lambda_tilde_rules():
    mob=torch.tensor([[True,True,False]]); lt=lambda_tilde(0.6,mob,mob)
    assert abs(lt[0,0,1]-0.6)<1e-6 and abs(lt[0,0,2]-0.8)<1e-6 and abs(lt[0,2,2]-1.0)<1e-6

def test_pair_row_delta_matches_full_energy_delta_at_lambda1():
    torch.manual_seed(1); B,N,L=3,24,3.2
    x=(torch.rand(B,N,3)*L).double(); s=torch.tensor([[0]*19+[1]*5]*B); mob=torch.ones(B,N,dtype=torch.bool)
    rows=torch.arange(B); i=torch.tensor([2,5,7]); xi=torch.remainder(x[rows,i]+0.1,L)
    delta_row = shrink_pair_row(x,s,(rows,i),xi,L,1.0,mob) - shrink_pair_row(x,s,(rows,i),x[rows,i],L,1.0,mob)
    xnew=x.clone(); xnew[rows,i]=xi
    delta_full = ka_energy(xnew,s,L) - ka_energy(x,s,L)
    assert torch.allclose(delta_row, delta_full, atol=1e-4)

def test_shrinkage_effect_lambda_half_mixed_mobile():
    torch.manual_seed(2); B,N,L=2,24,3.4; lam=0.5
    x=(torch.rand(B,N,3)*L).double(); s=torch.tensor([[0]*19+[1]*5]*B)
    mob=torch.zeros(B,N,dtype=torch.bool); mob[:, :16]=True          # 16 mobile / 8 pinned
    ref = _ref_particle_energy(x,s,L,lam,mob)
    got = shrink_particle_energy(x,s,L,lam,mob)
    # dense shrinkage path vs INDEPENDENT hand-scaled reference
    assert torch.allclose(got, ref, atol=1e-8)
    # shrinkage is not a no-op: lam=0.5 (mixed mask) differs materially from lam=1
    assert not torch.allclose(got, shrink_particle_energy(x,s,L,1.0,mob), atol=1e-3)
    # O(N)-row shrinkage path: moving a MOBILE particle -> row delta == independent full-energy delta
    rows=torch.arange(B); i=torch.tensor([3,3]); xi=torch.remainder(x[rows,i]+0.15,L)
    delta_row = shrink_pair_row(x,s,(rows,i),xi,L,lam,mob) - shrink_pair_row(x,s,(rows,i),x[rows,i],L,lam,mob)
    xnew=x.clone(); xnew[rows,i]=xi
    delta_ref = 0.5*_ref_particle_energy(xnew,s,L,lam,mob).sum(1) - 0.5*_ref_particle_energy(x,s,L,lam,mob).sum(1)
    assert torch.allclose(delta_row, delta_ref, atol=1e-8)
