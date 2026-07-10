import math
import torch
from liquid_coupling_flow.ka3d_shrink import lambda_tilde, shrink_particle_energy, shrink_pair_row, cavity_move, replica_exchange
from liquid_coupling_flow.ka_pmc_3d import particle_energies
from liquid_coupling_flow.ka_energy import ka_energy, SIGMA, EPS, RCUT_FACTOR
from liquid_coupling_flow.ka_cavity import assert_mobile_inside


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


def test_cavity_move_never_escapes():
    torch.manual_seed(0); N=80; L=(N/1.2)**(1/3); center=torch.tensor([L/2]*3); R=1.8
    x=torch.rand(1,N,3)*L; s=torch.tensor([[0]*64+[1]*16])
    d=x-center; d=d-L*torch.round(d/L); mob=d.square().sum(-1)<R*R
    r=(x-center).norm(dim=-1,keepdim=True).clamp_min(1e-6)
    x=torch.where(mob[...,None]&(r>=R), center+(x-center)/r*(R*0.9), x)   # valid start
    U=0.5*shrink_particle_energy(x,s,L,1.0,mob).sum(1)
    for _ in range(200): x,U,_=cavity_move(x,s,U,mob,center,R,L,2.0,1.0,0.08)
    assert_mobile_inside(x,mob,center,R,L)


def test_replica_exchange_weight_ratio_explicit():
    # Two replicas, two fixed configs: log_acc is deterministic given the configs (the accept coin is
    # drawn separately), so a single call suffices to check it against the explicit joint-weight ratio.
    # double precision: random-uniform-at-density cores overlap hard enough that energies/logA reach
    # ~1e8-1e9 (see below), where float32's ~1e-7 relative precision alone blows past the atol=1e-4
    # deterministic-equality check.
    torch.manual_seed(3); N=40; L=(N/1.2)**(1/3); center=torch.tensor([L/2]*3, dtype=torch.double); R=1.6
    x=(torch.rand(2,N,3)*L).double(); s=torch.tensor([[0]*32+[1]*8]*2); mob=torch.ones(2,N,dtype=torch.bool)
    betas=torch.tensor([2.0,1.3], dtype=torch.double); lams=torch.tensor([1.0,0.8], dtype=torch.double)
    def H(cfg_x, lam): return float(0.5*shrink_particle_energy(cfg_x[None],s[:1],L,float(lam),mob[:1]).sum(1))
    Ha_xa=H(x[0],1.0); Hb_xb=H(x[1],0.8); Ha_xb=H(x[1],1.0); Hb_xa=H(x[0],0.8)
    logA=-(2.0*(Ha_xb-Ha_xa)) - (1.3*(Hb_xa-Hb_xb))
    # random-uniform N=40 @ rho=1.2 gives astronomically overlapping cores (logA ~ 1e8+): min(1,exp(x))
    # is well-defined for any real x, but math.exp(x) itself overflows there, so guard it (mathematically
    # identical to min(1.0, math.exp(loga)) for every real loga, just without the OverflowError crash).
    def acc_prob(loga): return 1.0 if loga >= 0 else math.exp(loga)
    expected=acc_prob(logA)
    _,_,la=replica_exchange(x.clone(), s.clone(), mob, center, R, L, betas, lams)
    assert abs(float(la[0]) - logA) < 1e-4
    assert abs(acc_prob(float(la[0])) - expected) < 1e-4
