import torch
from liquid_coupling_flow.ka3d_shrink import lambda_tilde, shrink_particle_energy, shrink_pair_row
from liquid_coupling_flow.ka_pmc_3d import particle_energies
from liquid_coupling_flow.ka_energy import ka_energy

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
