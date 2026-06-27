import math, torch
from liquid_coupling_flow.ipl44.ipl_energy import ipl_box, ipl_energy, load_ipl_reference


def test_box_matches_density():
    N, L = ipl_box()
    assert N == 44 and abs(L - math.sqrt(88)) < 1e-6      # rho=0.5 -> L=sqrt(N/rho)=sqrt(88)


def test_energy_finite_and_equilibrium_on_reference():
    pos, sp = load_ipl_reference()
    U = ipl_energy(pos[:256], sp[:256])
    assert U.shape == (256,)
    assert torch.isfinite(U).all()                        # no NaN/inf from overlaps in equilibrium data
    # equilibrium configs have a tight energy band (low relative spread)
    assert float(U.std() / U.mean().abs()) < 0.2, float(U.std()/U.mean().abs())
    print("reference mean U =", float(U.mean()), " std =", float(U.std()))


if __name__ == "__main__":
    test_box_matches_density(); test_energy_finite_and_equilibrium_on_reference(); print("IPL ENERGY TESTS PASSED")
