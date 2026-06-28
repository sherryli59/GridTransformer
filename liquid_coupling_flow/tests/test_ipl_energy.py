import math, torch, pytest
from liquid_coupling_flow.ipl44.ipl_energy import ipl_box, ipl_energy, ipl_gr, load_ipl_reference


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


def test_gr_physical_on_reference():
    """g(r) must be a real radial distribution: ~0 inside the excluded volume (the data's min pair
    distance is ~1.0) and ->1 at large r. Guards against histogramming difference-vector components
    (their RDF bug), which produces an unphysical g(r->0) spike."""
    pos, sp = load_ipl_reference()
    N, L = ipl_box()
    r, g = ipl_gr(pos[:2048], sp[:2048], L, bins=120)
    assert r.shape == g.shape == (120,)
    core = g[r < 0.8]                                      # excluded volume: no pairs this close
    assert float(core.max()) < 0.1, f"unphysical g(r) inside core: max {float(core.max())}"
    assert float(g.max()) > 1.5, f"missing first-shell peak: max g {float(g.max())}"
    tail = g[r > L / 2 - 1.0]                              # g(r) -> 1 at large r
    assert abs(float(tail.mean()) - 1.0) < 0.3, float(tail.mean())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA")
def test_energy_cuda_matches_cpu():
    """Verify per-device cache: CUDA and CPU energies agree (catches un-moved mask crash)."""
    pos, sp = load_ipl_reference()
    pos_cpu, sp_cpu = pos[:16], sp[:16]
    U_cpu = ipl_energy(pos_cpu, sp_cpu)
    U_cuda = ipl_energy(pos_cpu.cuda(), sp_cpu.cuda())
    assert torch.allclose(U_cpu, U_cuda.cpu(), atol=1e-3), (
        f"CPU/CUDA energy mismatch: max |diff| = {(U_cpu - U_cuda.cpu()).abs().max()}"
    )


if __name__ == "__main__":
    test_box_matches_density(); test_energy_finite_and_equilibrium_on_reference(); print("IPL ENERGY TESTS PASSED")
