import numpy as np
import torch
import pytest
from liquid_coupling_flow.ptu.tables import build_tables_u, build_tables_lam
from liquid_coupling_flow.ptu.kernels import row_e, mobile_U, disp_sweep, seed_numba
from liquid_coupling_flow.ka_energy import ka_energy
from liquid_coupling_flow.ka3d_cavity_carve import carve
from liquid_coupling_flow.ka3d_cavity_ar import _mic

torch.set_grad_enabled(False)
BIGL = 100.0


@pytest.fixture(scope="module")
def cavity():
    D = torch.load("liquid_coupling_flow/artifacts/ka3d_dataset_N512_T0.5.pt",
                   map_location="cpu", weights_only=False)
    X, S, L = D["x"].float(), D["s"].long(), float(D["L"])
    g = torch.Generator().manual_seed(3)
    ci = 950  # held-out config
    c = torch.rand(3, generator=g) * L
    p = carve(X[ci], S[ci], c, 2.0, L)
    xin = _mic(p["x_in"], c, L)
    xout = _mic(p["x_out"], c, L)
    bm = xout.norm(dim=-1) < (2.0 + 2.5)
    bnd, sb = xout[bm], p["s_out"][bm]
    allx = np.concatenate([xin.double().numpy(), bnd.double().numpy()])
    alls = np.concatenate([p["s_in"].numpy(), sb.numpy()]).astype(np.int64)
    n = xin.shape[0]
    return allx, alls, n, allx.shape[0], xin, p["s_in"], bnd, sb


def test_u1_matches_ka_energy(cavity):
    """At u=1 the kernel must reproduce ka_energy's mobile-involved energy to 1e-8/particle.
    ka_energy(all) - ka_energy(boundary only) = mobile-mobile + mobile-pinned."""
    allx, alls, n, n_tot, xin, sin, bnd, sb = cavity
    tabs = build_tables_u(1.0)
    u_kernel = mobile_U(allx, alls, n, n_tot, *tabs)
    x_all = torch.tensor(allx)[None]
    s_all = torch.tensor(alls)[None]
    u_full = float(ka_energy(x_all, s_all, BIGL)[0])
    u_bb = float(ka_energy(torch.tensor(allx[n:])[None], torch.tensor(alls[n:])[None], BIGL)[0])
    assert abs(u_kernel - (u_full - u_bb)) / n < 1e-8


def test_u0_label_permutation_invariance(cavity):
    """At u=0 mobile-mobile is species-blind: permuting MOBILE labels changes mobile_U only
    through the mobile-pinned term; with a label-permutation among mobiles ONLY, and mp tables
    at half-interpolation, the mm part must be exactly invariant. Test the mm-only invariance
    by using mp tables == mm tables (fully species-blind everywhere)."""
    allx, alls, n, n_tot, *_ = cavity
    sig_mm, eps_mm, _, _ = build_tables_u(0.0)
    tabs_blind = (sig_mm, eps_mm, sig_mm, eps_mm)
    u0 = mobile_U(allx, alls, n, n_tot, *tabs_blind)
    rng = np.random.default_rng(0)
    alls_perm = alls.copy()
    alls_perm[:n] = rng.permutation(alls[:n])
    u1 = mobile_U(allx, alls_perm, n, n_tot, *tabs_blind)
    assert abs(u0 - u1) < 1e-10


def test_lam_arm_matches_prior_implementation(cavity):
    """lam-arm kernel at lam=0.9 must agree with the (verified-correct) torch Cavity.full_U
    convention: deformed sigma inside LJ, fixed rc=2.5*deformed sigma, lam-independent shift.
    Regression pin: compute once with the torch reference formula inline."""
    allx, alls, n, n_tot, *_ = cavity
    lam = 0.9
    tabs = build_tables_lam(lam)
    u_kernel = mobile_U(allx, alls, n, n_tot, *tabs)
    # inline torch reference (same math as bcy_gpts_v2.Cavity.full_U, f64)
    from liquid_coupling_flow.ka_energy import SIGMA, EPS
    x = torch.tensor(allx); s = torch.tensor(alls)
    d2 = torch.cdist(x[None], x[None])[0] ** 2
    d2.fill_diagonal_(1e12)
    sig0 = torch.tensor(SIGMA, dtype=torch.float64)[s[:, None], s[None, :]]
    eps0 = torch.tensor(EPS, dtype=torch.float64)[s[:, None], s[None, :]]
    lam_col = torch.full((n_tot,), 0.5 * (1 + lam), dtype=torch.float64)
    lam_col[:n] = lam
    sig = sig0 * lam_col[None, :]
    inv6 = (sig ** 2 / d2) ** 3
    e = 4 * eps0 * (inv6 ** 2 - inv6)
    s6 = (1.0 / 2.5) ** 6
    e = torch.where(d2 < (2.5 * sig) ** 2, e - 4 * eps0 * (s6 ** 2 - s6), torch.zeros_like(e))
    u_ref = float(e[:n, :n].sum() * 0.5 + e[:n, n:].sum())
    assert abs(u_kernel - u_ref) / n < 1e-8


def test_disp_sweep_runs_and_accepts(cavity):
    allx, alls, n, n_tot, *_ = cavity
    tabs = build_tables_u(1.0)
    a = allx.copy()
    seed_numba(0)
    acc = disp_sweep(a, alls, n, n_tot, 2.0, 2.0, 0.3, *tabs)
    assert 0 < acc < n                     # some but not all moves accepted at T=0.5
    assert np.all(np.abs(a[n:] - allx[n:]) == 0.0)   # pinned particles never move
