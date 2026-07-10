"""Tests for ExcludedVolumeBase: the non-learned analytic soft-core insertion base (Task-11 control).

Exactness rests on the PIECEWISE-CONSTANT quadrature scheme: sample() and log_q() read the density at
the same point (the containing cell's center), so log_q of a config sample() produced equals its
returned logq up to float. The physics gates verify the base does its ONE job (carve excluded volume)
WITHOUT the job it must leave to the learned model (install shells)."""
import math
import torch
import pytest

from liquid_coupling_flow.mw.mw_base import ExcludedVolumeBase
from liquid_coupling_flow.mw.mw_energy import mw_energy, RHO_STAR


def _box(N):
    return (N / RHO_STAR) ** (1.0 / 3.0)


def _min_image_dists(x, L):
    """All ordered pairwise min-image distances [B,N,N] (diagonal set to +inf)."""
    d = x[:, :, None, :] - x[:, None, :, :]
    d = d - L * torch.round(d / L)
    r = d.norm(dim=-1)
    N = x.shape[1]
    r = r + torch.eye(N, device=x.device)[None] * 1e9
    return r


def test_sample_logprob_exact():
    # The weight-path exactness invariant: log_q(x) == sample()'s accumulated logq for EVERY config.
    N = 8
    L = _box(N)
    base = ExcludedVolumeBase(N, L)
    gen = torch.Generator().manual_seed(0)
    x, logq_sampled = base.sample(16, gen, return_logq=True)
    logq_scored = base.log_q(x)                                  # default preordered=True
    err = (logq_sampled - logq_scored).abs().max().item()
    assert err < 1e-4, f"sample/log_q exactness broke: max|dlogq|={err:.2e}"


def test_normalized():
    # For one fixed prefix the discrete density integrates to 1: sum_cells p(cell)*V_cell == 1.
    N = 8
    L = _box(N)
    base = ExcludedVolumeBase(N, L)
    gen = torch.Generator().manual_seed(1)
    prefix = torch.rand(1, 3, 3, generator=gen) * L             # 3 placed particles
    centers = base._grid_centers("cpu")[None]                   # [1,M,3]
    phi = base._phi_at(centers, prefix, L)[0]                   # [M]
    w = torch.exp(-phi)
    Z_disc = (w * base.V_cell).sum()
    total = ((w / Z_disc) * base.V_cell).sum().item()           # sum_cells p(cell)*V_cell
    assert abs(total - 1.0) < 1e-5, f"discrete density not normalized: {total:.8f}"


def test_carves_core():
    # The base does its job: far fewer clashes (r<0.8) and much lower energy than a uniform draw.
    N = 64
    L = _box(N)
    base = ExcludedVolumeBase(N, L)
    gen = torch.Generator().manual_seed(2)
    x = base.sample(16, gen)
    xu = torch.rand(16, N, 3, generator=gen) * L

    close_x = (_min_image_dists(x, L) < 0.8).sum().item() / 2   # each pair once
    close_u = (_min_image_dists(xu, L) < 0.8).sum().item() / 2
    assert close_x < 0.2 * close_u + 1.0, f"core not carved: close pairs base={close_x} uniform={close_u}"

    e_x = mw_energy(x, L).mean().item()
    e_u = mw_energy(xu, L).mean().item()
    assert e_x < e_u, f"base energy {e_x:.1f} not << uniform {e_u:.1f}"
    # store for the report
    print(f"\n[carves_core] close(r<0.8): base={close_x:.0f} uniform={close_u:.0f} | "
          f"<U>: base={e_x:.1f} uniform={e_u:.1f}")


def test_no_shells():
    # Pure core, no attraction: g(r) at the mW second-shell radius (~1.85) must be ~1 (no shell).
    N = 64
    L = _box(N)
    base = ExcludedVolumeBase(N, L)
    gen = torch.Generator().manual_seed(3)
    x = base.sample(24, gen)
    rho = N / L ** 3

    r_probe, dr = 1.85, 0.15
    r = _min_image_dists(x, L)                                   # [B,N,N], self=+inf
    in_bin = ((r > r_probe - dr / 2) & (r < r_probe + dr / 2)).sum().item()  # ordered pairs
    ideal = x.shape[0] * N * rho * 4.0 * math.pi * r_probe ** 2 * dr         # ordered-pair ideal count
    g = in_bin / ideal
    print(f"\n[no_shells] g(r={r_probe})={g:.3f}")
    assert abs(g - 1.0) < 0.3, f"unexpected structure at r={r_probe}: g={g:.3f} (base must install NO shell)"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA smoke")
def test_cuda_smoke():
    N = 64
    L = _box(N)
    base = ExcludedVolumeBase(N, L)
    gen = torch.Generator(device="cuda").manual_seed(0)
    x, logq_sampled = base.sample(8, gen, return_logq=True)
    assert x.is_cuda
    err = (logq_sampled - base.log_q(x)).abs().max().item()
    print(f"\n[cuda_smoke N=64] max|dlogq|={err:.2e}")
    assert err < 1e-4
