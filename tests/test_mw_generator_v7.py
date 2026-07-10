import math

import pytest
import torch

from liquid_coupling_flow.mw.mw_energy import RHO_STAR
from liquid_coupling_flow.mw.mw_generator import mw_scaffold, wrap_pm
from liquid_coupling_flow.mw.mw_generator_v7 import (
    MWNeighborChart,
    R_CAP,
    S_C,
    load_generator_v7,
    train,
)

ARMS = ("radial", "frame")


def _model(head, seed=10, **kw):
    torch.manual_seed(seed)
    base = dict(d_model=32, n_layers=1, n_heads=2, rail_k=4, num_bins=8, knn=6)
    base.update(kw)
    return MWNeighborChart(head=head, **base)


def _perturb(m, seed):
    torch.manual_seed(seed)
    with torch.no_grad():
        for p in m.parameters():
            p.add_(0.03 * torch.randn_like(p))


# ---------------------------------------------------------------------------
# Exactness: sample() logq == log_prob(preordered) for its own output, every draw.
# N=8 (fast): domain-fit is irrelevant here because sample SCORES the placed point
# through the same mixture density as log_prob, so the two mirror by construction.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("head", ARMS)
def test_sample_logprob_exact_as_trained(head):
    m = _model(head); L = 3.7
    x, lq = m.sample(12, 8, L, gen=torch.Generator().manual_seed(5))
    scored = m.log_prob(x, L, preordered=True)
    assert torch.allclose(lq, scored, atol=2e-4), float((lq - scored).abs().max())


@pytest.mark.parametrize("head", ARMS)
def test_sample_logprob_exact_perturbed(head):
    m = _model(head); L = 3.7
    _perturb(m, 6)
    x, lq = m.sample(12, 8, L, gen=torch.Generator().manual_seed(7))
    scored = m.log_prob(x, L, preordered=True)
    assert torch.allclose(lq, scored, atol=3e-4), float((lq - scored).abs().max())


# ---------------------------------------------------------------------------
# 3D normalization quadrature (bedrock for the mixture + Jacobian + domain surfaces).
# N=64 is the smallest size where D fits the fundamental domain (L/2 = 2.598 >= 2.5).
# The radial arm's Cartesian density has an integrable 1/r^2 singularity at v=0 that a
# 48^3 midpoint grid cannot resolve (~1e-2 error, measured), so the singular core
# r < r_core is excluded from the grid and added back via a 1D radial integral of p_r
# (no singularity) -- the grid still carries the /r^2 Jacobian on the annulus.
# ---------------------------------------------------------------------------

def _fixed_fields(m, N, L, step, seed_cfg=1):
    torch.manual_seed(seed_cfg)
    x0 = torch.remainder(torch.rand(1, N, 3) * L, L)
    t, _, _ = mw_scaffold(N, L, "cpu")
    with torch.no_grad():
        h, v, n1, R, z, has_nbr = m._fields(x0, t, L)
    Rj = None if R is None else R[0, step]
    return h[0, step], n1[0, step], Rj, z[0, step]


@pytest.mark.parametrize("head", ARMS)
def test_normalization_quadrature(head):
    N = 64
    L = (N / RHO_STAR) ** (1.0 / 3.0)
    m = _model(head, d_model=16)
    _perturb(m, 2)
    hj, n1j, Rj, zj = _fixed_fields(m, N, L, step=N - 1)
    w = float(torch.sigmoid(zj))

    G = 48
    dx = L / G
    coord = (torch.arange(G) + 0.5) * dx
    gx, gy, gz = torch.meshgrid(coord, coord, coord, indexing="ij")
    V = torch.stack([gx, gy, gz], -1).reshape(-1, 3)
    v = V - L * torch.round(V / L)                  # min-image; v=0 is a cell CORNER (no sample at r=0)
    hrep = hj[None].expand(v.shape[0], -1)
    with torch.no_grad():
        if head == "radial":
            lp_local, in_D = m.head.local_logp(hrep, v)
        else:
            lp_local, in_D = m.head.local_logp(hrep, v, Rj[None].expand(v.shape[0], 3, 3))

    uniform_int = 1.0 - w                            # uniform component integrates exactly over the box
    if head == "radial":
        r = v.norm(dim=-1)
        r_core = 0.3
        grid = float((w * torch.exp(lp_local) * (in_D & (r >= r_core))).sum() * dx ** 3)
        Gr = 2000
        rr = (torch.arange(Gr) + 0.5) * r_core / Gr
        with torch.no_grad():
            lp_r = m.head.log_p_r(hj[None].expand(Gr, -1), rr)
        core = float(w * (torch.exp(lp_r).sum() * (r_core / Gr)))
        total = uniform_int + grid + core
    else:
        total = float((w * torch.exp(lp_local) * in_D).sum() * dx ** 3) + uniform_int
    assert abs(total - 1.0) < 5e-3, f"{head} integral {total}"


# ---------------------------------------------------------------------------
# w=0 fallback at j=0: exactly uniform (-3 log L), even with a perturbed w-head.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("head", ARMS)
def test_j0_is_exactly_uniform(head):
    m = _model(head); L = 3.7
    _perturb(m, 3)
    N = 8
    torch.manual_seed(4)
    x = torch.remainder(torch.rand(5, N, 3) * L, L)
    t, _, _ = mw_scaffold(N, L, "cpu")
    with torch.no_grad():
        h, v, _n1, R, z, has_nbr = m._fields(x, t, L)
        logp = m._mixture_logp(h, v, R, z, has_nbr, L)   # [B, N]
    step0 = logp[:, 0]
    assert torch.allclose(step0, torch.full_like(step0, -3.0 * math.log(L)), atol=1e-5), \
        float((step0 + 3.0 * math.log(L)).abs().max())


# ---------------------------------------------------------------------------
# Frame arm: orthonormal, right-handed frames everywhere, including fallbacks.
# ---------------------------------------------------------------------------

def test_frame_orthonormal_det_plus_one_over_steps():
    m = _model("frame"); L = 3.7
    N = 8
    torch.manual_seed(8)
    x = torch.remainder(torch.rand(3, N, 3) * L, L)
    t, _, _ = mw_scaffold(N, L, "cpu")
    with torch.no_grad():
        _h, _v, _n1, R, _z, _hn = m._fields(x, t, L)   # steps 0/1 exercise 0-/1-neighbor fallbacks
    eye = torch.eye(3).expand_as(R)
    assert torch.allclose(R @ R.transpose(-1, -2), eye, atol=1e-5)
    assert torch.allclose(torch.det(R), torch.ones(3, N), atol=1e-5)


def test_frame_fallbacks_are_right_handed():
    m = _model("frame")
    B = 5
    e1_dir = torch.randn(B, 3)
    nbr = torch.randn(B, 6, 3)
    # 0 valid neighbors; 1 valid; all-collinear-with-e1 => axis fallback path
    for n_valid, rel in [
        (torch.zeros(B, dtype=torch.long), torch.zeros(B, 6, 3)),
        (torch.ones(B, dtype=torch.long), nbr),
        (torch.full((B,), 3, dtype=torch.long),
         (e1_dir / e1_dir.norm(dim=-1, keepdim=True))[:, None, :] * torch.tensor([1.0, 2.0, 3.0, 0, 0, 0])[None, :, None]),
    ]:
        R = m._frame_from_anchor(e1_dir, rel, n_valid)
        assert torch.allclose(R @ R.transpose(-1, -2), torch.eye(3).expand(B, 3, 3), atol=1e-5)
        assert torch.allclose(torch.det(R), torch.ones(B), atol=1e-5)


# ---------------------------------------------------------------------------
# Radial arm: finite density at r -> 0; a point at r > R_CAP scores only the
# uniform branch (local branch masked out by the domain indicator).
# ---------------------------------------------------------------------------

def test_radial_density_finite_at_small_r():
    m = _model("radial")
    _perturb(m, 9)
    h = torch.randn(4, m.d_model)
    v_tiny = torch.tensor([[1e-9, 0.0, 0.0]]).expand(4, 3)
    with torch.no_grad():
        lp_local, in_D = m.head.local_logp(h, v_tiny)
    assert torch.isfinite(lp_local).all() and bool(in_D.all())
    # a full sample also has finite logq (r never hits the 1e-6 clamp in practice)
    _x, lq = m.sample(8, 8, 3.7, gen=torch.Generator().manual_seed(1))
    assert torch.isfinite(lq).all()


def test_radial_outside_ball_is_uniform_only():
    m = _model("radial"); L = 5.2
    _perturb(m, 11)
    h = torch.randn(4, m.d_model)
    z = m.w_head(h).squeeze(-1)
    v_far = torch.tensor([[R_CAP + 0.1, 0.0, 0.0]]).expand(4, 3)   # |v| > R_CAP => out of D
    has_nbr = torch.ones(4, dtype=torch.bool)
    with torch.no_grad():
        logp = m._mixture_logp(h, v_far, None, z, has_nbr, L)
    want = torch.nn.functional.logsigmoid(-z) - 3.0 * math.log(L)   # log((1-w)/L^3)
    assert torch.allclose(logp, want, atol=1e-5), float((logp - want).abs().max())


# ---------------------------------------------------------------------------
# Checkpoint round-trip through the real train() path + load_generator_v7.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("head", ARMS)
def test_train_and_load_roundtrip(tmp_path, head):
    N = 64
    L = (N / RHO_STAR) ** (1.0 / 3.0)
    cfgs = torch.rand(32, N, 3, generator=torch.Generator().manual_seed(9)) * L
    bank = tmp_path / "bank.pt"
    torch.save({"cfgs": cfgs}, bank)
    result = train(steps=1, batch=2, val_every=1, out=str(tmp_path / f"v7_{head}.pt"),
                   primary_thin=1, val_frac=0.5, art_path=str(bank), extra_banks=[],
                   device="cpu", head=head, d_model=16, n_layers=1, n_heads=2,
                   rail_k=4, num_bins=4, knn=4)
    for key in ("nll", "struct", "last"):
        ck = torch.load(result[key], map_location="cpu", weights_only=False)
        assert ck["architecture"] == f"neighbor_chart_v7_{head}"
        assert ck["head"] == head
        assert {"peak_err", "core_mass", "composite"} <= ck["struct"].keys()
    m = load_generator_v7(result["last"], device="cpu")
    assert m.head_kind == head
    x = torch.remainder(cfgs[:2], L)
    lp = m.log_prob(x, L)
    assert lp.shape == (2,) and torch.isfinite(lp).all()
