import os

import torch

from liquid_coupling_flow.mw.mw_energy import RHO_STAR, mw_energy
from liquid_coupling_flow.mw.mw_generator import canonical_order, mw_scaffold, wrap_pm
from liquid_coupling_flow.mw.mw_generator_v4 import MWGlobalAR
from liquid_coupling_flow.mw.mw_generator_v10 import (
    MWV4ToroidalResidual, ToroidalResidualHead, _MODEL_KEYS, _insertion_energy,
    _smooth_clip, _struct_loss, finetune_struct, load_generator_v10, load_v4_exact,
)


def _pair(seed=1):
    torch.manual_seed(seed)
    v4 = MWGlobalAR(d_model=24, n_layers=1, n_heads=2, n_mix=6, rail_k=4)
    v10 = MWV4ToroidalResidual(d_model=24, n_layers=1, n_heads=2,
                               n_mix=6, rail_k=4, num_bins=7)
    body = {}
    for key, value in v4.state_dict().items():
        body["head.base." + key[5:] if key.startswith("head.") else key] = value
    v10.load_state_dict(body, strict=False)
    return v4.eval(), v10.eval()


def test_identity_head_matches_v4_density_and_samples():
    v4, v10 = _pair(); h = torch.randn(13, 24); u = torch.rand(13, 3) * 7.8 - 3.9
    assert torch.allclose(v4.head.log_prob(h, u, 4), v10.head.log_prob(h, u, 4), atol=2e-5)
    g1 = torch.Generator().manual_seed(3); g2 = torch.Generator().manual_seed(3)
    u1, lp1 = v4.head.sample(h, 4, g1); u2, lp2 = v10.head.sample(h, 4, g2)
    assert torch.allclose(u1, u2, atol=2e-6)
    assert torch.allclose(lp1, lp2, atol=2e-5)


def test_complete_model_identity_matches_v4_n64():
    v4, v10 = _pair(); N = 64; L = (N / RHO_STAR) ** (1 / 3)
    x = torch.rand(3, N, 3) * L
    assert torch.allclose(v4.log_prob(x, L), v10.log_prob(x, L), atol=3e-4)


def test_perturbed_transport_sample_logprob_exact_and_periodic():
    _, m = _pair(); N = 64; L = (N / RHO_STAR) ** (1 / 3)
    torch.manual_seed(4)
    with torch.no_grad():
        for p in m.head.transforms.parameters(): p.add_(0.04 * torch.randn_like(p))
    x, lq = m.sample(3, N, L, gen=torch.Generator().manual_seed(5))
    lp = m.log_prob(x, L, preordered=True)
    assert torch.allclose(lq, lp, atol=5e-4), float((lq - lp).abs().max())
    h = torch.randn(9, 24); y = torch.rand(9, 3) * 8 - 4
    shift = torch.tensor([[8.0, -8.0, 16.0]])
    assert torch.allclose(m.head.log_prob(h, y, 4), m.head.log_prob(h, y + shift, 4), atol=3e-5)


def test_canonical_generation_order_is_not_resorted(monkeypatch):
    import liquid_coupling_flow.mw.mw_generator_v4 as v4mod
    _, m = _pair(); N = 64; L = (N / RHO_STAR) ** (1 / 3)
    real = v4mod.canonical_order; calls = {"n": 0}
    def counted(*args, **kwargs): calls["n"] += 1; return real(*args, **kwargs)
    monkeypatch.setattr(v4mod, "canonical_order", counted)
    m.log_prob(torch.rand(1, N, 3) * L, L)
    assert calls["n"] == 1
    x, lq = m.sample(2, N, L, gen=torch.Generator().manual_seed(7))
    assert torch.allclose(lq, m.log_prob(x, L, preordered=True), atol=4e-4)
    assert calls["n"] == 1


def test_real_checkpoint_remap_accepts_only_transport_missing(tmp_path):
    v4, _ = _pair(); ck = tmp_path / "v4.pt"
    torch.save({"state_dict": v4.state_dict()}, ck)
    m = MWV4ToroidalResidual(d_model=24, n_layers=1, n_heads=2,
                             n_mix=6, rail_k=4, num_bins=7)
    load_v4_exact(m, str(ck), "cpu")
    for key, value in v4.state_dict().items():
        target = "head.base." + key[5:] if key.startswith("head.") else key
        assert torch.equal(m.state_dict()[target], value)


# ---------------------------------------------------------------------------
# Structure-aware fine-tune: reparameterized sample-energy penalty.
# ---------------------------------------------------------------------------

def _tiny_v10(seed=1, **kw):
    torch.manual_seed(seed)
    base = dict(d_model=24, n_layers=1, n_heads=2, n_mix=6, rail_k=4, num_bins=7)
    base.update(kw)
    return MWV4ToroidalResidual(**base)


def _ordered(x, L):
    """Canonicalize [B,64,3] positions the way finetune_struct does."""
    x = torch.remainder(x, L)
    t, rank, R = mw_scaffold(x.shape[1], L, x.device)
    perm = canonical_order(x, L, R, rank)
    xo = torch.gather(x, 1, perm[..., None].expand(-1, -1, 3))
    return xo, t, R


def test_struct_grad_flows():
    # Isolate the reparam path: L_struct only (lam=1, NLL removed). Grads must reach BOTH the MDN
    # (component mu/L via the within-component reparam draw) and the transport spline (forward_transport).
    m = _tiny_v10(); m.train()
    N = 64; L = (N / RHO_STAR) ** (1.0 / 3.0)
    torch.manual_seed(2)
    x = torch.remainder(torch.rand(3, N, 3) * L, L)
    xo, t, R = _ordered(x, L)
    s, bound = (L / R) / 2.0, float(R)
    m.zero_grad()
    l_struct, u_ins = _struct_loss(m, xo, t, s, bound, L, clip_e=50.0, three_body=True)
    assert float(u_ins.abs().sum()) > 0, "insertion energy identically zero -> nothing to learn from"
    l_struct.backward()
    base_g = m.head.base.out.weight.grad
    tr_g = [p.grad for p in m.head.transforms.parameters() if p.grad is not None]
    assert base_g is not None and float(base_g.abs().sum()) > 0, "no gradient to the MDN through the draw"
    assert tr_g and any(float(g.abs().sum()) > 0 for g in tr_g), "no gradient to the transport spline"


def test_upen_smooth_clip():
    clip_e = 10.0
    u = torch.linspace(-100.0, 100.0, 41)
    pen = _smooth_clip(u, clip_e)
    # |clip_e*tanh(.)| <= clip_e strictly in exact arithmetic; float32 tanh(10) rounds to 1.0 so the
    # extreme entries land exactly at clip_e -- bounded BY clip_e, never exceeding it.
    assert bool((pen.abs() <= clip_e).all()), "penalty not bounded by clip_e"
    small = torch.linspace(-0.5, 0.5, 11)                          # linear regime: ~identity, not clamped
    assert torch.allclose(_smooth_clip(small, clip_e), small, atol=1e-2)
    u3 = torch.tensor([3.0 * clip_e], requires_grad=True)   # 3x the clip: clamp would zero the gradient
    _smooth_clip(u3, clip_e).backward()
    assert float(u3.grad.abs()) > 0, "gradient vanished at U_ins = 3*clip_e (tanh must preserve it)"


def test_insertion_energy_matches_du():
    # Ground-truth anchor: U_ins (pair + 3body) == mw_energy(prefix + placement) - mw_energy(prefix),
    # for arbitrary placements against the true prefix. float64 so the 1e-4 tolerance is meaningful.
    N = 64; L = (N / RHO_STAR) ** (1.0 / 3.0)
    torch.manual_seed(0)
    x = torch.remainder(torch.rand(3, N, 3) * L, L).double()
    xo, _t, _R = _ordered(x, L)
    placed = torch.remainder(torch.rand(3, N, 3).double() * L, L)   # placements != the config continuation
    U = _insertion_energy(placed, xo, L, three_body=True)           # full pair + 3-body
    U_pair = _insertion_energy(placed, xo, L, three_body=False)
    for j in (1, 5, 20, 40, 63):
        prefix = xo[:, :j]
        sub = torch.cat([prefix, placed[:, j:j + 1]], dim=1)
        du = mw_energy(sub, L) - mw_energy(prefix, L)
        assert torch.allclose(U[:, j], du, atol=1e-4), f"j={j}: {float((U[:, j] - du).abs().max())}"
        # SW 3-body = lambda*(cos-cos0)^2*h*h >= 0, so the pair-only insertion energy never exceeds it.
        assert bool((U_pair[:, j] <= U[:, j] + 1e-9).all())
    # placed == xo reproduces the AR factorization of the true config energy, step by step.
    U_true = _insertion_energy(xo, xo, L, three_body=True)
    assert torch.allclose(U_true.sum(1), mw_energy(xo, L), atol=1e-4), \
        float((U_true.sum(1) - mw_energy(xo, L)).abs().max())


def test_finetune_smoke(tmp_path):
    N = 64; L = (N / RHO_STAR) ** (1.0 / 3.0)
    cfgs = torch.rand(32, N, 3, generator=torch.Generator().manual_seed(9)) * L
    bank = tmp_path / "bank.pt"
    torch.save({"cfgs": cfgs}, bank)
    warm_model = _tiny_v10(seed=3, d_model=16, num_bins=4)
    warm = tmp_path / "warm.pt"
    torch.save({"state_dict": warm_model.state_dict(),
                **{k: getattr(warm_model, k) for k in _MODEL_KEYS}}, warm)
    result = finetune_struct(steps=5, batch=2, lr=1e-4, lam=0.05, lam_warmup=2, clip_e=10.0,
                             three_body=True, val_every=2, warm=str(warm),
                             out=str(tmp_path / "v10s.pt"), seed=7, primary_thin=1, val_frac=0.5,
                             art_path=str(bank), extra_banks=[], device="cpu")
    for key in ("nll", "struct", "last"):
        assert os.path.exists(result[key])
        ck = torch.load(result[key], map_location="cpu", weights_only=False)
        assert ck["architecture"] == "v10_struct_finetune"
        assert "shell2" in ck["struct"]
    # Density UNTOUCHED by the objective: sample() log q still mirrors log_prob(preordered).
    m = load_generator_v10(result["last"], device="cpu")
    x, lq = m.sample(3, N, L, gen=torch.Generator().manual_seed(1))
    lp = m.log_prob(x, L, preordered=True)
    assert torch.allclose(lq, lp, atol=3e-4), float((lq - lp).abs().max())
