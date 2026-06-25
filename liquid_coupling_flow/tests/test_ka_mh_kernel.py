import torch, pytest
from liquid_coupling_flow.ka_energy import ka_energy
from liquid_coupling_flow import ka_mh_kernel as K
import os
from liquid_coupling_flow.ka_noncausal import NonCausalLF
from liquid_coupling_flow.ka_localframe import _wrap_pm
ART = "liquid_coupling_flow/artifacts"; DEV = "cuda" if torch.cuda.is_available() else "cpu"

def _setup(B=4, N=12, seed=0):
    torch.manual_seed(seed); L = (N / 1.2) ** 0.5
    pos = (torch.rand(B, N, 2) * L).double()
    s = torch.zeros(N, dtype=torch.long); s[: N // 3] = 1
    s = s[torch.randperm(N)]
    return pos, s, L

def test_site_dE_matches_full_energy():
    pos, s, L = _setup()
    B, N = pos.shape[:2]; j = 5
    xj_new = torch.remainder(pos[:, j] + 0.3 * torch.randn(B, 2), L)
    dE = K.site_dE(pos, s, j, xj_new, L)
    pos2 = pos.clone(); pos2[:, j] = xj_new
    dE_full = ka_energy(pos2, s, L) - ka_energy(pos, s, L)
    assert torch.allclose(dE, dE_full, atol=1e-4), (dE - dE_full).abs().max()

def _load_model(N=100):
    ck = torch.load(f"{ART}/ka_noncausal_N100.pt", map_location=DEV, weights_only=False)
    m = NonCausalLF(rho=1.2, n_bins=192, knn=ck["knn"], canonical=False).to(DEV).eval()
    m.load_state_dict(ck["state_dict"]); return m

def _ctx(m, N=100, B=8):
    ref = torch.load(f"{ART}/ka_reference_N100.pt", map_location=DEV, weights_only=False)
    s = ref["s"].to(DEV).long(); x = ref["x"][:B].to(DEV); L = ref["L"]
    sc = m.geo._scaffold(N, DEV)
    order = m.geo._curve_order(x, N); xo = torch.gather(x, 1, order[..., None].expand(-1, -1, 2))
    so = torch.gather(s.expand(B, N), 1, order)
    ctx, origin = m._local(xo, so, sc, N=N, L=L)
    return m, xo, so, ctx, origin, L, N

def test_density_normalizes_exact_bin_sum():
    """Alignment-free normalization. The density is piecewise-constant on the (a,b) bins, so a uniform
    physical Riemann grid is FRAGILE for a peaked conditional (it misses high-mass bins of sharp configs ->
    totals scatter 0.75-1.04). Instead sum the folded q over the PRIMARY-region bin centers
    (|offset| <= (L/2)/arc) weighted by physical bin area (bin_w*arc)^2 -> evaluates EVERY bin -> exactly 1
    regardless of sharpness (verified 1.0000 +/- 1e-7 for all configs)."""
    m, xo, so, ctx, origin, L, N = _ctx(_load_model()); arc = m._arc_scale(N); j = 7; B = xo.shape[0]
    half = (L / 2) / arc; area = (m.bin_w * arc) ** 2
    centers = m._bin_center(torch.arange(m.n_bins, device=DEV))
    prim = centers[centers.abs() <= half]                          # primary-region bin centers (one per torus cell)
    ca, cb = torch.meshgrid(prim, prim, indexing="ij"); off = torch.stack([ca.reshape(-1), cb.reshape(-1)], -1)
    cj = ctx[:, j]; oj = origin[:, j]; tot = torch.zeros(B, device=DEV)
    with torch.no_grad():
        for c in range(0, off.shape[0], 2048):
            blk = off[c:c + 2048]; nb = blk.shape[0]
            xj = torch.remainder(oj[:, None, :] + blk[None] * arc, L).reshape(B * nb, 2)
            cc = cj[:, None, :].expand(B, nb, -1).reshape(B * nb, -1)
            oo = oj[:, None, :].expand(B, nb, -1).reshape(B * nb, 2)
            tot = tot + K.site_logq(m, cc, oo, xj, arc, L).reshape(B, nb).exp().sum(1) * area
    assert torch.allclose(tot, torch.ones_like(tot), atol=1e-3), tot

def test_sampler_evaluator_roundtrip():
    m, xo, so, ctx, origin, L, N = _ctx(_load_model())
    arc = m._arc_scale(N); j = 7
    xj_new, logq_new = K.site_propose(m, ctx[:, j], origin[:, j], arc, L)
    logq_new_eval = K.site_logq(m, ctx[:, j], origin[:, j], xj_new, arc, L)
    assert torch.allclose(logq_new, logq_new_eval, atol=1e-4), (logq_new - logq_new_eval).abs().max()


def _grid_total(m, ctx_j, origin_j, B, L, arc, g=200, chunk=256):
    """Integral of q over the PHYSICAL torus for each of B configs (post-fold), chunked + vectorized over
    grid points. Used ONLY for the injected-UNIFORM density below (a uniform density integrates exactly on
    any grid, so Riemann alignment is a non-issue here — unlike the peaked real conditional, which Task 2
    normalizes via the alignment-free bin-sum)."""
    dev = ctx_j.device
    xs = (torch.arange(g, device=dev) + 0.5) * (L / g)
    gx, gy = torch.meshgrid(xs, xs, indexing="ij"); pts = torch.stack([gx.reshape(-1), gy.reshape(-1)], -1)
    cell = (L / g) ** 2; G = pts.shape[0]; tot = torch.zeros(B, device=dev)
    with torch.no_grad():
        for c in range(0, G, chunk):
            blk = pts[c:c + chunk]; nb = blk.shape[0]
            cj = ctx_j[:, None, :].expand(B, nb, -1).reshape(B * nb, -1)
            oj = origin_j[:, None, :].expand(B, nb, -1).reshape(B * nb, 2)
            xj = blk[None].expand(B, nb, 2).reshape(B * nb, 2)
            tot = tot + K.site_logq(m, cj, oj, xj, arc, L).reshape(B, nb).exp().sum(1) * cell
    return tot


class _UniformHead(torch.nn.Module):
    """Stub nn.Module that returns zero logits (uniform softmax) for any input."""
    def __init__(self, nb): super().__init__(); self.nb = nb
    def forward(self, x): return torch.zeros(*x.shape[:-1], self.nb, device=x.device)

def test_fold_normalizes_with_injected_wrap_mass():
    """The trained model never proposes in the wrap region (0% mass), so the fold ships untested. Inject
    UNIFORM bin logits so the wrap-region bins (|center|>L/2/arc) carry real mass -> the fold MUST fire and
    must conserve mass (fold aliased bins onto their physical partner, no double-count) -> total over the
    physical torus == 1. (Uniform density -> any grid resolves it; the g=200 default is ample.)"""
    m, xo, so, ctx, origin, L, N = _ctx(_load_model()); arc = m._arc_scale(N); j = 7; nb = m.n_bins
    m.head_a = _UniformHead(nb)   # uniform over all bins (incl. wrap)
    m.head_b = _UniformHead(nb)
    total = _grid_total(m, ctx[:, j], origin[:, j], xo.shape[0], L, arc)
    assert torch.allclose(total, torch.ones_like(total), atol=3e-2), total


def test_zero_outside_range():
    """Genuine zero outside support (NOT clamped edge-bin mass). Shrink arc_range so the grid under-covers the
    torus (arc_range*arc=2.58 < L/2=4.56); then a query whose wrapped offset exceeds arc_range must get q=0
    (logq ~ log(1e-30) ~ -69), while an in-range query stays finite. Targeted check (no grid integral)."""
    m, xo, so, ctx, origin, L, N = _ctx(_load_model()); arc = m._arc_scale(N); j = 7
    m.arc_range = 1.2; m.bin_w = 2 * m.arc_range / m.n_bins
    oj = origin[:, j]
    inside = torch.remainder(oj + 0.3 * arc, L)                    # |offset| ~0.3 arc < 1.2 -> in range
    outside = torch.remainder(oj + 2.0 * arc, L)                   # |offset| ~2.0 arc > 1.2 -> out of range -> q=0
    assert (K.site_logq(m, ctx[:, j], oj, inside, arc, L) > -50).all()
    assert (K.site_logq(m, ctx[:, j], oj, outside, arc, L) < -60).all()


def test_sweeps_relax_energy_from_hot_start():
    m = _load_model(); N = 100
    ref = torch.load(f"{ART}/ka_reference_N100.pt", map_location=DEV, weights_only=False)
    s = ref["s"].to(DEV).long(); L = ref["L"]; sc = m.geo._scaffold(N, DEV); arc = m._arc_scale(N)
    B = 16; pos = torch.rand(B, N, 2, device=DEV) * L
    from liquid_coupling_flow.ka_energy import ka_energy
    U0 = ka_energy(pos, s, L).mean()
    g = torch.Generator(device=DEV).manual_seed(0)
    for _ in range(3):
        pos, nacc = K.learned_position_sweep(m, pos, s, sc, L, N, 0.5, arc, g)
    assert pos.min() >= 0 and pos.max() < L
    assert ka_energy(pos, s, L).mean() < U0, "learned sweep did not reduce energy"
    posu = torch.rand(B, N, 2, device=DEV) * L; U0u = ka_energy(posu, s, L).mean()
    for _ in range(3):
        posu, _ = K.uniform_position_sweep(posu, s, L, N, 0.5, 0.12, g)
    assert ka_energy(posu, s, L).mean() < U0u, "uniform sweep did not reduce energy"


def _pi_samples(N=5, B=4000, kT=0.5, n_eq=400):
    torch.manual_seed(1); L = (N / 1.2) ** 0.5
    s = torch.zeros(N, dtype=torch.long, device=DEV); s[:2] = 1; s = s[torch.randperm(N)]
    pos = torch.rand(B, N, 2, device=DEV) * L
    g = torch.Generator(device=DEV).manual_seed(2)
    for _ in range(n_eq):
        pos, _ = K.uniform_position_sweep(pos, s, L, N, kT, 0.25, g)
        pos, _ = K.swap_sweep(pos, s, L, kT, 1, g)
    return pos, s, L

@pytest.mark.parametrize("mode", ["position", "swap", "interleaved"])
def test_tinyN_detailed_balance_preserves_pi(mode):
    from liquid_coupling_flow.ka_energy import ka_energy
    pos, s, L = _pi_samples(); N = pos.shape[1]; kT = 0.5
    U_before = ka_energy(pos, s, L).mean().item()
    g = torch.Generator(device=DEV).manual_seed(3)
    for _ in range(60):                                           # apply the kernel under test
        if mode in ("position", "interleaved"):
            pos, _ = K.mock_position_sweep(pos, s, L, N, kT, 0.25, g)
        if mode in ("swap", "interleaved"):
            pos, _ = K.swap_sweep(pos, s, L, kT, 1, g)
    U_after = ka_energy(pos, s, L).mean().item()
    sem = ka_energy(pos, s, L).std().item() / pos.shape[0] ** 0.5
    assert abs(U_after - U_before) < 5 * sem, (mode, U_before, U_after, sem)


def test_context_independent_of_xj():
    m = _load_model(); N = 100; ref = torch.load(f"{ART}/ka_reference_N100.pt", map_location=DEV, weights_only=False)
    s = ref["s"].to(DEV).long(); pos = ref["x"][:4].to(DEV); L = ref["L"]; sc = m.geo._scaffold(N, DEV)
    sp = s.expand(4, N)
    ctx0, ori0 = m._local(pos, sp, sc, N=N, L=L)
    for j in (0, 1, 50, 99):
        p2 = pos.clone(); p2[:, j] = torch.remainder(p2[:, j] + torch.tensor([1.3, -0.7], device=DEV), L)
        ctx2, ori2 = m._local(p2, sp, sc, N=N, L=L)
        assert (ctx2[:, j] - ctx0[:, j]).abs().max() == 0, j
        assert (ori2[:, j] - ori0[:, j]).abs().max() == 0, j
        other = torch.ones(N, dtype=torch.bool, device=DEV); other[j] = False
        assert (ctx2[:, other] - ctx0[:, other]).abs().max() > 0, f"vacuous: no other slice changed at j={j}"
