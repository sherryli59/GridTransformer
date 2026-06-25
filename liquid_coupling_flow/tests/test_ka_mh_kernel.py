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
