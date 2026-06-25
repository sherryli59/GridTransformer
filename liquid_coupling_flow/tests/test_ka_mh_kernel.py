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

def test_density_normalizes_over_physical_cells():
    m, xo, so, ctx, origin, L, N = _ctx(_load_model())
    arc = m._arc_scale(N); j = 7
    # integrate q over a fine grid of PHYSICAL torus positions (post-fold) -> ~1
    g = 120; xs = (torch.arange(g, device=DEV) + 0.5) * (L / g)
    gx, gy = torch.meshgrid(xs, xs, indexing="ij")
    pts = torch.stack([gx.reshape(-1), gy.reshape(-1)], -1)        # [g*g,2]
    cell = (L / g) ** 2
    lq = torch.stack([K.site_logq(m, ctx[:, j], origin[:, j], pts[k].expand(xo.shape[0], 2), arc, L)
                      for k in range(pts.shape[0])], 0)            # [g*g, B]
    total = (lq.exp().sum(0) * cell)                               # [B] ~ 1
    assert torch.allclose(total, torch.ones_like(total), atol=2e-2), total

def test_sampler_evaluator_roundtrip():
    m, xo, so, ctx, origin, L, N = _ctx(_load_model())
    arc = m._arc_scale(N); j = 7
    xj_new, logq_new = K.site_propose(m, ctx[:, j], origin[:, j], arc, L)
    logq_new_eval = K.site_logq(m, ctx[:, j], origin[:, j], xj_new, arc, L)
    assert torch.allclose(logq_new, logq_new_eval, atol=1e-4), (logq_new - logq_new_eval).abs().max()
