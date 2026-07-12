"""Task 1 exactness gate: 3D isolated (non-periodic) CavityCondEGNN -- analytic per-particle divergence
(movers only) vs a brute-force autograd trace of the velocity Jacobian. Double precision, n_species=2,
random 3D cloud of k movers + cage (cage is fixed context, not in the divergence). Assert max abs diff < 1e-6.
See docs/superpowers/specs/2026-07-12-egnn-cavity-block-flow-corrector-design.md (Task 1)."""
import torch

from liquid_coupling_flow.ka3d_cavity_egnn import CavityCondEGNN

torch.manual_seed(0)
dev = "cpu"
B, k, n_cage = 3, 6, 14
P = k + n_cage
r_c = 2.5

model = CavityCondEGNN(n_cage=n_cage, k=k, r_c=r_c, hidden_nf=32, n_layers=3, n_species=2,
                        max_neighbors=10).to(dev).double()


def sample_cloud(gen):
    # movers: a compact cluster near the origin (the cavity block).
    mov = torch.randn(B, k, 3, generator=gen, dtype=torch.float64) * 0.8
    # cage: a shell around the movers (boundary + retained context) -- reaches within cutoff r_c.
    dirs = torch.randn(B, n_cage, 3, generator=gen, dtype=torch.float64)
    dirs = dirs / dirs.norm(dim=-1, keepdim=True)
    rad = 1.6 + 1.2 * torch.rand(B, n_cage, generator=gen, dtype=torch.float64)
    cage = dirs * rad[..., None]
    return torch.cat([mov, cage], dim=1)


gen = torch.Generator().manual_seed(1)
cloud = sample_cloud(gen).requires_grad_(True)
sp = torch.randint(0, 2, (B, P), generator=gen)
t = torch.full((B,), 0.42, dtype=torch.float64)

vel, div = model.vel_div(cloud, t, sp, k)          # vel [B,k,3], div [B]  (differentiable: grad is enabled here)
assert vel.shape == (B, k, 3), vel.shape
assert div.shape == (B,), div.shape

div_brute = torch.zeros(B, dtype=torch.float64)
for i in range(k):
    for d in range(3):
        grad = torch.autograd.grad(vel[:, i, d].sum(), cloud, retain_graph=True)[0]
        div_brute = div_brute + grad[:, i, d]        # d vel[:,i,d] / d cloud[:,i,d], summed over movers*dims

err = (div - div_brute).abs()
print(f"analytic div:      {div.detach().tolist()}")
print(f"brute-force div:   {div_brute.detach().tolist()}")
print(f"max abs diff:      {float(err.max()):.3e}")
print(f"mean abs diff:     {float(err.mean()):.3e}")
ok = float(err.max()) < 1e-6
print("PASS" if ok else "FAIL")
assert ok, f"exactness FAILED: max abs diff {float(err.max()):.3e} >= 1e-6"
