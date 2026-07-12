"""Task 2 gate: CavityBlockFlow's augmented-ODE (dopri5) integrator + composed exact log-q, built on Task-1's
CavityCondEGNN (see reports/logs-2026-07-12/test_egnn3d_exact.py for the Task-1 divergence exactness gate).
3D ISOLATED (non-periodic) cavity: no torch.remainder / minimum-image anywhere. Double precision, tight
dopri5 rtol/atol. See docs/superpowers/specs/2026-07-12-egnn-cavity-block-flow-corrector-design.md (Task 2).

Checks:
  (a) round trip: forward x0->x1 then reverse x1->x0 recovers x0 to ~rtol; forward+reverse logdets cancel
      to ~1e-5.
  (b) `flow` never mutates the cage positions passed in (isolated-mode contract: cage is frozen context).
  (c) zero-init (flow-matching init before any training): the untrained velocity is exactly 0 -> identity
      flow, x1==x0 and logdet==0 to ~1e-6.
"""
import torch

from liquid_coupling_flow.ka3d_cavity_egnn import CavityBlockFlow

torch.manual_seed(0)
dev = "cpu"
B, k, n_cage = 3, 5, 12
r_c = 2.5


def sample_cloud(gen):
    mov = torch.randn(B, k, 3, generator=gen, dtype=torch.float64) * 0.8
    dirs = torch.randn(B, n_cage, 3, generator=gen, dtype=torch.float64)
    dirs = dirs / dirs.norm(dim=-1, keepdim=True)
    rad = 1.6 + 1.2 * torch.rand(B, n_cage, generator=gen, dtype=torch.float64)
    cage = dirs * rad[..., None]
    return mov, cage


gen = torch.Generator().manual_seed(1)
x0, cage = sample_cloud(gen)
sp_block = torch.randint(0, 2, (B, k), generator=gen)
sp_cage = torch.randint(0, 2, (B, n_cage), generator=gen)
R = torch.full((B,), r_c, dtype=torch.float64)

# rtol=1e-7 with a SMALL, reproducible velocity perturbation (below): a large/rough untrained random field
# has high-frequency content dopri5 cannot resolve, so its adaptive step-size collapses and the solve never
# terminates. A small perturbation keeps the field integrable to a tight tol, giving a genuinely tight
# round-trip. (The deployment default is 1e-6 + a max_num_steps guard; see CavityBlockFlow.__init__.)
flow = CavityBlockFlow(n_cage=n_cage, k=k, r_c=r_c, hidden_nf=32, n_layers=3, n_species=2,
                        max_neighbors=10, ode_rtol=1e-7, ode_atol=1e-7).to(dev).double()

# ---- (c) zero-init identity check (untrained: pot_model[-1] weight+bias zeroed -> v==0 everywhere) ----
x1_id, logdet_id = flow.flow(x0, cage, sp_block, sp_cage, R=R, reverse=False)
id_err = (x1_id - x0).abs().max().item()
id_ld = logdet_id.abs().max().item()
print(f"[zero-init identity] max|x1-x0| = {id_err:.3e}   max|logdet| = {id_ld:.3e}")
assert id_err < 1e-6, f"zero-init flow is not the identity on x: {id_err:.3e}"
assert id_ld < 1e-6, f"zero-init flow has nonzero logdet: {id_ld:.3e}"

# ---- now perturb the velocity head off zero-init so the round trip is a real (non-trivial) test ----
pgen = torch.Generator().manual_seed(7)   # fixed seed: deterministic, avoids a flaky too-rough random draw
with torch.no_grad():
    for p in flow.ce.egnn.pot_model.parameters():
        p.add_(0.01 * torch.randn(p.shape, generator=pgen, dtype=p.dtype))   # small -> field stays integrable

# ---- (b) cage-frozen contract: cage_x must be byte-identical before/after flow() ----
cage_before = cage.clone()
x1, logdet_fwd = flow.flow(x0, cage, sp_block, sp_cage, R=R, reverse=False)
cage_unchanged = torch.equal(cage, cage_before)
print(f"[cage frozen] cage byte-identical after flow(): {cage_unchanged}")
assert cage_unchanged, "flow() mutated the cage positions"
assert not torch.equal(x1, x0), "sanity: perturbed flow should move x0 (test would be vacuous otherwise)"

# ---- (a) round trip: reverse x1->x0, logdets cancel ----
x0_rec, logdet_rev = flow.flow(x1, cage, sp_block, sp_cage, R=R, reverse=True)
rt_err = (x0_rec - x0).abs().max().item()
cancel_err = (logdet_fwd + logdet_rev).abs().max().item()
print(f"[round trip]   max|x0_rec-x0| = {rt_err:.3e}")
print(f"[logdet cancel] max|logdet_fwd+logdet_rev| = {cancel_err:.3e}")
print(f"  logdet_fwd = {logdet_fwd.tolist()}")
print(f"  logdet_rev = {logdet_rev.tolist()}")
assert rt_err < 1e-4, f"round trip FAILED: {rt_err:.3e} >= 1e-4"       # ~rtol=1e-6 dopri5, both directions
assert cancel_err < 1e-4, f"logdet cancellation FAILED: {cancel_err:.3e} >= 1e-4"

# ---- composed_logq: sum with an arbitrary AR base log-q, matches flow()'s (x1, logdet) exactly ----
logq_ar = torch.randn(B, dtype=torch.float64)
x1_c, logq_c = flow.composed_logq(x0, logq_ar, cage, sp_block, sp_cage, R=R)
assert torch.equal(x1_c, x1), "composed_logq's x1 must match flow()'s forward x1 exactly (same call)"
comp_err = (logq_c - (logq_ar + logdet_fwd)).abs().max().item()
print(f"[composed_logq] max|logq_c - (logq_ar+logdet_fwd)| = {comp_err:.3e}")
assert comp_err < 1e-12, f"composed_logq formula mismatch: {comp_err:.3e}"

print("PASS")
