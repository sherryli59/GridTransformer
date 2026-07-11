"""DECISIVE: exact autodiff log|det J| of the discrete forward sampling map vs the -integral(div)dt
the likelihood claims. A near-constant gap (predicted ~ -30 nats total, ~ -0.47/particle) would be
invisible to fwd/rev consistency, step-refinement, SNIS, and sampling -- but IS the logZ offset.
Mechanism suspect: kNN-switch discontinuities of the trimmed velocity field (surface terms missing
from the continuity-equation accumulation)."""
import importlib.util, math, sys, torch
sys.path.insert(0, "/mnt/ssd/GridTransformer")
spec = importlib.util.spec_from_file_location(
    "mw_flow_base", "/mnt/ssd/GridTransformer/reports/logs-2026-07-11/mw_flow_base.py")
mfb = importlib.util.module_from_spec(spec); spec.loader.exec_module(mfb)
from liquid_coupling_flow.mw.mw_energy import RHO_STAR

DEV = "cuda"; N = 64; L = (N / RHO_STAR) ** (1 / 3)
fb = mfb.FlowBase("liquid_coupling_flow/mw/artifacts/mw_ersi_N64_checkpoints/last.ckpt", N, L, DEV, steps=40)
STEPS = 40
ts = torch.linspace(0, 1, STEPS + 1, device=DEV)
B = 4
g = torch.Generator(device=DEV).manual_seed(21)
x0 = torch.rand(B, N, 3, device=DEV, generator=g) * L
a1 = torch.zeros(1, N, dtype=torch.long, device=DEV)


def step_map(xflat, i):
    """One midpoint step of the sampler for a SINGLE config, as a flat 3N->3N map (no wrap on the
    output so the map is differentiable; wrap is applied outside between steps -- measure-preserving)."""
    x = xflat.view(1, N, 3)
    dt = float(ts[i + 1] - ts[i])
    v1, _ = fb.e.forward_and_divergence(x, ts[i].expand(1), a1)
    xm = torch.remainder(x + 0.5 * dt * v1, L)
    vm, _ = fb.e.forward_and_divergence(xm, (ts[i] + 0.5 * dt).expand(1), a1)
    return (x + dt * vm).reshape(-1)


print(f"{'cfg':>4} {'sum logdetJ (exact)':>20} {'-int div dt (claimed)':>22} {'gap (exact-claimed)':>20}")
gaps = []
for b in range(B):
    x = x0[b:b+1].clone()
    ld_exact, ld_claim = 0.0, 0.0
    for i in range(STEPS):
        dt = float(ts[i + 1] - ts[i])
        # claimed increment: midpoint divergence (what FlowBase/sampling accumulate)
        with torch.no_grad():
            v1, _ = fb.e.forward_and_divergence(x, ts[i].expand(1), a1)
            xm = torch.remainder(x + 0.5 * dt * v1, L)
            _, dm = fb.e.forward_and_divergence(xm, (ts[i] + 0.5 * dt).expand(1), a1)
        ld_claim += dt * float(dm)
        # exact increment: autodiff Jacobian of the actual step map
        J = torch.autograd.functional.jacobian(lambda xf: step_map(xf, i), x.reshape(-1), vectorize=True)
        ld_exact += float(torch.linalg.slogdet(J)[1])
        with torch.no_grad():
            x = torch.remainder(x.view(1, N, 3) + dt * fb.e.forward_and_divergence(
                torch.remainder(x.view(1, N, 3) + 0.5 * dt * v1, L), (ts[i] + 0.5 * dt).expand(1), a1)[0], L)
    gaps.append(ld_exact - ld_claim)
    print(f"{b:>4} {ld_exact:>20.2f} {ld_claim:>22.2f} {gaps[-1]:>20.2f}", flush=True)
gm = sum(gaps) / len(gaps)
print(f"\nmean gap {gm:+.2f} nats ({gm/N:+.4f}/particle)   [logZ offset to explain: ~ +23.5 => "
      f"predicted gap ~ -23 to -30 if this is the bug]", flush=True)
