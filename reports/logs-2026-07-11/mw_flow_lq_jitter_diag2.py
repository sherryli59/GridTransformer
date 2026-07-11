"""Is log_q repeat-eval jitter STATE-DEPENDENT? Measure same-batch double-eval max|d| vs perturbation
off the flow manifold (the failing guard evaluates MUTATED configs, not flow samples)."""
import importlib.util, sys, torch
sys.path.insert(0, "/mnt/ssd/GridTransformer")
spec = importlib.util.spec_from_file_location(
    "mw_flow_base", "/mnt/ssd/GridTransformer/reports/logs-2026-07-11/mw_flow_base.py")
mfb = importlib.util.module_from_spec(spec); spec.loader.exec_module(mfb)
from liquid_coupling_flow.mw.mw_energy import RHO_STAR
DEV = "cuda"; N = 64; L = (N / RHO_STAR) ** (1 / 3)
fb = mfb.FlowBase("liquid_coupling_flow/mw/artifacts/mw_ersi_N64_checkpoints/last.ckpt", N, L, DEV, steps=40)
g = torch.Generator(device=DEV).manual_seed(1)
x = fb.sample(64, g)
print(f"{'eps':>6} {'jitter max|d|':>14} {'jitter med|d|':>14}")
for eps in (0.0, 0.005, 0.01, 0.02, 0.05):
    xp = torch.remainder(x + eps * torch.randn(x.shape, device=DEV, generator=g), L) if eps > 0 else x
    l1, l2, l3 = fb.log_q(xp), fb.log_q(xp), fb.log_q(xp)
    d = torch.stack([(l1-l2).abs(), (l1-l3).abs(), (l2-l3).abs()]).max(0).values
    print(f"{eps:>6.3f} {float(d.max()):>14.2e} {float(d.median()):>14.2e}")
