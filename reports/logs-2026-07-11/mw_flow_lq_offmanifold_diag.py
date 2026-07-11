"""Is the reverse-ODE lq step-converged OFF the flow manifold? All prior convergence checks used the
flow's OWN samples (on-manifold). The SMC population anneals toward Boltzmann configs (off-manifold);
if lq there is step-UNconverged or systematically shifted, that state-dependent density error is the
mechanism behind the +23.5-nat logZ inflation (phi = -beta*U - lq inflates where lq undershoots)."""
import importlib.util, sys, torch
sys.path.insert(0, "/mnt/ssd/GridTransformer")
spec = importlib.util.spec_from_file_location(
    "mw_flow_base", "/mnt/ssd/GridTransformer/reports/logs-2026-07-11/mw_flow_base.py")
mfb = importlib.util.module_from_spec(spec); spec.loader.exec_module(mfb)
from liquid_coupling_flow.mw.mw_energy import RHO_STAR
DEV = "cuda"; N = 64; L = (N / RHO_STAR) ** (1 / 3)
CK = "liquid_coupling_flow/mw/artifacts/mw_ersi_N64_checkpoints/last.ckpt"
ext = torch.load("liquid_coupling_flow/mw/artifacts/mw_ref_N64_ext.pt", map_location="cpu", weights_only=False)
xb = ext["cfgs"][-64:].to(DEV)                       # equilibrated Boltzmann configs (OFF-manifold)
fb40 = mfb.FlowBase(CK, N, L, DEV, steps=40)
xf = fb40.sample(64, torch.Generator(device=DEV).manual_seed(9))   # ON-manifold control
for name, x in (("flow-samples (ON) ", xf), ("bank configs (OFF)", xb)):
    l40 = fb40.log_q(x)
    line = f"{name}: lq40 mean {float(l40.mean()):8.1f}"
    for st in (80, 160):
        fb = mfb.FlowBase(CK, N, L, DEV, steps=st)
        d = fb.log_q(x) - l40
        line += f"  | d{st} mean {float(d.mean()):+7.2f} std {float(d.std()):6.2f}"
    print(line, flush=True)
