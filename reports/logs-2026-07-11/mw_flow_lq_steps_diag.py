"""Attribute the +16.7-nat logZ offset: reverse log_q at 40 vs 80 vs 160 steps on the same configs.
A systematic per-particle shift = discretization bias of the density (the SMC's reference measure);
if lq(80)-lq(40) ~ +0.2-0.3/particle * 64 ~ +13-19 nats, the offset is explained."""
import importlib.util, sys, torch
sys.path.insert(0, "/mnt/ssd/GridTransformer")
spec = importlib.util.spec_from_file_location(
    "mw_flow_base", "/mnt/ssd/GridTransformer/reports/logs-2026-07-11/mw_flow_base.py")
mfb = importlib.util.module_from_spec(spec); spec.loader.exec_module(mfb)
from liquid_coupling_flow.mw.mw_energy import RHO_STAR
DEV = "cuda"; N = 64; L = (N / RHO_STAR) ** (1 / 3)
CK = "liquid_coupling_flow/mw/artifacts/mw_ersi_N64_checkpoints/last.ckpt"
g = torch.Generator(device=DEV).manual_seed(3)
fb40 = mfb.FlowBase(CK, N, L, DEV, steps=40)
x = fb40.sample(64, g)
l40 = fb40.log_q(x)
for st in (80, 160):
    fb = mfb.FlowBase(CK, N, L, DEV, steps=st)
    l = fb.log_q(x)
    d = (l - l40)
    print(f"steps {st:3d} vs 40: mean shift {float(d.mean()):+8.2f} nats ({float(d.mean())/N:+.4f}/particle)  "
          f"std {float(d.std()):.2f}")
