"""Diagnose the rung-guard trip: is carried-vs-fresh log_q mismatch (i) nondeterministic kernels
amplified by the expanding reverse ODE, or (ii) genuine cross-batch content coupling (a bug)?
Measures: (1) same batch evaluated twice -> pure nondeterminism; (2) same config in different batch
CONTENT -> content coupling; (3) batch-size dependence (CuBLAS algo selection)."""
import importlib.util, sys, torch
sys.path.insert(0, "/mnt/ssd/GridTransformer")
spec = importlib.util.spec_from_file_location(
    "mw_flow_base", "/mnt/ssd/GridTransformer/reports/logs-2026-07-11/mw_flow_base.py")
mfb = importlib.util.module_from_spec(spec); spec.loader.exec_module(mfb)
from liquid_coupling_flow.mw.mw_energy import RHO_STAR

DEV = "cuda"
N = 64
L = (N / RHO_STAR) ** (1 / 3)
fb = mfb.FlowBase("liquid_coupling_flow/mw/artifacts/mw_ersi_N64_checkpoints/last.ckpt", N, L, DEV, steps=40)
g = torch.Generator(device=DEV).manual_seed(1)
x = fb.sample(64, g)

# (1) determinism: same batch twice
a = fb.log_q(x); b = fb.log_q(x)
print(f"(1) same batch twice:        max|d| {float((a-b).abs().max()):.2e}")

# (2) content coupling: replace second half of batch with OTHER configs; compare first half
x2 = x.clone()
x2[32:] = torch.remainder(x2[32:] + 0.5, L)   # drastically different content in slots 32..63
c = fb.log_q(x2)
print(f"(2) other-slot content swap: max|d| on kept half {float((a[:32]-c[:32]).abs().max()):.2e}")

# (3) batch size: evaluate first half alone
d = fb.log_q(x[:32])
print(f"(3) batch-size 64 vs 32:     max|d| on shared half {float((a[:32]-d).abs().max()):.2e}")

# (4) the actual failing pattern: one accepted move, fresh re-eval of the mixed batch
xp = x.clone()
xp[0] = torch.remainder(xp[0] + 0.0098 * torch.randn(N, 3, device=DEV, generator=g), L)
lq_prop = fb.log_q(xp)          # 'carried' value for config 0 computed in all-proposal batch
xmix = x.clone(); xmix[0] = xp[0]
lq_mix = fb.log_q(xmix)         # 'fresh' value in mixed batch
print(f"(4) carried-vs-fresh, 1 accepted cfg: |d| {float((lq_prop[0]-lq_mix[0]).abs()):.2e}  "
      f"(rejected cfgs: max|d| {float((a[1:]-lq_mix[1:]).abs().max()):.2e})")
