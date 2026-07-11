"""Forward-composed log q (accumulate -div along the FORWARD sampling trajectory) vs reverse log_q
on the SAME samples. mean(lq_fwd - lq_rev) ~ +16.7 would fully attribute the SMC logZ offset to
forward/reverse discrete-map inconsistency (initialization measure mismatch, logw=0 assumption)."""
import importlib.util, math, sys, torch
sys.path.insert(0, "/mnt/ssd/GridTransformer")
spec = importlib.util.spec_from_file_location(
    "mw_flow_base", "/mnt/ssd/GridTransformer/reports/logs-2026-07-11/mw_flow_base.py")
mfb = importlib.util.module_from_spec(spec); spec.loader.exec_module(mfb)
from liquid_coupling_flow.mw.mw_energy import RHO_STAR
DEV = "cuda"; N = 64; L = (N / RHO_STAR) ** (1 / 3)
fb = mfb.FlowBase("liquid_coupling_flow/mw/artifacts/mw_ersi_N64_checkpoints/last.ckpt", N, L, DEV, steps=40)

@torch.no_grad()
def sample_with_fwd_logq(B, seed):
    g = torch.Generator(device=DEV).manual_seed(seed)
    x = torch.rand(B, N, 3, device=DEV, generator=g) * L
    a = torch.zeros(B, N, dtype=torch.long, device=DEV)
    ts = torch.linspace(0, 1, fb.steps + 1, device=DEV)
    lq = torch.full((B,), -N * 3 * math.log(L), device=DEV, dtype=torch.float64)
    for i in range(fb.steps):
        dt = float(ts[i + 1] - ts[i])
        v1, _ = fb.e.forward_and_divergence(x, ts[i].expand(B), a)
        xm = torch.remainder(x + 0.5 * dt * v1, L)
        vm, dm = fb.e.forward_and_divergence(xm, (ts[i] + 0.5 * dt).expand(B), a)
        x = torch.remainder(x + dt * vm, L)
        lq = lq - dt * dm.double()          # same midpoint div as the reverse pass uses
    return x, lq.float()

x, lq_f = sample_with_fwd_logq(128, 7)
lq_r = torch.cat([fb.log_q(x[i:i+64]) for i in (0, 64)])
d = lq_f - lq_r
print(f"lq_fwd mean {float(lq_f.mean()):8.2f}   lq_rev mean {float(lq_r.mean()):8.2f}")
print(f"mean(lq_fwd - lq_rev) {float(d.mean()):+7.2f} nats ({float(d.mean())/N:+.4f}/particle)   "
      f"std {float(d.std()):.2f}   [SMC logZ offset to explain: +16.7]")
