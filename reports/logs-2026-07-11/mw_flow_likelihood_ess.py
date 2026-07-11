"""PART A -- does the flow's LIKELIHOOD help, and how does it scale? Measure the one-shot importance-
weight ESS of the flow as an IS distribution targeting e^{-beta U}, at N=64/216/512. logw = -beta U -
log q0(flow); ESS(logw)/B is the fraction of the population the likelihood keeps. This is the raw
'likelihood as weights' quality; if it collapses ~exp(-N*delta) with size, the flow's DENSITY is
IS-cursed at scale (independent of its SAMPLE quality, which we separately showed transfers flat).
Also reports SNIS <U>/N per size (constant-Z-invariant estimator) as a correctness cross-check."""
import importlib.util, sys, torch
sys.path.insert(0, "/mnt/ssd/GridTransformer")
spec = importlib.util.spec_from_file_location(
    "mw_flow_base", "/mnt/ssd/GridTransformer/reports/logs-2026-07-11/mw_flow_base.py")
mfb = importlib.util.module_from_spec(spec); spec.loader.exec_module(mfb)
FlowBase = mfb.FlowBase
from liquid_coupling_flow.mw.mw_energy import mw_energy_chunked, T_STAR, RHO_STAR

DEV = "cuda"
CK = "liquid_coupling_flow/mw/artifacts/mw_ersi_N64_checkpoints/last.ckpt"
beta = 1.0 / T_STAR
REF = -1.627
B = 256
print(f"beta={beta:.3f}  B={B}  (flow trained @ N=64)\n", flush=True)
print(f"{'N':>5} {'log q0 mean':>12} {'raw U/N':>9} {'SNIS U/N':>9} {'ESS':>10} {'ESS/B':>8}  {'-log(ESS/B)/N':>13}", flush=True)
rows = []
for N in (64, 216, 512):
    L = (N / RHO_STAR) ** (1 / 3)
    fb = FlowBase(CK, N, L, DEV, steps=40)
    g = torch.Generator(device=DEV).manual_seed(0)
    # chunk generation to fit memory at N=512
    xs = []
    for c in range(0, B, 64):
        gc = torch.Generator(device=DEV).manual_seed(c)
        xs.append(fb.sample(min(64, B - c), gc))
    x = torch.cat(xs)
    lq = torch.cat([fb.log_q(x[c:c + 64]) for c in range(0, B, 64)]).double()
    U = mw_energy_chunked(x, L).double()
    logw = -beta * U - lq
    w = torch.softmax(logw, 0)
    ess = float(1.0 / (w ** 2).sum())
    u_snis = float((w * U).sum()) / N
    u_raw = float(U.mean()) / N
    per_particle = -torch.log(torch.tensor(ess / B)).item() / N
    print(f"{N:>5} {float(lq.mean()):>12.1f} {u_raw:>9.3f} {u_snis:>9.3f} {ess:>7.2f}/{B} {ess/B:>8.4f}  {per_particle:>13.4f}", flush=True)
    rows.append({"N": N, "lq_mean": float(lq.mean()), "u_raw": u_raw, "u_snis": u_snis,
                 "ess": ess, "ess_frac": ess / B, "per_particle_kl_proxy": per_particle})
    del fb, x, lq, U, logw, w
    torch.cuda.empty_cache()
torch.save({"rows": rows, "beta": beta, "B": B},
           "liquid_coupling_flow/mw/artifacts/mw_flow_likelihood_ess.pt")
print("\nREADING: ESS/B ~ exp(-N*delta). If -log(ESS/B)/N is ~constant across N, the per-particle KL"
      " gap delta is size-invariant and the weight ESS decays EXACTLY exponentially in N -> the flow"
      " LIKELIHOOD (as IS weights) is cursed at scale, even though its SAMPLES transfer flat.", flush=True)
