"""PART B -- FULL-LIKELIHOOD geometric-path SMC with the eRSI flow as base, at N=64.

Path: pi_lam ∝ q0(x)^(1-lam) e^(-lam*beta*U(x)), lam 0->1 adaptive (ESS-target 0.6, certified
machinery: next_lambda/ess/_resample reused from mw_smc). log q0 appears in EVERY incremental weight
(dlw = dlam*(-beta*U - lq)) -- the maximal exact use of the flow likelihood.

Mutation: pi_lam-invariant GLOBAL random-walk MH (whole-config proposal). One flow log_q + one full
energy per proposal -- this is what makes a q0-dependent bath affordable (single-site moves would
need one full reverse-ODE per particle move: 5.7 s/move, dead end). sigma adapted between rungs
toward acceptance ~0.25 (kernel stays exactly pi_lam-invariant: sigma fixed within a rung).

Cost accounting (MLIP-fair site-evals, per config): du_move = 1; FULL-config energy = N (an MLIP
evaluates all N atomic environments). Baselines from the N=64 3-arm seed-anneal (B=256, per config):
uniform 9216 / exclvol 6912 / flow-seeded 3072 site-evals to U/N <= -1.60.

Exactness anchors:
  (a) lam=0 invariance test: global moves with acceptance = dlog q0 only must preserve the flow
      population (U/N + mean lq stationary).
  (b) lam=1 invariance test: energy-only global moves must preserve an equilibrated reference
      population (from mw_ref_N64_ext).
  (c) per-rung guard: carried lq/U vs fresh re-eval (full-batch lq -- CuBLAS batch-size coupling).
  (d) logZ vs the CERTIFIED uniform-path value 1198.1 +/- 0.4 (mw_smc_g2_s0..2_N64): two independent
      paths to the same Z_beta; the offset measures the flow density's log-normalization error
      (the honest G-b).
Pre-registered verdict metric: cumulative per-config site-evals to U/N <= -1.60 vs 3072 (seed-only).
"""
import importlib.util, math, os, sys, time, torch

sys.path.insert(0, "/mnt/ssd/GridTransformer")
spec = importlib.util.spec_from_file_location(
    "mw_flow_base", "/mnt/ssd/GridTransformer/reports/logs-2026-07-11/mw_flow_base.py")
mfb = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mfb)
FlowBase = mfb.FlowBase
from liquid_coupling_flow.mw.mw_smc import next_lambda, ess, _resample, mutation_sweeps, LAM_FLOOR
from liquid_coupling_flow.mw.mw_base import UniformBase
from liquid_coupling_flow.mw.mw_energy import mw_energy, T_STAR, RHO_STAR

DEV = "cuda"
N, B = 64, 64
L = (N / RHO_STAR) ** (1 / 3)
beta = 1.0 / T_STAR
CKPT = "liquid_coupling_flow/mw/artifacts/mw_ersi_N64_checkpoints/last.ckpt"
ART = "liquid_coupling_flow/mw/artifacts/mw_flow_smc_geo_N64.pt"
N_GLOBAL = 8          # global moves per rung
MAX_RUNGS = 150
THRESH = -1.60        # arm-S comparison threshold (U/N)
LOGZ_REF = 1198.1     # certified uniform-path logZ (3 seeds: 1198.08/1198.50/1197.69)

fb = FlowBase(CKPT, N, L, DEV, steps=40)
state = {"N": N, "B": B, "protocol": {"n_global": N_GLOBAL, "ess_target": 0.6, "steps_ode": 40,
                                      "thresh": THRESH, "logz_ref": LOGZ_REF}}


def save():
    torch.save(state, ART)


@torch.no_grad()
def global_moves(x, U, lq, lam, sigma, n_moves, gen):
    """pi_lam-invariant whole-config RW-MH. Returns (x, U, lq, acceptance, per-config site-evals)."""
    Bc = x.shape[0]
    acc_n, se = 0, 0
    for _ in range(n_moves):
        prop = torch.remainder(x + sigma * torch.randn(x.shape, device=x.device, generator=gen), L)
        U_p = mw_energy(prop, L)
        se += N                                     # full-config energy = N site-evals per config
        lq_p = fb.log_q(prop)                       # neural: free in MLIP accounting
        loga = -lam * beta * (U_p - U) + (1.0 - lam) * (lq_p - lq)
        acc = torch.log(torch.rand(Bc, device=x.device, generator=gen).clamp_min(1e-38)) < loga
        x = torch.where(acc[:, None, None], prop, x)
        U = torch.where(acc, U_p, U)
        lq = torch.where(acc, lq_p, lq)
        acc_n += int(acc.sum())
    return x, U, lq, acc_n / max(1, n_moves * Bc), se


# ---------- exactness anchors (a) + (b) ----------
gen = torch.Generator(device=DEV).manual_seed(0)
print("=== invariance tests ===", flush=True)
x0 = fb.sample(B, gen)
U0, lq0 = mw_energy(x0, L), fb.log_q(x0)
u_before, lq_before = float(U0.mean()) / N, float(lq0.mean())
xt, Ut, lqt, acc0, _ = global_moves(x0, U0, lq0, lam=0.0, sigma=0.02, n_moves=30, gen=gen)
u_after, lq_after = float(Ut.mean()) / N, float(lqt.mean())
print(f"(a) lam=0 q0-invariance: U/N {u_before:+.4f} -> {u_after:+.4f}  mean-lq {lq_before:.1f} -> "
      f"{lq_after:.1f}  acc {acc0:.2f}", flush=True)
assert abs(u_after - u_before) < 0.05 and abs(lq_after - lq_before) < 5.0, "lam=0 invariance FAILED"

ext = torch.load("liquid_coupling_flow/mw/artifacts/mw_ref_N64_ext.pt", map_location="cpu", weights_only=False)
xr = ext["cfgs"][-B:].to(DEV)
Ur, lqr = mw_energy(xr, L), fb.log_q(xr)
u_before = float(Ur.mean()) / N
xr2, Ur2, _, acc1, _ = global_moves(xr, Ur, lqr, lam=1.0, sigma=0.0783 / math.sqrt(N), n_moves=30, gen=gen)
u_after = float(Ur2.mean()) / N
print(f"(b) lam=1 eq-invariance: U/N {u_before:+.4f} -> {u_after:+.4f}  acc {acc1:.2f}", flush=True)
assert abs(u_after - u_before) < 0.02, "lam=1 invariance FAILED"
state["tests"] = {"lam0": (u_before, u_after, acc0), "lam1_acc": acc1}
save()

# ---------- the geometric-path SMC ----------
print("\n=== geometric-path SMC (flow base, global moves) ===", flush=True)
gen = torch.Generator(device=DEV).manual_seed(1)
x = fb.sample(B, gen)
U = mw_energy(x, L)
lq = fb.log_q(x)
logw = torch.zeros(B, device=DEV)
lam, logZ, rung, floor_streak = 0.0, 0.0, 0, 0
se_cum = N            # initial full-energy eval
sigma = 0.0783 / math.sqrt(N)
hist = []
crossed = None
t0 = time.time()
print(f"seed pop: U/N {float(U.mean())/N:+.4f}  lq mean {float(lq.mean()):.1f}", flush=True)

while lam < 1.0 and rung < MAX_RUNGS:
    phi = -beta * U - lq
    lam_new = next_lambda(logw, phi, lam, 0.6, B)
    dlam = lam_new - lam
    dlw = dlam * phi
    logZ += float(torch.logsumexp(logw + dlw, 0) - torch.logsumexp(logw, 0))
    logw = logw + dlw
    lam = lam_new
    floor_streak = floor_streak + 1 if (dlam <= LAM_FLOOR + 1e-9 and lam < 1.0) else 0
    if floor_streak >= 10:
        print(f"STALLED: lam floor for {floor_streak} rungs at lam={lam:.5f}", flush=True)
        state["status"] = "stalled"
        break

    cur_ess = ess(logw)
    resampled = cur_ess <= 0.6 * B + 1e-6
    if resampled:
        x, U, lq, logw = _resample(x, U, lq, logw, gen)

    x, U, lq, acc, se = global_moves(x, U, lq, lam, sigma, N_GLOBAL, gen)
    se_cum += se

    # rung guard (c): carried vs fresh (full-batch lq; U on [:8], relative tol). Tolerance sits above
    # the MEASURED reverse-ODE nondeterminism floor (same-batch repeat eval: max|d| 1.1e-3, tail 5.6e-3
    # across B=64 — scatter-add atomics amplified by the expanding reverse trajectory; diag 2026-07-11)
    # and far below O(1)-nat bookkeeping errors the guard exists to catch.
    lq_fresh = fb.log_q(x)
    assert float((lq - lq_fresh).abs().max()) < 5e-2, "carried log_q drifted"
    U_fresh = mw_energy(x[:8], L)
    assert float((U[:8] - U_fresh).abs().max()) < 1e-4 * max(1.0, float(U_fresh.abs().max())), "carried U drifted"

    um = float(U.mean()) / N
    hist.append({"rung": rung, "lam": lam, "dlam": dlam, "ess": cur_ess, "resampled": resampled,
                 "acc": acc, "sigma": sigma, "U_mean": um, "U_std": float((U / N).std()),
                 "lq_mean": float(lq.mean()), "se_cum": se_cum, "logZ": logZ})
    if crossed is None and lam >= 1.0 and um <= THRESH:
        crossed = se_cum
    print(f"rung {rung:3d} lam {lam:.4f} (d {dlam:.4f}) ess {cur_ess:5.1f}{' R' if resampled else '  '} "
          f"acc {acc:.2f} sig {sigma:.4f}  U/N {um:+.4f}  lq {float(lq.mean()):7.1f}  "
          f"se/cfg {se_cum:>7,}  logZ {logZ:8.2f}  ({time.time()-t0:.0f}s)", flush=True)
    # between-rung sigma adaptation toward acc ~0.25
    if acc < 0.15:
        sigma = max(sigma * 0.7, 1e-3)
    elif acc > 0.35:
        sigma = min(sigma * 1.3, 0.05)
    rung += 1
    state.update({"hist": hist, "x": x.cpu(), "U": U.cpu(), "logw": logw.cpu(), "logZ": logZ,
                  "se_cum": se_cum, "lam": lam})
    save()

print(f"\nlam-path done: lam {lam:.4f}  rungs {rung}  logZ {logZ:.2f} (certified uniform-path "
      f"{LOGZ_REF} +/- 0.4; offset = flow log-normalization error)", flush=True)

# ---------- lam=1 finisher: cheap exact single-site energy-only sweeps ----------
ub = UniformBase(N, L)
fin = []
for sw in range(60):
    x, U, _, info = mutation_sweeps(x, ub, 1.0, beta, L, 1, 0.0783, gen)
    se_cum += info["evals"] // B
    um = float(U.mean()) / N
    fin.append({"sweep": sw, "U_mean": um, "se_cum": se_cum})
    if crossed is None and um <= THRESH:
        crossed = se_cum
        print(f"finisher sweep {sw:2d}: U/N {um:+.4f}  se/cfg {se_cum:,}  << crossed {THRESH}", flush=True)
    if sw % 10 == 9:
        print(f"finisher sweep {sw:2d}: U/N {um:+.4f}  se/cfg {se_cum:,}", flush=True)
        state.update({"fin": fin, "x": x.cpu(), "U": U.cpu(), "se_cum": se_cum, "crossed": crossed})
        save()
    if crossed is not None and um <= THRESH - 0.005:
        break

state.update({"fin": fin, "x": x.cpu(), "U": U.cpu(), "se_cum": se_cum, "crossed": crossed,
              "status": state.get("status", "done"), "wall": time.time() - t0})
save()

print("\n=== VERDICT: full-likelihood SMC vs seed-only anneal (per-config site-evals to "
      f"U/N <= {THRESH}) ===", flush=True)
print(f"  seed-only  (flow seeds, energy-only ladder): 3,072   [N=64 3-arm baseline]", flush=True)
if crossed is not None:
    print(f"  full-likelihood geometric SMC:               {crossed:,}", flush=True)
else:
    print(f"  full-likelihood geometric SMC:               NOT REACHED (final U/N {float(U.mean())/N:+.4f})",
          flush=True)
print(f"  (uniform baseline: 9,216 / exclvol: 6,912)", flush=True)
print(f"  logZ(flow path) {logZ:.2f} vs certified {LOGZ_REF} -> flow log-norm error "
      f"{logZ - LOGZ_REF:+.2f} nats", flush=True)
print("DONE", flush=True)
