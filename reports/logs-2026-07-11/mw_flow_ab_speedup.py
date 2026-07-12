"""A+B DENSITY-SPEEDUP EXPERIMENT (N=64, B=256, paired seeds, matched energy budgets).

Baseline: seed-only warm ladder (the 3,072 se/cfg champion) — flow seeds, enter rung 7 of the
          12-rung ladder, 12 energy-only single-site sweeps/rung. Pure MCMC, no weights.
Arm A:    + ENTRY REWEIGHT at the natural rung (logw = -beta7*U - log q0, log q0 accumulated FREE
          during forward generation) -> resample (prunes left-tail seeds) -> exact SMC ladder
          (inter-rung dlw = -dbeta*U, resample at ESS<0.6B). Same mutation budget as baseline.
Arm B:    Arm A + 2 flow INDEPENDENCE proposals/rung (whole-config teleports, exact MH:
          log alpha = -beta_k (U'-U) + lq(x) - lq(x')) REPLACING 2 sweeps -> identical
          768 se/cfg/rung budget (1 full energy = N site-evals, MLIP accounting; ODEs free).
Diagnostic: acceptance-vs-beta curve of independence proposals on arm A's saved per-rung
          populations + the equilibrium bank at target beta (feasibility study for the
          beta-conditioned CRAFT flow).
Verdict metric: cumulative se/cfg to population mean U/N <= -1.60 (per-rung granularity),
          plus entry ESS, distinct-ancestor count, final shells. Everything saved per rung."""
import importlib.util, math, sys, time, torch
sys.path.insert(0, "/mnt/ssd/GridTransformer")
spec = importlib.util.spec_from_file_location(
    "mw_flow_base", "/mnt/ssd/GridTransformer/reports/logs-2026-07-11/mw_flow_base.py")
mfb = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mfb)
FlowBase = mfb.FlowBase
from liquid_coupling_flow.mw.mw_smc import mutation_sweeps, ess
from liquid_coupling_flow.mw.mw_base import UniformBase
from liquid_coupling_flow.mw.mw_energy import mw_energy, T_STAR, RHO_STAR
from liquid_coupling_flow.mw.mw_reference import g_r

DEV = "cuda"; N, B = 64, 256
L = (N / RHO_STAR) ** (1 / 3); beta_t = 1.0 / T_STAR
STEP = 0.0783; SW = 12; ENTRY = 7; THRESH = -1.60
LADDER = (0.5 * (beta_t / 0.5) ** torch.linspace(0, 1, 12)).tolist()
base = UniformBase(N, L)
fb = FlowBase("liquid_coupling_flow/mw/artifacts/mw_ersi_N64_checkpoints/last.ckpt", N, L, DEV, steps=40)
ART = "liquid_coupling_flow/mw/artifacts/mw_flow_ab_speedup.pt"
state = {"protocol": {"B": B, "ladder": LADDER, "entry": ENTRY, "sweeps": SW, "thresh": THRESH}}


def save():
    torch.save(state, ART)


@torch.no_grad()
def sample_fwd_logq(Btot, seed, chunk=64):
    """Forward generation WITH free log q0 accumulation (same midpoint div as the reverse pass)."""
    xs, ls = [], []
    a = torch.zeros(chunk, N, dtype=torch.long, device=DEV)
    ts = torch.linspace(0, 1, fb.steps + 1, device=DEV)
    for ci in range(0, Btot, chunk):
        g = torch.Generator(device=DEV).manual_seed(seed + ci)
        x = torch.rand(chunk, N, 3, device=DEV, generator=g) * L
        lq = torch.full((chunk,), -N * 3 * math.log(L), device=DEV, dtype=torch.float64)
        for i in range(fb.steps):
            dt = float(ts[i + 1] - ts[i])
            v1, _ = fb.e.forward_and_divergence(x, ts[i].expand(chunk), a)
            xm = torch.remainder(x + 0.5 * dt * v1, L)
            vm, dm = fb.e.forward_and_divergence(xm, (ts[i] + 0.5 * dt).expand(chunk), a)
            x = torch.remainder(x + dt * vm, L)
            lq = lq - dt * dm.double()
        xs.append(x); ls.append(lq.float())
    return torch.cat(xs), torch.cat(ls)


def lq_reverse(x, chunk=64):
    return torch.cat([fb.log_q(x[i:i + chunk]) for i in range(0, x.shape[0], chunk)])


def resample(x, U, lq, logw, gen):
    idx = torch.multinomial(torch.softmax(logw.double(), 0), B, replacement=True, generator=gen)
    return x[idx], U[idx], (lq[idx] if lq is not None else None), torch.zeros(B, device=DEV), idx


print("generating shared seeds (forward logq accumulated free)...", flush=True)
fx, flq = sample_fwd_logq(B, seed=0)
fU = mw_energy(fx, L)
print(f"seeds: U/N {float(fU.mean())/N:+.4f}  lq mean {float(flq.mean()):.1f}", flush=True)


def run_arm(tag, use_entry_weights, n_ind):
    """n_ind independence proposals/rung replace n_ind sweeps (budget-matched)."""
    gen = torch.Generator(device=DEV).manual_seed(42)
    x, U, lq = fx.clone(), fU.clone(), flq.clone()
    logw = torch.zeros(B, device=DEV)
    se = 0; hist = []; crossed = None; snaps = {}
    if use_entry_weights:
        logw = -LADDER[ENTRY] * U - lq
        e0 = ess(logw)
        x, U, lq, logw, idx = resample(x, U, lq, logw, gen)
        anc = int(torch.unique(idx).numel())
        print(f"[{tag}] entry ESS {e0:.1f}/{B}  distinct ancestors after resample {anc}/{B}", flush=True)
        hist.append({"event": "entry", "ess": float(e0), "ancestors": anc})
    lq_valid = use_entry_weights                     # lq of population is current only until MCMC moves
    for k in range(ENTRY, 12):
        bk = LADDER[k]
        if k > ENTRY and use_entry_weights:          # exact inter-rung reweight (pure-Boltzmann path)
            logw = logw - (bk - LADDER[k - 1]) * U
            ek = ess(logw)
            if ek <= 0.6 * B:
                x, U, lq, logw, _ = resample(x, U, lq if lq_valid else None, logw, gen)
        acc_ind = None
        if n_ind > 0:
            if not lq_valid:
                lq = lq_reverse(x); lq_valid = True   # free (neural); needed for exact MH ratio
            px, plq = sample_fwd_logq(B, seed=1000 * k)
            pU = mw_energy(px, L)
            n_acc = 0
            for r in range(n_ind):
                if r > 0:
                    px, plq = sample_fwd_logq(B, seed=1000 * k + r)
                    pU = mw_energy(px, L)
                se += N                               # one full energy per proposal round
                la = -bk * (pU - U) + (lq - plq)
                a = torch.log(torch.rand(B, device=DEV, generator=gen).clamp_min(1e-38)) < la
                x = torch.where(a[:, None, None], px, x)
                U = torch.where(a, pU, U)
                lq = torch.where(a, plq, lq)
                n_acc += int(a.sum())
            acc_ind = n_acc / (n_ind * B)
        x, U, _, info = mutation_sweeps(x, base, 1.0, bk, L, SW - n_ind, STEP, gen)
        lq_valid = False
        se += info["evals"] // B
        um = float(U.mean()) / N
        if crossed is None and um <= THRESH:
            crossed = se
        hist.append({"rung": k, "beta": bk, "U_mean": um, "se_cfg": se, "acc_ind": acc_ind})
        snaps[k] = x.cpu().clone()
        print(f"[{tag}] rung {k:2d} beta {bk:6.2f}  U/N {um:+.4f}  se/cfg {se:>6,}"
              + (f"  acc_ind {acc_ind:.3f}" if acc_ind is not None else ""), flush=True)
        state[tag] = {"hist": hist, "crossed": crossed, "X_final": x.cpu()}
        save()
    r, gr = g_r(x.cpu(), L)
    i1, i2 = int(1.19 / (L / 2) * len(r)), int(1.85 / (L / 2) * len(r))
    state[tag]["shells"] = (float(gr[i1]), float(gr[i2]))
    save()
    return snaps


t0 = time.time()
run_arm("baseline", use_entry_weights=False, n_ind=0)
run_arm("armA", use_entry_weights=True, n_ind=0)
snapsB = run_arm("armB", use_entry_weights=True, n_ind=2)
print(f"arms done ({time.time()-t0:.0f}s)", flush=True)

# ---------- diagnostic: independence-acceptance vs beta (feasibility of the CRAFT refresher) ----------
print("\n=== acceptance-vs-beta diagnostic (arm-B rung populations + bank at target) ===", flush=True)
ext = torch.load("liquid_coupling_flow/mw/artifacts/mw_ref_N64_ext.pt", map_location="cpu", weights_only=False)
curve = []
pops = {k: snapsB[k].to(DEV) for k in snapsB}
pops["bank"] = ext["cfgs"][-B:].to(DEV)
for key, xp in pops.items():
    bk = beta_t if key == "bank" else LADDER[key]
    Up = mw_energy(xp, L)
    lqp = lq_reverse(xp)
    px, plq = sample_fwd_logq(B, seed=777)
    pU = mw_energy(px, L)
    la = (-bk * (pU - Up) + (lqp - plq)).double()
    acc = float(torch.minimum(torch.ones_like(la), la.exp()).mean())
    curve.append((key, float(bk), acc, float(la.median())))
    print(f"  pop {str(key):>5}  beta {bk:6.2f}  E[min(1,e^la)] {acc:.4f}  median logalpha {float(la.median()):+8.1f}", flush=True)
state["acc_curve"] = curve
save()

print("\n=== VERDICT (se/cfg to U/N <= -1.60; baseline champion was 3,072) ===", flush=True)
for tag in ("baseline", "armA", "armB"):
    c = state[tag]["crossed"]
    s1, s2 = state[tag]["shells"]
    print(f"  {tag:8s}: {c if c else 'not reached'}  final shells {s1:.2f}/{s2:.2f}", flush=True)
print("DONE", flush=True)
