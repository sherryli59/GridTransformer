"""Annealed lambda-path SMC for mW: pi_lambda ∝ q0^{1-lambda} (e^{-beta*U})^lambda, lambda: 0->1.
Adaptive lambda schedule (bisection to an ESS target) + multinomial resampling + single-site MH
mutation sweeps that are pi_lambda-invariant. With a LOGQ_CONST base (uniform q0), log q0 is a
CONSTANT and must drop out of every mutation acceptance ratio — mutation_sweeps branches on
base.LOGQ_CONST so that reduction is exercised (and Spy-test certified) on the identical code path
that will later carry a neural base, before that base exists (Phase-2 fault isolation).
"""
from __future__ import annotations
import os, time, torch
from liquid_coupling_flow.mw.mw_energy import mw_energy, du_move

ART = os.path.join(os.path.dirname(__file__), "artifacts")
DEV = "cuda" if torch.cuda.is_available() else "cpu"

LAM_FLOOR = 1e-4
STALL_RUNGS = 20


def ess(logw):
    """Effective sample size of log-weights [B]. Softmax done in double precision — at float32,
    100 equal (zero) log-weights round-trip to ESS off by ~1.5e-5, past a 1e-6 exactness gate."""
    w = torch.softmax(logw.double(), 0)
    return float(1.0 / (w ** 2).sum())


def next_lambda(logw, phi, lam, ess_target, B):
    """Largest lam' in (lam,1] with ESS(logw + (lam'-lam)*phi) >= ess_target*B; bisection, floor 1e-4.
    phi = -beta*U - log q0 (the d/dlam of log pi_lambda)."""
    if ess(logw + (1.0 - lam) * phi) >= ess_target * B:
        return 1.0
    lo, hi = lam, 1.0
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        (lo, hi) = (mid, hi) if ess(logw + (mid - lam) * phi) >= ess_target * B else (lo, mid)
    return min(1.0, max(lo, lam + LAM_FLOOR))


def _resample(x, U, lq, logw, gen):
    idx = torch.multinomial(torch.softmax(logw, 0), logw.shape[0], replacement=True, generator=gen)
    return x[idx], U[idx], lq[idx], torch.zeros_like(logw)


def mutation_sweeps(x, base, lam, beta, L, n_sweeps, step, gen):
    """pi_lambda-invariant single-site MH. Uniform base: log q0 terms drop (LOGQ_CONST)."""
    B, N, _ = x.shape
    U = mw_energy(x, L)
    lq = None if base.LOGQ_CONST else base.log_q(x)
    n_acc = 0; n_evals = 0
    for _ in range(n_sweeps):
        for i in range(N):
            xi = torch.remainder(x[:, i] + step * torch.randn(B, 3, device=x.device, generator=gen), L)
            dU = du_move(x, i, xi, L); n_evals += B
            loga = -lam * beta * dU
            if not base.LOGQ_CONST:
                x_prop = x.clone(); x_prop[:, i] = xi
                lq_new = base.log_q(x_prop)
                loga = loga + (1.0 - lam) * (lq_new - lq)
            acc = torch.log(torch.rand(B, device=x.device, generator=gen).clamp_min(1e-38)) < loga
            x = x.clone(); x[acc, i] = xi[acc]
            U = torch.where(acc, U + dU, U)
            if not base.LOGQ_CONST:
                lq = torch.where(acc, lq_new, lq)
            n_acc += int(acc.sum())
    if base.LOGQ_CONST:
        lq = base.log_q(x)
    return x, U, lq, {"acc": n_acc / max(1, n_sweeps * N * B), "evals": n_evals}


def smc_run(base, N, L, beta, B=256, ess_target=0.6, n_sweeps=3, step=None, seed=0, save_tag=""):
    """Anneal lambda: 0 -> 1 with adaptive ESS-targeted steps, resample-when-degenerate, mutate to
    re-equilibrate at the new lambda. Returns x, U, logw, logZ, history, evals, wall; saves the
    running result dict to artifacts/mw_smc{save_tag}_N{N}.pt after every rung."""
    if step is None:
        step = 0.15
    os.makedirs(ART, exist_ok=True)
    path = os.path.join(ART, f"mw_smc{save_tag}_N{N}.pt")

    t0 = time.time()
    device = DEV
    gen = torch.Generator(device=device).manual_seed(seed)

    x = base.sample(B, gen)
    U = mw_energy(x, L)
    lq = base.log_q(x)
    logw = torch.zeros(B, device=x.device)
    lam = 0.0
    logZ = 0.0
    evals = B
    history = []
    floor_streak = 0
    rung = 0

    while lam < 1.0:
        # (1) reweight toward the next lambda rung
        phi = -beta * U - lq
        lam_new = next_lambda(logw, phi, lam, ess_target, B)
        dlam = lam_new - lam
        dlw = dlam * phi
        logZ += float(torch.logsumexp(logw + dlw, 0) - torch.logsumexp(logw, 0))
        logw = logw + dlw
        lam = lam_new

        # stall guard: lambda crawling at the bisection floor means the schedule is not progressing
        if dlam <= LAM_FLOOR + 1e-9 and lam < 1.0:
            floor_streak += 1
        else:
            floor_streak = 0
        if floor_streak >= STALL_RUNGS:
            cur_ess = ess(logw)
            raise RuntimeError(
                f"smc_run stalled: lam advanced only by the {LAM_FLOOR} floor for "
                f"{floor_streak} consecutive rungs (lam={lam:.6f}, ess={cur_ess:.2f}/{B}, "
                f"U/N mean={float((U / N).mean()):.4f}, U/N std={float((U / N).std()):.4f})"
            )

        # (2) resample if degenerate (weights reset to zero)
        cur_ess = ess(logw)
        if cur_ess <= ess_target * B + 1e-6:
            x, U, lq, logw = _resample(x, U, lq, logw, gen)

        # (3) mutate to re-equilibrate at the new lambda
        x, U, lq, info = mutation_sweeps(x, base, lam, beta, L, n_sweeps, step, gen)
        evals += info["evals"]

        # (4) rung guard: carried U/lq must equal a fresh re-evaluation (weight-path exactness)
        lq_fresh = base.log_q(x[:8])
        assert float((lq[:8] - lq_fresh).abs().max()) < 1e-3, "carried log_q drifted from fresh re-eval"
        U_fresh = mw_energy(x[:8], L)
        assert float((U[:8] - U_fresh).abs().max()) < 1e-4, "carried U drifted from fresh re-eval"

        # (5) record + per-rung save
        history.append({
            "rung": rung, "lam": lam, "ess": cur_ess,
            "U_mean": float((U / N).mean()), "dlam": dlam, "evals": info["evals"],
        })
        rung += 1
        result = {"x": x, "U": U, "logw": logw, "logZ": torch.tensor(logZ), "history": history,
                  "evals": evals, "wall": time.time() - t0}
        torch.save(result, path)

    result["wall"] = time.time() - t0
    torch.save(result, path)
    return result
