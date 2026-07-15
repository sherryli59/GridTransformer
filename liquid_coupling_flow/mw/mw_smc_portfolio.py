"""lambda-bridge SMC with a scheduled collective-kernel portfolio (spec 2026-07-14).

pi_lambda proportional to q0_canonical^{1-lam} e^{-lam beta U}, lam: 0 -> 1. Storage is kept
canonical. Init: x ~ q0.sample (slot order), logw0 = logq0_canonical(x) - logq_sample(x) corrects
the (few-%) noncanonical-sample mismatch EXACTLY (proposal density in slot order vs the canonical
lift that defines pi_0). The logZ estimator mirrors mw_smc.smc_run's telescoping per-rung increment
(logsumexp(logw+dlw) - logsumexp(logw)); its baseline logsumexp(logw0) - log(B) folds the init IS
reweight so the total telescopes to logsumexp(final logw) - log(B).

Per-rung mutations are a lambda-scheduled mixture of the Task-5 collective kernels (suffix redraw,
two-blob union regen) plus single-site MH sweeps -- each pi_lambda-invariant. The harness is robust
to all-reject mutation rounds (empty accept-stats guarded; ESS/resample machinery is independent of
mutation acceptance).

Energy ledger: mw_energy / du_move / du_block self-increment the active EnergyEvalCounter inside
mw_energy.py, so wrapping the whole run in count_energy_evals() counts kernel energy work too --
no manual counter calls are needed here or in mw_kernels.py.

Per-rung persistence: the running result dict is saved after EVERY rung (CLAUDE.md durability).
"""
from __future__ import annotations
import math, os, time, torch

from liquid_coupling_flow.mw.mw_energy import mw_energy, count_energy_evals
from liquid_coupling_flow.mw.mw_smc import (ess, next_lambda, _resample, mutation_sweeps,
                                            LAM_FLOOR, STALL_RUNGS)
from liquid_coupling_flow.mw.mw_kernels import (recanonicalize, suffix_move, two_blob_move,
                                                conveyor_regions)

ART = os.path.join(os.path.dirname(__file__), "artifacts")


class CanonicalQ0Base:
    """mutation_sweeps-compatible base scoring the CANONICAL-lift q0 density.

    mw_base.GeneratorBase.log_q scores with preordered=True (labeled-space AR density); the bridge
    target here is the permutation-invariant CANONICAL lift (q0_model.log_prob(x, L) default
    preordered=False), so it is wrapped in a thin base with LOGQ_CONST=False. mutation_sweeps then
    evaluates base.log_q per site-move on the identical code path the uniform base certified.
    """
    LOGQ_CONST = False

    def __init__(self, model, N, L):
        self.model, self.N, self.L = model, N, L

    @torch.no_grad()
    def log_q(self, x, chunk=64):
        outs = [self.model.log_prob(x[i:i + chunk], self.L)
                for i in range(0, x.shape[0], chunk)]
        return torch.cat(outs, 0)


def smc_run_portfolio(q0_model, block_model, N, L, beta, B=64, ess_target=0.6,
                      n_site_sweeps=2, suffix_cfg=(2, 0.30, 24, 48),
                      twoblob_cfg=(2, 0.70, 6, 1.5), conveyor=False, m_frac=0.4,
                      step=0.15, seed=0, save_tag="port"):
    """Anneal lam: 0 -> 1 over pi_lambda ∝ q0_canonical^{1-lam} e^{-lam beta U} with an adaptive
    ESS-targeted schedule, resample-when-degenerate, and a lambda-scheduled kernel portfolio for
    re-equilibration. suffix_cfg = (moves_per_rung, lam_max, m_lo, m_hi); twoblob_cfg =
    (moves_per_rung, lam_max, K, min_sep). Returns {x, U, logw, logZ, history, evals, wall_s};
    saves the running dict to artifacts/mw_kportfolio_{save_tag}_N{N}.pt after every rung."""
    device = next(q0_model.parameters()).device
    gen = torch.Generator(device=device).manual_seed(seed)
    base = CanonicalQ0Base(q0_model, N, L)
    os.makedirs(ART, exist_ok=True)
    path = os.path.join(ART, f"mw_kportfolio_{save_tag}_N{N}.pt")
    regions = conveyor_regions(N, L, device, m_frac) if conveyor else None
    n_sfx, lam_sfx, m_lo, m_hi = suffix_cfg
    n_tb, lam_tb, K_tb, sep_tb = twoblob_cfg
    logB = math.log(float(B))

    t0 = time.time()
    with count_energy_evals() as counter:
        # ---- init: draw from q0 (slot order); exact reweight onto pi_0 = q0 canonical lift -----
        with torch.no_grad():
            x, lq_samp = q0_model.sample(B, N, L, gen=gen)     # slot-order proposal density
            lq0 = base.log_q(x)                                # canonical-lift density (== pi_0)
        x = recanonicalize(x, L)          # canonical storage; the lift is perm-invariant so lq0 stands
        U = mw_energy(x, L)
        logw = lq0 - lq_samp              # exact init reweight (noncanonicality correction)
        logZ = float(torch.logsumexp(logw, 0)) - logB   # baseline folds the init IS reweight
        lam, history = 0.0, []
        floor_streak, rung = 0, 0

        while lam < 1.0:
            # (1) reweight toward the next lambda rung (mirrors mw_smc.smc_run) --------------
            phi = -beta * U - lq0                              # d/dlam log pi_lambda
            lam_new = next_lambda(logw, phi, lam, ess_target, B)
            dlam = lam_new - lam
            dlw = dlam * phi
            logZ += float(torch.logsumexp(logw + dlw, 0) - torch.logsumexp(logw, 0))
            logw = logw + dlw
            lam = lam_new

            # stall guard: lam crawling at the bisection floor means the schedule is stuck
            if dlam <= LAM_FLOOR + 1e-9 and lam < 1.0:
                floor_streak += 1
            else:
                floor_streak = 0
            if floor_streak >= STALL_RUNGS:
                raise RuntimeError(
                    f"smc_run_portfolio stalled: lam advanced only by the {LAM_FLOOR} floor for "
                    f"{floor_streak} consecutive rungs (lam={lam:.6f}, ess={ess(logw):.2f}/{B}, "
                    f"U/N mean={float((U / N).mean()):.4f})")

            # (2) resample if degenerate (weights reset to zero) ----------------------------
            cur_ess = ess(logw)
            if cur_ess <= ess_target * B + 1e-6:
                x, U, lq0, logw = _resample(x, U, lq0, logw, gen)

            stats = {"rung": rung, "lam": lam, "ess": cur_ess, "dlam": dlam}

            # (3) lambda-scheduled portfolio mutations (each pi_lambda-invariant) ------------
            if lam < lam_sfx:
                x = recanonicalize(x, L)          # suffix_move ASSERTS canonical storage on entry
                accs = []
                for _ in range(n_sfx):
                    x, U, st = suffix_move(x, U, q0_model, m_lo, m_hi, lam, beta, L, gen)
                    accs.append(st["acc"])
                lq0 = base.log_q(x)               # refresh: suffix redraws the tail (lq0 changed)
                stats["suffix_acc"] = sum(accs) / max(1, len(accs))

            if lam < lam_tb:
                accs = []
                for _ in range(n_tb):
                    # two_blob_move maintains lq0 across accepts (relabel-free); lam==1 -> lq0=None
                    x, U, lq0, st = two_blob_move(x, U, lq0 if lam < 1.0 else None,
                                                  q0_model, block_model, K_tb, lam, beta,
                                                  L, sep_tb, gen, regions=regions)
                    accs.append(st["acc"])
                stats["twoblob_acc"] = sum(accs) / max(1, len(accs))

            # (4) single-site sweeps; returns the maintained (U, lq0) --------------------------
            x, U, lq0, mstats = mutation_sweeps(x, base, lam, beta, L, n_site_sweeps, step, gen)
            x = recanonicalize(x, L)             # canonical storage; lift-invariant so lq0 stands
            stats["site_acc"] = mstats["acc"]
            stats["cost_units"] = counter.as_dict()["cost_units"]

            history.append(stats)
            rung += 1
            result = {"x": x.cpu(), "U": U.cpu(), "logw": logw.cpu(),
                      "logZ": torch.tensor(logZ), "history": history,
                      "evals": counter.as_dict(), "wall_s": time.time() - t0}
            torch.save(result, path)             # per-rung persistence (CLAUDE.md durability)

    result = {"x": x, "U": U, "logw": logw, "logZ": torch.tensor(logZ), "history": history,
              "evals": counter.as_dict(), "wall_s": time.time() - t0}
    return result
