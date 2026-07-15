"""eRSI-seeded, one-q0-correction thermal SMC for mW.

The eRSI forward pass jointly produces initial configurations and their log
density.  That density is used exactly once in ``-beta_hot*U - logq0``.  After
the corrected population is resampled, the bath is
``pi_beta(x) proportional exp(-beta U_mW(x))``: later weights contain only
energy differences and mutation uses local single-particle ``du_move`` calls.
No reverse ODE or per-rung flow density is evaluated.
"""
from __future__ import annotations

import argparse
import gc
import math
from pathlib import Path
import time

import torch

from liquid_coupling_flow.mw.mw_energy import (
    EnergyEvalCounter, count_energy_evals, du_move, mw_energy_chunked,
)
from liquid_coupling_flow.mw.mw_ersi import load_ersi
from liquid_coupling_flow.mw.mw_smc import ess


def multinomial_resample(x, U, logw, gen):
    """Resample all walkers and reset their weights; return ancestry indices."""
    weights = torch.softmax(logw.double(), dim=0).to(logw.dtype)
    idx = torch.multinomial(weights, len(logw), replacement=True, generator=gen)
    return x[idx].clone(), U[idx].clone(), torch.zeros_like(logw), idx


def thermal_reweight(logw, U, beta_old, beta_new):
    """Flow-free incremental weight: ``-(beta_new-beta_old) U``."""
    if beta_new < beta_old:
        raise ValueError("thermal ladder must be nondecreasing in beta")
    return logw.double() - (float(beta_new) - float(beta_old)) * U.double()


def initial_log_weights(U, logq0, beta_hot):
    """Exact proposal-to-hot-target correction, up to a shared constant."""
    if U.shape != logq0.shape:
        raise ValueError(f"U and logq0 must have the same shape, got {U.shape} and {logq0.shape}")
    if not torch.isfinite(logq0).all():
        raise ValueError("initial logq0 contains non-finite values")
    return -float(beta_hot) * U.double() - logq0.double()


def beta_schedule(beta_hot, beta_target, n_rungs):
    """Linear beta ladder excluding the already-realized hot endpoint."""
    beta_hot, beta_target = float(beta_hot), float(beta_target)
    if beta_hot < 0 or beta_target < beta_hot:
        raise ValueError("require 0 <= beta_hot <= beta_target")
    if n_rungs < 0:
        raise ValueError("n_rungs must be nonnegative")
    if beta_target == beta_hot:
        return []
    if n_rungs < 1:
        raise ValueError("beta_target > beta_hot requires at least one rung")
    return torch.linspace(beta_hot, beta_target, n_rungs + 1, dtype=torch.float64)[1:].tolist()


def local_mutation_sweeps(x, U, beta, L, n_sweeps, step, gen):
    """Local random-walk Metropolis sweeps invariant to ``exp(-beta U)``."""
    B, N, _ = x.shape
    attempts = accepts = 0
    for _ in range(int(n_sweeps)):
        # Random scan removes a persistent relation between particle index and
        # mutation time without changing the invariant kernel.
        order = torch.randperm(N, device=x.device, generator=gen)
        for i in order.tolist():
            xi = torch.remainder(
                x[:, i] + float(step) * torch.randn(B, 3, device=x.device, generator=gen), L)
            dU = du_move(x, i, xi, L)
            loga = -float(beta) * dU
            accept = torch.log(torch.rand(B, device=x.device, generator=gen).clamp_min(1e-38)) < loga
            x[accept, i] = xi[accept]
            U = torch.where(accept, U + dU, U)
            accepts += int(accept.sum()); attempts += B
    return x, U, {"acceptance": accepts / max(1, attempts),
                  "accepted": accepts, "attempts": attempts}


def _summary(stage, beta, x, U, logw, counter, **extra):
    B, N, _ = x.shape
    u = (U / N).detach().float()
    row = {"stage": stage, "beta": float(beta),
           "temperature_star": (1.0 / float(beta) if float(beta) > 0 else float("inf")),
           "ess": ess(logw), "ess_fraction": ess(logw) / B,
           "U_per_N_mean": float(u.mean()), "U_per_N_std": float(u.std(unbiased=False)),
           "U_per_N_q10": float(u.quantile(.1)), "U_per_N_q90": float(u.quantile(.9)),
           **counter.as_dict()}
    row.update(extra)
    return row


def _cpu_result(x, U, logw, history, counter, protocol, seed_source, wall, rng_state):
    init_batches = seed_source.get("initial_forward_batches", 1) \
        if isinstance(seed_source, dict) else 1
    return {"x": x.detach().cpu(), "U": U.detach().cpu(),
            "logw": logw.detach().cpu(),
            "weights": torch.softmax(logw.detach().double().cpu(), dim=0),
            "history": history, "energy_counter": counter.as_dict(),
            "protocol": protocol, "seed_source": seed_source,
            "wall_seconds": float(wall), "rng_state": rng_state.cpu(),
            "initial_flow_density_stages": 1,
            "initial_flow_density_batches": int(init_batches),
            "initial_flow_density_values": int(x.shape[0]),
            "per_rung_flow_density_evaluations": 0,
            "reverse_ode_calls": 0,
            "endpoint_is_weighted": bool(float(logw.abs().max()) > 0.0),
            "exactness_note": "standard finite-particle SMC approximation: one exact q0 correction, "
                              "then exact thermal increments and invariant local kernels"}


def thermal_smc(initial_x, initial_logq, L, beta_hot, beta_target, *, n_rungs=8,
                ess_resample_fraction=.5, hot_sweeps=5, mutation_sweeps=3,
                final_sweeps=10, step=.08, seed=0, save_path=None,
                seed_source="eRSI", refresh_each_rung=True, verbose=False):
    """Run Phase-E SMC from joint eRSI ``(x, logq0)`` draws.

    The ladder is fixed in beta, so the stated ESS<1/2 resampling rule is used
    literally.  ``refresh_each_rung`` performs the handoff's one full U
    evaluation per reweight rung and makes its cost explicit in the counter.
    All local mutations are incremental ``du_move`` calls.
    """
    if initial_x.ndim != 3 or initial_x.shape[-1] != 3:
        raise ValueError(f"initial_x must be [M,N,3], got {tuple(initial_x.shape)}")
    initial_logq = torch.as_tensor(initial_logq, device=initial_x.device)
    if initial_logq.shape != (initial_x.shape[0],):
        raise ValueError(f"initial_logq must be [M], got {tuple(initial_logq.shape)}")
    if not 0 < ess_resample_fraction <= 1:
        raise ValueError("ess_resample_fraction must be in (0,1]")
    x = torch.remainder(initial_x.detach().clone().float(), float(L))
    B, N, _ = x.shape
    gen = torch.Generator(device=x.device).manual_seed(int(seed))
    betas = beta_schedule(beta_hot, beta_target, int(n_rungs))
    protocol = {"N": N, "M": B, "L": float(L), "beta_hot": float(beta_hot),
                "beta_target": float(beta_target), "betas": tuple(betas),
                "n_rungs": len(betas), "ess_resample_fraction": float(ess_resample_fraction),
                "hot_sweeps": int(hot_sweeps), "mutation_sweeps": int(mutation_sweeps),
                "final_sweeps": int(final_sweeps), "step": float(step),
                "refresh_each_rung": bool(refresh_each_rung),
                "initial_target": "exp(-beta_hot*U_mW)/q0 importance correction",
                "bath": "exp(-beta*U_mW), flow-density-free after initialization",
                "resampling": "multinomial"}
    out_path = Path(save_path) if save_path is not None else None
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
    counter = EnergyEvalCounter(); history = []; t0 = time.time()
    with count_energy_evals(counter):
        U = mw_energy_chunked(x, float(L))
        logw = initial_log_weights(U, initial_logq.to(x.device), beta_hot)
        history.append(_summary("initial_correction", beta_hot, x, U, logw, counter,
                                resampled=False, mutation_acceptance=float("nan")))
        if verbose: print(history[-1], flush=True)

        pre_ess = ess(logw)
        x, U, logw, ancestry = multinomial_resample(x, U, logw, gen)
        history.append(_summary(
            "initial_resample", beta_hot, x, U, logw, counter,
            ess_before_resample=pre_ess, resampled=True,
            unique_parents=int(ancestry.unique().numel()),
            mutation_acceptance=float("nan")))
        if verbose: print(history[-1], flush=True)

        if hot_sweeps:
            x, U, info = local_mutation_sweeps(
                x, U, beta_hot, L, hot_sweeps, step, gen)
            history.append(_summary("hot_heal", beta_hot, x, U, logw, counter,
                                    resampled=False, mutation_acceptance=info["acceptance"],
                                    mutation_attempts=info["attempts"]))
            if verbose: print(history[-1], flush=True)

        beta = float(beta_hot)
        for rung, beta_new in enumerate(betas):
            if refresh_each_rung:
                U = mw_energy_chunked(x, float(L))
            logw = thermal_reweight(logw, U, beta, beta_new)
            pre_ess = ess(logw); threshold = ess_resample_fraction * B
            resampled = pre_ess < threshold
            unique_parents = B
            if resampled:
                x, U, logw, ancestry = multinomial_resample(x, U, logw, gen)
                unique_parents = int(ancestry.unique().numel())
            x, U, info = local_mutation_sweeps(
                x, U, beta_new, L, mutation_sweeps, step, gen)
            beta = float(beta_new)
            history.append(_summary(
                "rung", beta, x, U, logw, counter, rung=rung,
                delta_beta=beta - (float(beta_hot) if rung == 0 else float(betas[rung - 1])),
                ess_before_resample=pre_ess, resampled=resampled,
                unique_parents=unique_parents, mutation_acceptance=info["acceptance"],
                mutation_attempts=info["attempts"]))
            if verbose: print(history[-1], flush=True)
            if out_path is not None:
                torch.save(_cpu_result(x, U, logw, history, counter, protocol, seed_source,
                                       time.time() - t0, gen.get_state()), out_path)

        # Ordinary endpoint histograms and g(r) require an unweighted sample.
        # If the final rung did not trigger the ESS rule, resample once here.
        if float(logw.abs().max()) > 0.0:
            pre_ess = ess(logw)
            x, U, logw, ancestry = multinomial_resample(x, U, logw, gen)
            history.append(_summary(
                "endpoint_resample", beta_target, x, U, logw, counter,
                ess_before_resample=pre_ess, resampled=True,
                unique_parents=int(ancestry.unique().numel()),
                mutation_acceptance=float("nan")))
            if verbose: print(history[-1], flush=True)

        if final_sweeps:
            x, U, info = local_mutation_sweeps(
                x, U, beta_target, L, final_sweeps, step, gen)
            history.append(_summary("target_heal", beta_target, x, U, logw, counter,
                                    resampled=False, mutation_acceptance=info["acceptance"],
                                    mutation_attempts=info["attempts"]))
            if verbose: print(history[-1], flush=True)

        # A counted final refresh is both a drift guard and the endpoint energy.
        U_fresh = mw_energy_chunked(x, float(L))
        drift = float((U - U_fresh).abs().max())
        U = U_fresh
        history.append(_summary("endpoint", beta_target, x, U, logw, counter,
                                resampled=False, mutation_acceptance=float("nan"),
                                carried_energy_max_drift=drift))
        if verbose: print(history[-1], flush=True)

    result = _cpu_result(x, U, logw, history, counter, protocol, seed_source,
                         time.time() - t0, gen.get_state())
    if out_path is not None:
        torch.save(result, out_path)
    return result


def continue_target_smc(result, n_sweeps, *, report_every=20, step=None, seed=0,
                        save_path=None, verbose=False):
    """Continue an unweighted SMC endpoint with its exact target kernel."""
    if result.get("endpoint_is_weighted", True):
        raise ValueError("resample a weighted endpoint before target continuation")
    protocol = dict(result["protocol"])
    beta = float(protocol["beta_target"])
    L = float(protocol["L"])
    step = float(protocol["step"] if step is None else step)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.as_tensor(result["x"], device=device).float().clone()
    U = torch.as_tensor(result["U"], device=device).float().clone()
    logw = torch.zeros(len(x), device=device, dtype=torch.float64)
    gen = torch.Generator(device=device).manual_seed(int(seed))
    old = result["energy_counter"]
    counter = EnergyEvalCounter(full_calls=int(old["full_calls"]),
                                move_calls=int(old["move_calls"]),
                                block_calls=int(old["block_calls"]),
                                units=float(old["cost_units"]))
    history = list(result["history"])
    source = result["seed_source"]
    out_path = Path(save_path) if save_path is not None else None
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
    prior_sweeps = int(protocol.get("continued_target_sweeps", 0))
    protocol["continued_target_sweeps"] = prior_sweeps + int(n_sweeps)
    t0 = time.time()
    with count_energy_evals(counter):
        done = 0
        while done < int(n_sweeps):
            take = min(int(report_every), int(n_sweeps) - done)
            x, U, info = local_mutation_sweeps(x, U, beta, L, take, step, gen)
            done += take
            history.append(_summary(
                "target_continue", beta, x, U, logw, counter,
                continuation_sweeps=prior_sweeps + done, resampled=False,
                mutation_acceptance=info["acceptance"],
                mutation_attempts=info["attempts"]))
            if verbose:
                print(history[-1], flush=True)
            if out_path is not None:
                torch.save(_cpu_result(x, U, logw, history, counter, protocol, source,
                                       result.get("wall_seconds", 0.0) + time.time() - t0,
                                       gen.get_state()), out_path)
        U_fresh = mw_energy_chunked(x, L)
        drift = float((U - U_fresh).abs().max())
        U = U_fresh
        history.append(_summary(
            "endpoint", beta, x, U, logw, counter, resampled=False,
            mutation_acceptance=float("nan"), carried_energy_max_drift=drift,
            continuation_sweeps=prior_sweeps + int(n_sweeps)))
    out = _cpu_result(x, U, logw, history, counter, protocol, source,
                      result.get("wall_seconds", 0.0) + time.time() - t0,
                      gen.get_state())
    if out_path is not None:
        torch.save(out, out_path)
    return out


def _get_nested(obj, dotted):
    for part in dotted.split("."):
        if not isinstance(obj, dict) or part not in obj:
            raise KeyError(f"seed artifact has no key {dotted!r}")
        obj = obj[part]
    return obj


def seeds_from_bank(path, key, logq_key, M, device, gen):
    obj = torch.load(path, map_location="cpu", weights_only=False)
    X = torch.as_tensor(_get_nested(obj, key)).float()
    if X.ndim != 3 or X.shape[-1] != 3:
        raise ValueError(f"seed tensor {key} must be [B,N,3], got {tuple(X.shape)}")
    logq = torch.as_tensor(_get_nested(obj, logq_key)).flatten().double()
    if logq.shape != (len(X),):
        raise ValueError(f"seed logq tensor {logq_key} must be [{len(X)}], got {tuple(logq.shape)}")
    idx = torch.randint(len(X), (M,), generator=torch.Generator().manual_seed(gen.initial_seed())) \
        if len(X) < M else torch.randperm(len(X), generator=torch.Generator().manual_seed(gen.initial_seed()))[:M]
    return X[idx].to(device), logq[idx].to(device)


def sample_ersi_initial(flow, M, gen, chunk):
    """Joint forward samples/logq in memory-bounded initialization batches."""
    if chunk < 1:
        raise ValueError("flow chunk must be positive")
    xs, logqs = [], []
    for start in range(0, M, chunk):
        x, logq = flow.sample_and_logq(min(chunk, M - start), gen)
        xs.append(x.detach())
        logqs.append(logq.detach())
        # The ODE result retains a large divergence graph.  Drop it before the
        # next RHS is constructed; the detached outputs are all SMC needs.
        del x, logq
        gc.collect()
        if flow.device.type == "cuda":
            torch.cuda.empty_cache()
    return torch.cat(xs), torch.cat(logqs), len(xs)


def main():
    ap = argparse.ArgumentParser(description="eRSI-seeded exact-in-weights thermal SMC for mW")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--ersi-checkpoint")
    src.add_argument("--seed-bank")
    ap.add_argument("--seed-key", default="flow.X")
    ap.add_argument("--seed-logq-key", default="flow.logq")
    ap.add_argument("--out", required=True)
    ap.add_argument("--M", type=int, default=64)
    ap.add_argument("--flow-chunk", type=int, default=64)
    ap.add_argument("--beta-hot", type=float, required=True)
    ap.add_argument("--beta-target", type=float, required=True)
    ap.add_argument("--rungs", type=int, default=8)
    ap.add_argument("--ess-resample", type=float, default=.5)
    ap.add_argument("--hot-sweeps", type=int, default=5)
    ap.add_argument("--mutation-sweeps", type=int, default=3)
    ap.add_argument("--final-sweeps", type=int, default=10)
    ap.add_argument("--step", type=float, default=.08)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args(); dev = torch.device(a.device)
    gen = torch.Generator(device=dev).manual_seed(a.seed)
    if a.ersi_checkpoint:
        flow = load_ersi(a.ersi_checkpoint, dev)
        X, logq0, flow_batches = sample_ersi_initial(flow, a.M, gen, a.flow_chunk)
        L = flow.L; source = {"type": "eRSI_checkpoint", "path": a.ersi_checkpoint,
                              "initial_forward_logq": True,
                              "initial_forward_batches": flow_batches,
                              "reverse_ode_used": False}
        del flow; gc.collect()
        if dev.type == "cuda": torch.cuda.empty_cache()
    else:
        X, logq0 = seeds_from_bank(
            a.seed_bank, a.seed_key, a.seed_logq_key, a.M, dev, gen)
        L = float((X.shape[1] / 0.4564) ** (1 / 3))
        source = {"type": "eRSI_seed_bank", "path": a.seed_bank, "key": a.seed_key,
                  "logq_key": a.seed_logq_key, "initial_forward_logq": True,
                  "reverse_ode_used": False}
    print(f"thermal SMC: source={source} M={len(X)} N={X.shape[1]} L={L:.5f} "
          f"beta={a.beta_hot:g}->{a.beta_target:g} rungs={a.rungs} "
          f"sweeps={a.hot_sweeps}/{a.mutation_sweeps}/{a.final_sweeps}", flush=True)
    result = thermal_smc(
        X, logq0, L, a.beta_hot, a.beta_target, n_rungs=a.rungs,
        ess_resample_fraction=a.ess_resample, hot_sweeps=a.hot_sweeps,
        mutation_sweeps=a.mutation_sweeps, final_sweeps=a.final_sweeps,
        step=a.step, seed=a.seed + 1, save_path=a.out, seed_source=source, verbose=True)
    print(f"saved {a.out}; counter={result['energy_counter']} "
          f"initial_flow_density_stages={result['initial_flow_density_stages']} "
          f"initial_flow_density_batches={result['initial_flow_density_batches']} "
          f"per_rung_flow_density_evaluations={result['per_rung_flow_density_evaluations']}", flush=True)


if __name__ == "__main__":
    main()
