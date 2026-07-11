"""One-shot importance-sampling measurements for an mW eRSI proposal.

This module deliberately does not hide likelihood failures: it reports the
weights produced by the proposal supplied to it.  A production evaluation must
be preceded by the G-b normalization gate.
"""
from __future__ import annotations

import argparse
import os

import torch

from liquid_coupling_flow.mw.mw_energy import T_STAR, mw_energy_chunked
from liquid_coupling_flow.mw.mw_ersi import MWeRSIFlow, load_ersi
from liquid_coupling_flow.mw.mw_reference import g_r


ART = os.path.join(os.path.dirname(__file__), "artifacts")


def _generator_for(model: MWeRSIFlow, seed: int) -> torch.Generator:
    return torch.Generator(device=model.device).manual_seed(seed)


def oneshot_is(model: MWeRSIFlow, B: int = 512, seed: int = 0, *, nbins: int = 120) -> dict:
    """Draw the flow once, then compute stable IS weights and raw/reweighted g(r)."""
    if B < 2:
        raise ValueError("B must be at least two for an ESS measurement")
    gen = _generator_for(model, seed)
    X, logq = model.sample_and_logq(B, gen)
    U = mw_energy_chunked(X, model.L)
    logw = -U.double() / T_STAR - logq.double()
    if not torch.isfinite(logw).all():
        raise FloatingPointError("non-finite eRSI importance weights")
    weights = torch.softmax(logw, dim=0)
    ess = float(1.0 / weights.square().sum())
    resample_idx = torch.multinomial(weights, B, replacement=True, generator=gen)
    X_resampled = X[resample_idx]
    return {
        "X": X, "U": U, "logq": logq, "logw": logw, "weights": weights,
        "ess": ess, "u_reweighted": float((weights * U.double()).sum() / model.N),
        "gr_raw": g_r(X, model.L, nbins=nbins),
        "gr_reweighted": g_r(X_resampled, model.L, nbins=nbins),
    }


def ladder_gr(models: dict[str, object], B: int = 512, seed: int = 0, *, nbins: int = 120) -> dict:
    """Comparable raw g(r) curves for eRSI and any existing mW base adapters."""
    out = {}
    for i, (name, model) in enumerate(models.items()):
        if isinstance(model, MWeRSIFlow):
            out[name] = oneshot_is(model, B=B, seed=seed + i, nbins=nbins)["gr_raw"]
        else:
            gen = torch.Generator().manual_seed(seed + i)
            X = model.sample(B, gen)
            out[name] = g_r(X, model.L, nbins=nbins)
    return out


def _at_radius(gr, radius: float) -> float:
    r, values = gr
    return float(values[(r - radius).abs().argmin()])


def gc_gd_report(model: MWeRSIFlow, B: int = 512, seed: int = 0, *, reference_path: str | None = None,
                 save_path: str | None = None) -> dict:
    """Compute G-c shell observables and G-d ESS/energy observables.

    ``reference_path`` is optional so the small-model harness remains usable;
    when provided it must be an mW ``mc_run`` artifact containing ``cfgs``.
    """
    out = oneshot_is(model, B=B, seed=seed)
    report = {
        "ess": out["ess"], "ess_over_B": out["ess"] / B,
        "u_reweighted": out["u_reweighted"],
        "shell1_raw": _at_radius(out["gr_raw"], 1.19),
        "shell2_raw": _at_radius(out["gr_raw"], 1.85),
    }
    if reference_path is not None:
        ref = torch.load(reference_path, map_location=model.device, weights_only=False)
        if "cfgs" not in ref:
            raise ValueError(f"{reference_path} has no reference cfgs")
        cfgs = torch.remainder(torch.as_tensor(ref["cfgs"], device=model.device), model.L)
        ref_u = mw_energy_chunked(cfgs, model.L).mean().item() / model.N
        ref_gr = g_r(cfgs, model.L)
        report.update({
            "reference_u": ref_u, "u_error": report["u_reweighted"] - ref_u,
            "reference_shell1": _at_radius(ref_gr, 1.19),
            "reference_shell2": _at_radius(ref_gr, 1.85),
        })
        out["gr_reference"] = ref_gr
    if save_path is None:
        os.makedirs(ART, exist_ok=True)
        save_path = os.path.join(ART, "mw_ersi_eval.pt")
    torch.save({"report": report, **out}, save_path)
    report["artifact"] = save_path
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description="evaluate a trained mW eRSI flow with one-shot IS")
    parser.add_argument("checkpoint")
    parser.add_argument("--B", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--reference")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)
    report = gc_gd_report(load_ersi(args.checkpoint, args.device), B=args.B, seed=args.seed,
                          reference_path=args.reference)
    for key, value in report.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
