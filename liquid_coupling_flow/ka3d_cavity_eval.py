"""Evaluation helpers for generator-vs-PT cavity overlap distributions."""
from __future__ import annotations

import torch

from liquid_coupling_flow.ka3d_cavity_generator import collate_cavities
from liquid_coupling_flow.ka3d_pts_observables import core_overlap


def assemble_generated(pair, x_in_relative):
    """Insert generated relative interior positions into a frozen boundary."""
    M = x_in_relative.shape[0]; N = pair["mobile"].numel(); L = pair["L"]
    x = x_in_relative.new_empty(M, N, 3)
    s = pair["s_in"].new_empty(M, N)
    xin = torch.remainder(x_in_relative + pair["center"], L)
    x[:, pair["idx_in"]] = xin
    x[:, pair["idx_out"]] = pair["x_out"][None]
    s[:, pair["idx_in"]] = pair["s_in"][None]
    s[:, pair["idx_out"]] = pair["s_out"][None]
    return x, s


@torch.no_grad()
def generator_overlap_samples(model, pair, n_samples=32):
    """Draw independent generator configurations and pair two sample halves."""
    if n_samples < 2: raise ValueError("need at least two samples")
    b = collate_cavities([pair] * n_samples, n_max=model.n_max, n_ctx_max=model.n_ctx_max)
    rel = model.sample(b["s_in"], b["mask"], b["x_ctx"], b["s_ctx"],
                       b["ctx_mask"], b["R"], b["T"])
    full_x, full_s = assemble_generated(pair, rel[:, :pair["n_in"]])
    m = n_samples // 2
    return core_overlap(full_x[:m], full_s[:m], full_x[m:2*m], full_s[m:2*m],
                        pair["center"], pair["L"])


def pt_overlap_samples(ref_arm, random_arm, center, L, tail_fraction=.5):
    """Pair tail samples from the independently initialized PT arms."""
    xr, sr = ref_arm["x_samples"], ref_arm["s_samples"]
    xx, sx = random_arm["x_samples"], random_arm["s_samples"]
    center = torch.as_tensor(center, device=xr.device, dtype=xr.dtype)
    nr = max(1, int(len(xr) * tail_fraction)); nx = max(1, int(len(xx) * tail_fraction))
    n = min(nr, nx)
    return core_overlap(xr[-n:], sr[-n:], xx[-n:], sx[-n:], center, L)


def compare_overlap_distributions(q_gen, q_pt, bins=10):
    """Small-sample diagnostics: means/SEMs and histogram total variation."""
    qg = torch.as_tensor(q_gen).float().flatten(); qp = torch.as_tensor(q_pt).float().flatten()
    if qg.numel() < 2 or qp.numel() < 2: raise ValueError("each distribution needs at least two values")
    edges = torch.linspace(0, 1, bins + 1)
    hg = torch.histogram(qg.cpu(), bins=edges).hist.float(); hp = torch.histogram(qp.cpu(), bins=edges).hist.float()
    hg /= hg.sum().clamp_min(1); hp /= hp.sum().clamp_min(1)
    mg, mp = float(qg.mean()), float(qp.mean())
    seg = float(qg.std(unbiased=True) / qg.numel() ** .5); sep = float(qp.std(unbiased=True) / qp.numel() ** .5)
    return {"generator_mean": mg, "pt_mean": mp, "generator_sem": seg, "pt_sem": sep,
            "mean_abs_diff": abs(mg - mp), "histogram_tv": float(.5 * (hg - hp).abs().sum()),
            "n_generator": qg.numel(), "n_pt": qp.numel()}
