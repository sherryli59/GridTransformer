"""IPL adaptation of the +A curve-flow generator (KACurveFlowModel). Only the geometry changes: rho=0.5
(box L=sqrt(N/0.5)), N=44, 50:50 species. Re-derives nothing about the head; re-validates that the IPL cage
fits the (a,b) offset support (arc_range + flow tail_bound) before training."""
from __future__ import annotations
import torch
from liquid_coupling_flow.ka_curveflow import KACurveFlowModel
from liquid_coupling_flow.ka_localframe import _wrap_pm
from liquid_coupling_flow.ipl44.ipl_energy import load_ipl_reference, ipl_box


def make_ipl_model(num_bins=8, tail_bound=5.0, knn=16, arc_range=4.0, device="cpu"):
    # rho=0.5 -> the geo builds L=sqrt(N/rho); arc_range/tail_bound re-checked by support_coverage
    m = KACurveFlowModel(rho=0.5, n_bins=192, knn=knn, arc_range=arc_range,
                         num_bins=num_bins, tail_bound=tail_bound).to(device)
    return m


@torch.no_grad()
def support_coverage(model, B=2048, device="cpu"):
    """Fraction of equilibrium (a,b) offsets that fall outside arc_range (the bin grid) and outside the flow
    tail_bound. Both must be ~0 or the target distribution is truncated."""
    N, L = ipl_box()
    pos, sp = load_ipl_reference(device); pos, sp = pos[:B], sp[:B]
    order = model.geo._curve_order(pos, N)
    xo = torch.gather(pos, 1, order[..., None].expand(-1, -1, 2))
    so = torch.gather(sp, 1, order)
    _, origin = model._local(xo, so, model.geo._scaffold(N, device), L, N)
    ab = _wrap_pm(xo - origin, L) / model._arc_scale(N)
    amax = ab.abs().max(-1).values                                       # per-particle max |offset|
    return {"max_abs_offset": float(amax.max()),
            "oor_arc_range": float((amax > model.arc_range).float().mean()),
            "oor_tail_bound": float((amax > model.flow.spline.tail_bound).float().mean())}
