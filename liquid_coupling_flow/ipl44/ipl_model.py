"""IPL adaptation of the +A curve-flow generator (KACurveFlowModel). Only the geometry changes: rho=0.5
(box L=sqrt(N/0.5)), N=44, 50:50 species. Re-derives nothing about the head; re-validates that the IPL cage
fits the (a,b) offset support (arc_range + flow tail_bound) before training."""
from __future__ import annotations
import torch
from liquid_coupling_flow.ka_curveflow import KACurveFlowModel
from liquid_coupling_flow.ka_localframe import _wrap_pm
from liquid_coupling_flow.ipl44.ipl_energy import load_ipl_reference, ipl_box


def make_ipl_model(num_bins=8, tail_bound=2.75, knn=16, arc_range=2.75, device="cpu"):
    # rho=0.5 -> the geo builds L=sqrt(N/rho); arc_range/tail_bound re-checked by support_coverage.
    # Bounds re-tuned to the TRUE wrapped-data offset support (std 0.66, max 2.49): tail_bound 4.0 was
    # tuned on UNWRAPPED data (mis-registered std 1.42, same max ~2.5 so the max-based gate missed it),
    # spreading the 8-knot spline ~1.5x too wide. 2.75 hugs the support -> finer placement resolution.
    m = KACurveFlowModel(rho=0.5, n_bins=192, knn=knn, arc_range=arc_range,
                         num_bins=num_bins, tail_bound=tail_bound).to(device)
    return m


import os, time
ART = os.path.join(os.path.dirname(__file__), "data")


def train_ipl(steps=40000, num_bins=8, tail_bound=2.75, arc_range=2.75, knn=16, lr=3e-4,
              out="ipl44_curveflow.pt", device="cuda" if torch.cuda.is_available() else "cpu"):
    from liquid_coupling_flow.ka_gridformer_train import augment
    N, L = ipl_box()
    data, sp = load_ipl_reference(device)                                # [M,44,2],[M,44]
    nB = int((sp[0] == 1).sum())                                         # 22 (50:50 composition)
    m = make_ipl_model(num_bins=num_bins, tail_bound=tail_bound, arc_range=arc_range, knn=knn, device=device)
    m.train()
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=1e-4); B, t0 = 128, time.time(); warmup = 500
    for step in range(steps):
        for g in opt.param_groups:
            g["lr"] = lr * min(1.0, (step + 1) / warmup)
        idx = torch.randint(0, data.shape[0], (B,), device=device)
        loss = (-m.log_prob(augment(data[idx], L), sp[idx]) / N).mean()
        if not torch.isfinite(loss):
            raise FloatingPointError(f"NON-FINITE loss at step {step}")
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 5.0); opt.step()
        if step % 1000 == 0:
            print(f"  step {step:5d} nll/N {loss.item():.3f} {time.time()-t0:.0f}s", flush=True)
    epochs = steps * B / data.shape[0]; n_params = sum(p.numel() for p in m.parameters())
    torch.save({"state_dict": m.state_dict(), "num_bins": num_bins, "tail_bound": tail_bound,
                "arc_range": arc_range, "knn": knn, "n_B": nB, "steps": steps, "epochs": epochs,
                "n_params": n_params}, os.path.join(ART, out))
    print(f"saved {out}  (epochs={epochs:.0f}, params={n_params/1e3:.0f}k vs eRSI 22k/580k)", flush=True)
    return out


if __name__ == "__main__":
    import sys
    train_ipl(steps=int(sys.argv[1]) if len(sys.argv) > 1 else 40000)


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
