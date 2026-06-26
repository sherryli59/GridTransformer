"""In-dist N=100 verdict: does the spline-flow head's FREE-RUN g(r) grow sharp contact peaks AND empty the
excluded-volume core, vs the categorical baseline and data? Peak position+count co-primary with height."""
from __future__ import annotations
import os, math, torch, numpy as np, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from liquid_coupling_flow.ka_flowhead import KAFlowHeadModel
from liquid_coupling_flow.ka_exposure_lf import _load
from liquid_coupling_flow.ka_observables import partial_gr
ART = os.path.join(os.path.dirname(__file__), "artifacts"); SIG = {(0, 0): 1.0, (0, 1): 0.8, (1, 1): 0.88}


def _flow(ckpt, device):
    ck = torch.load(os.path.join(ART, ckpt), map_location=device, weights_only=False)
    m = KAFlowHeadModel(rho=1.2, n_bins=192, knn=16, num_bins=ck["num_bins"], tail_bound=ck["tail_bound"],
                        frame_mode=ck.get("frame_mode", "scaffold")).to(device)
    m.load_state_dict(ck["state_dict"]); m.eval(); return m


@torch.no_grad()
def main(flow_ckpt="ka_flowhead_N100_k8_scratch.pt", N=100, B=512,
         device="cuda" if torch.cuda.is_available() else "cpu"):
    ref = torch.load(os.path.join(ART, f"ka_reference_N{N}.pt"), map_location=device, weights_only=False)
    s, L, data = ref["s"].to(device).long(), ref["L"], ref["x"].to(device); nB = int((s == 1).sum())
    arms = {"data": (data[:B], s)}
    cat = _load("ka_localframe_N100_20k.pt", device); arms["categorical"] = cat.sample(B, N, n_B=nB, device=device)
    flow = _flow(flow_ckpt, device); arms["flow"] = flow.sample(B, N, n_B=nB, device=device)
    def pk(rc, g): p, _ = find_peaks(g, prominence=0.08, height=1.03); return [round(float(rc[i]), 2) for i in p]
    fig, ax = plt.subplots(1, 3, figsize=(18, 5.2))
    for axi, (pair, nm) in zip(ax, [((0, 0), "AA"), ((0, 1), "AB"), ((1, 1), "BB")]):
        axi.axvspan(0, SIG[pair], color="grey", alpha=0.1); axi.axhline(1, color="grey", lw=0.6)
        print(f"\n g_{nm}(r):")
        for tag, (x, sx) in arms.items():
            rc, g = partial_gr(x, sx, L, min(L / 2, 4.5), 180, pair)
            axi.plot(rc, g, lw=2.4 if tag == "data" else 1.6, label=tag)
            print(f"   {tag:11s} peak {g.max():.2f}  core g(r<{SIG[pair]}) {float(g[rc<SIG[pair]].mean()):.3f}  peaks@ {pk(rc,g)}")
        axi.set_xlim(0.5, 3.0); axi.set_title(f"g_{nm}(r)"); axi.set_xlabel("r"); axi.legend(fontsize=9); axi.grid(alpha=0.25)
    out = os.path.join(ART, f"ka_flowhead_gr_N{N}.png"); fig.tight_layout(); fig.savefig(out, dpi=130)
    print(f"\nsaved {out}", flush=True)


if __name__ == "__main__":
    import sys
    main(flow_ckpt=sys.argv[1] if len(sys.argv) > 1 else "ka_flowhead_N100_k8_scratch.pt")
