"""ONE-STEP TEACHER-FORCED species-resolved structure -- separates HEAD-expressiveness from ROLLOUT DRIFT.

For each particle j with the TRUE prefix (0..j-1 from data), sample j's position from the model's
conditional and accumulate the CAUSAL pair distances d(model-j, true-k<j) resolved by (species_j,
species_k). Compare to data d(true-j, true-k). Because every ordered pair (j,k>k) is counted once -- at
the moment the later particle j is placed -- this causal-pair RDF is exactly the g(r) the HEAD would
produce if the prefix were always correct, i.e. with NO exposure-bias drift.

Decisive read: if g_BB is ALREADY wrong here (in-distribution N=100), the conditional/head is the
culprit -- specifically the species-blind position head P(a|context) that cannot place a B closer than an
A in the same pocket [[localframe-conditional-transfers]]. If it MATCHES data, the in-distribution g(r)
error is rollout drift, not the head, and a species-coupled head will not help. Confirms/refutes the
species<->position decoupling hypothesis BEFORE any retrain."""
from __future__ import annotations
import os, math, torch, numpy as np, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch.nn.functional as F
from liquid_coupling_flow.ka_localframe import KALocalFrameModel
from liquid_coupling_flow.ka_gridformer import _wrap_pm

ART = os.path.join(os.path.dirname(__file__), "artifacts")
SIG = {(0, 0): 1.0, (0, 1): 0.8, (1, 1): 0.88}
PAIRS = ((0, 0), (0, 1), (1, 1))


@torch.no_grad()
def model_place_tf(m, xo, so, sc, L, N):
    """Teacher-forced one-step placement of EVERY particle given the TRUE prefix (m._local context)."""
    B = xo.shape[0]; arc = m._arc_scale(N)
    context, origin = m._local(xo, so, sc, L, N)
    ba = torch.multinomial(F.softmax(m.head_a(context).reshape(-1, m.n_bins), -1), 1).reshape(B, N)
    bb = torch.multinomial(F.softmax(m.head_b(context + m.bin_a_emb(ba)).reshape(-1, m.n_bins), -1), 1).reshape(B, N)
    a = m._bin_center(ba) + (torch.rand(B, N, device=xo.device) - 0.5) * m.bin_w
    bcoord = m._bin_center(bb) + (torch.rand(B, N, device=xo.device) - 0.5) * m.bin_w
    return torch.remainder(origin + torch.stack([a, bcoord], -1) * arc, L)


@torch.no_grad()
def causal_rdf(pos_j, pos_k, so, L, rmax, nbins=100):
    """Shell-normalized causal-pair density: pairs (j, k<j) with neighbour at TRUE pos_k, resolved by
    (s_j, s_k). pos_j = placed positions (model or true). Same scale for data & model (divide by the
    SAME true causal-pair count per species bucket), so peak position and small-r hole compare directly."""
    B, N, _ = pos_j.shape; dev = pos_j.device
    df = _wrap_pm(pos_j[:, :, None, :] - pos_k[:, None, :, :], L)        # [B,N(j),N(k),2]
    d = (df ** 2).sum(-1).sqrt()
    jj = torch.arange(N, device=dev); causal = jj[None, None, :] < jj[None, :, None]
    sj = so[:, :, None].expand(B, N, N); sk = so[:, None, :].expand(B, N, N)
    centers = (torch.arange(nbins, device=dev) + 0.5) * (rmax / nbins); dr = rmax / nbins
    out = {}
    for p in PAIRS:
        if p == (0, 1):
            spec = ((sj == 0) & (sk == 1)) | ((sj == 1) & (sk == 0))
        else:
            spec = (sj == p[0]) & (sk == p[1])
        npairs = float((spec & causal).sum().clamp_min(1))                # TRUE count (same for data & model)
        h = torch.histc(d[spec & causal], bins=nbins, min=0.0, max=float(rmax))
        g = h / npairs / (2 * math.pi * centers * dr).clamp_min(1e-9)
        out[p] = (centers.cpu().numpy(), g.cpu().numpy())
    return out


@torch.no_grad()
def main(device="cuda" if torch.cuda.is_available() else "cpu", B=256):
    ck = torch.load(os.path.join(ART, "ka_localframe_N100_20k.pt"), map_location=device, weights_only=False)
    m = KALocalFrameModel(rho=ck["rho"], n_bins=ck["n_bins"], knn=ck["knn"]).to(device); m.load_state_dict(ck["state_dict"]); m.eval()
    print(f"loaded ka_localframe_N100_20k.pt (step {ck.get('step')})", flush=True)
    sizes = (36, 100, 256)
    fig, ax = plt.subplots(3, 3, figsize=(16, 13)); quant = {}
    for r, N in enumerate(sizes):
        ref = torch.load(os.path.join(ART, f"ka_reference_N{N}.pt"), map_location=device, weights_only=False)
        s, L, data = ref["s"].to(device).long(), ref["L"], ref["x"].to(device)[:B]
        order = m.geo._curve_order(data, N); xo = torch.gather(data, 1, order[..., None].expand(-1, -1, 2))
        sfull = s.expand(data.shape[0], N) if s.dim() == 1 else s[:B]
        so = torch.gather(sfull[:B], 1, order); sc = m.geo._scaffold(N, device)
        rmax = min(L / 2, 4.0)
        pos_model = model_place_tf(m, xo, so, sc, L, N)
        g_data = causal_rdf(xo, xo, so, L, rmax)          # data: true j vs true prefix
        g_model = causal_rdf(pos_model, xo, so, L, rmax)  # model 1-step TF: placed j vs true prefix
        tag = "train" if N == 100 else "HELDOUT"
        for c, p in enumerate(PAIRS):
            rc, gd = g_data[p]; _, gm = g_model[p]
            ax[r, c].plot(rc, gd, "k", lw=2.2, label="data (TF)")
            ax[r, c].plot(rc, gm, "C0", lw=1.7, label="model 1-step TF")
            ax[r, c].axvline(2 ** (1 / 6) * SIG[p], color="grey", ls=":", lw=1); ax[r, c].set_xlim(0, 3)
            ax[r, c].set_title(f"N={N} ({tag})  causal g_{['A','B'][p[0]]}{['A','B'][p[1]]}(r)  [head-only, no drift]")
            ax[r, c].legend(fontsize=8); ax[r, c].grid(alpha=0.3)
        quant[N] = (g_data[(1, 1)], g_model[(1, 1)])
    fig.suptitle("ONE-STEP TEACHER-FORCED causal g(r): data vs model (NO rollout drift). "
                 "If g_BB (col 3) is wrong here, the species-blind position head is the culprit.", fontsize=12)
    fig.tight_layout(); out = os.path.join(ART, "ka_localframe_tfstruct.png"); fig.savefig(out, dpi=120); print(f"saved {out}", flush=True)

    print("\n  g_BB (head-only, teacher-forced, NO drift) -- data vs model 1-step:", flush=True)
    for N in sizes:
        (rc, gd), (_, gm) = quant[N]
        sm = rc < 0.88
        print(f"    N={N:3d}: peak data {gd.max():.2f}@{rc[gd.argmax()]:.2f}  model {gm.max():.2f}@{rc[gm.argmax()]:.2f}  | "
              f"g_BB(r<0.88) data {gd[sm].mean():.3f}  model {gm[sm].mean():.3f}  (spurious B-B contacts)", flush=True)


if __name__ == "__main__":
    main()
