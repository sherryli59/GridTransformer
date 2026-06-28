# liquid_coupling_flow/ipl44/ipl_benchmark.py
"""The comparison vs eRSI (Grenioux 2025): IS-free discard fraction + ESS-vs-R (IS cost) co-primary with
reweighted U / c_V / g(r). Self-normalized IS weight w = -beta*U - logq (full joint logq). Honest framing:
report raw AND reweighted; discard fraction + ESS R-bar are the headline IS-free / IS-cost numbers."""
from __future__ import annotations
import os, torch, numpy as np, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from liquid_coupling_flow.ipl44.ipl_model import make_ipl_model, ART
from liquid_coupling_flow.ipl44.ipl_energy import ipl_box, load_ipl_reference, ipl_energy, ipl_gr, BETA_IPL


@torch.no_grad()
def _gen(n, device):
    ck = torch.load(os.path.join(ART, "ipl44_curveflow.pt"), weights_only=False)
    m = make_ipl_model(num_bins=ck["num_bins"], tail_bound=ck["tail_bound"], arc_range=ck["arc_range"],
                       knn=ck["knn"], device=device); m.load_state_dict(ck["state_dict"]); m.eval()
    N, L = ipl_box(); xs, ss, lq = [], [], []
    for _ in range((n + 1023) // 1024):                                  # batch to fit memory
        x, s, q = m.sample(1024, N, n_B=ck["n_B"], device=device, return_logq=True)
        xs.append(x); ss.append(s); lq.append(q)
    return torch.cat(xs)[:n], torch.cat(ss)[:n], torch.cat(lq)[:n], ck


@torch.no_grad()
def _energy_chunked(x, s, chunk=4096):
    """ipl_energy over large arrays in chunks (the SoftSphere pairwise tensors are O(B*N^2);
    evaluating all 200k configs at once OOMs on smaller GPUs)."""
    return torch.cat([ipl_energy(x[i:i + chunk], s[i:i + chunk]) for i in range(0, x.shape[0], chunk)])


@torch.no_grad()
def benchmark(n_samples=200000, device="cuda" if torch.cuda.is_available() else "cpu"):
    N, L = ipl_box(); beta = BETA_IPL
    ref_pos, ref_sp = load_ipl_reference(device)
    assert ref_pos.shape[0] >= 8192, ref_pos.shape
    U_ref = _energy_chunked(ref_pos, ref_sp); U_ref_max = float(U_ref.max())
    x, s, logq, ck = _gen(n_samples, device)
    U = _energy_chunked(x, s)
    # (1) discard fraction: energy > 2x reference max
    discard = float((U > 2 * U_ref_max).float().mean())
    # self-normalized IS weights
    log_w = (-beta * U - logq); log_w = torch.where(torch.isfinite(log_w), log_w, torch.full_like(log_w, -1e30))
    # (2) ESS vs R
    Rs = np.unique(np.geomspace(100, n_samples, 25).astype(int))
    ess = []
    for R in Rs:
        w = torch.softmax(log_w[:R], 0); ess.append(float(1.0 / (w * w).sum()))
    # (3) reweighted observables
    w_all = torch.softmax(log_w, 0)
    U_raw, U_rw = float(U.mean()), float((w_all * U).sum())
    cV_raw = float(beta ** 2 * U.var(unbiased=False))
    cV_rw = float(beta ** 2 * ((w_all * U * U).sum() - (w_all * U).sum() ** 2))
    U_ref_mean = float(U_ref.mean())
    # g(r): target / raw / (reweighted via resampling by w)
    res = torch.multinomial(w_all, min(8192, n_samples), replacement=True)
    raw_idx = torch.randperm(x.shape[0], device=x.device)[:8192]
    gr_t = ipl_gr(ref_pos[:8192], ref_sp[:8192], L); gr_raw = ipl_gr(x[raw_idx], s[raw_idx], L); gr_rw = ipl_gr(x[res], s[res], L)
    out = {"discard_frac": discard, "ess_R": list(zip(Rs.tolist(), ess)), "U_ref": U_ref_mean,
           "U_raw": U_raw, "U_rw": U_rw, "cV_raw": cV_raw, "cV_rw": cV_rw,
           "n_params": ck["n_params"], "epochs": ck["epochs"]}
    print(f"discard_frac {discard:.3f} (eRSI 0.03, eFM 0.84, RSI 1.0) | ESS@max {ess[-1]:.0f}/{Rs[-1]} | "
          f"U ref {U_ref_mean:.2f} raw {U_raw:.2f} rw {U_rw:.2f} | cV raw {cV_raw:.1f} rw {cV_rw:.1f} | "
          f"params {ck['n_params']/1e3:.0f}k epochs {ck['epochs']:.0f}", flush=True)
    # figure: g(r), p(U) vs q(U), ESS-vs-R
    fig, ax = plt.subplots(1, 3, figsize=(18, 5))
    def _plot_gr(axi, gr, lab, c, lw):
        g = gr[1] if isinstance(gr, (tuple, list)) else gr; r = gr[0] if isinstance(gr, (tuple, list)) else np.arange(len(g))
        axi.plot(np.asarray(r), np.asarray(g), c, lw=lw, label=lab)
    _plot_gr(ax[0], gr_t, "target", "k", 2.4); _plot_gr(ax[0], gr_raw, "ours raw", "C0", 1.4); _plot_gr(ax[0], gr_rw, "ours reweighted", "C3", 1.6)
    ax[0].set_title("g(r)"); ax[0].set_xlabel("r"); ax[0].legend(fontsize=8)
    # log10(beta*U): raw generated energies span to ~1e36, useless on a linear axis
    lbt = np.log10(np.clip((beta * U_ref).cpu().numpy(), 1, None))
    lbg = np.log10(np.clip((beta * U).cpu().numpy(), 1, None))
    ax[1].hist(lbt, 60, density=True, alpha=0.55, color="k", label="target")
    ax[1].hist(lbg, 60, density=True, alpha=0.55, color="C3", label="ours")
    ax[1].axvline(np.log10(2 * beta * U_ref_max), color="b", ls="--", lw=1, label="discard thr")
    ax[1].set_title("energy log10(beta*U)"); ax[1].set_xlabel("log10(beta*U)"); ax[1].legend(fontsize=8)
    ax[2].loglog(Rs, ess, "C0-o", label="ours"); ax[2].loglog(Rs, Rs, "k:", lw=1, label="ESS=R (ideal)")
    ax[2].set_title("ESS vs R"); ax[2].set_xlabel("R (samples)"); ax[2].set_ylabel("ESS"); ax[2].legend(fontsize=8)
    fig.suptitle(f"AR +A curve-flow vs eRSI, N=44 IPL T=0.1 (discard {discard:.2f}, {ck['n_params']/1e3:.0f}k params)", fontsize=13)
    fig.tight_layout(); p = os.path.join(ART, "ipl44_benchmark.png"); fig.savefig(p, dpi=120); print("saved", p, flush=True)
    return out


if __name__ == "__main__":
    import sys
    benchmark(n_samples=int(sys.argv[1]) if len(sys.argv) > 1 else 200000)
