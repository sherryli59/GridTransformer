"""Energy-based finetuning of the (no-input-noise) GridFormer checkpoint. The checkpoint
fits the data well (teacher-forced -logq -286) but its free-running samples drift into
overlaps (0.52). Finetune with the Boltzmann-generator objective:
    reverse-KL  E_{x~q}[log q(x) + U(x)/kT]   (REINFORCE; pushes mass off overlaps)
  + forward-KL  -E_{data}[log q]              (anchor; prevents mode collapse)
Energy is CORE-CLAMPED so overlaps cost a large but finite penalty (no inf gradients)."""
from __future__ import annotations
import os, time, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, torch
from liquid_coupling_flow.ka_gridformer import KAGridformer
from liquid_coupling_flow.ka_energy import SIGMA, EPS, RCUT_FACTOR, ka_energy
from liquid_coupling_flow.ka_observables import partial_gr
from liquid_coupling_flow.particles import min_image

ART = os.path.join(os.path.dirname(__file__), "artifacts")


def overlap_frac(x, L, N, thresh=0.7):
    _, r = min_image(x, L)
    r = r + torch.eye(N, device=x.device)[None] * 1e3
    return float((r.min(dim=2).values.reshape(-1) < thresh).float().mean())


def clamped_energy(x, s, L, min_dist=0.6):
    """KA energy with r clamped to min_dist -> overlaps give a LARGE but FINITE penalty."""
    B, N, _ = x.shape
    sig = torch.tensor(SIGMA, device=x.device, dtype=x.dtype)[s.long()][:, s.long()]
    eps = torch.tensor(EPS, device=x.device, dtype=x.dtype)[s.long()][:, s.long()]
    rc = RCUT_FACTOR * sig
    diff = x[:, :, None, :] - x[:, None, :, :]
    diff = diff - L * torch.round(diff / L)
    r2 = (diff ** 2).sum(-1).clamp_min(min_dist ** 2)
    eye = torch.eye(N, device=x.device, dtype=torch.bool)
    r2 = r2.masked_fill(eye, 1e12)
    inv6 = (sig ** 2 / r2) ** 3
    e = 4 * eps * (inv6 ** 2 - inv6)
    src6 = (sig / rc) ** 6
    e = torch.where(r2 < rc ** 2, e - 4 * eps * (src6 ** 2 - src6), torch.zeros_like(e))
    return 0.5 * e.sum(dim=(1, 2))


def main(N=100, steps=600, kT=0.5, fkl_w=1.0, lr=5e-5, B=128,
         device="cuda" if torch.cuda.is_available() else "cpu"):
    ck = torch.load(os.path.join(ART, "ka_gridformer_N100_nonoise.pt"), map_location=device)
    L, s = ck["L"], ck["s"].to(device)
    ref = torch.load(os.path.join(ART, f"ka_reference_N{N}.pt"), map_location=device)
    data = ref["x"].to(device)
    ov_ref = overlap_frac(data[:512], L, N)
    U_ref = (ka_energy(data, s, L) / N).mean().item()
    m = KAGridformer(L=L, n_bins=96, d_model=192, n_head=6, n_layer=6, R=32).to(device)
    m.load_state_dict(ck["state_dict"])
    print(f"loaded no-noise ckpt (overlaps {ck['overlaps']:.3f}); ref overlaps {ov_ref:.3f}, <U>/N {U_ref:.3f}",
          flush=True)
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=0.0)
    t0 = time.time()
    for step in range(steps):
        with torch.no_grad():
            x, _ = m.sample(B, s, device=device)                    # samples (discrete)
        logq = m.log_prob(x, s)                                      # [B], with grad
        U = clamped_energy(x, s, L) / kT                             # [B]
        adv = logq.detach() + U                                     # reverse-KL integrand
        adv = (adv - adv.mean()) / (adv.std() + 1e-6)               # normalized advantage
        rkl = (adv.detach() * logq).mean()                          # REINFORCE surrogate
        idx = torch.randint(0, data.shape[0], (B,), device=device)
        fkl = -m.log_prob(data[idx], s).mean()                      # forward-KL anchor
        loss = rkl + fkl_w * fkl
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 5.0)
        opt.step()
        if step % 100 == 0 or step == steps - 1:
            with torch.no_grad():
                xo, _ = m.sample(256, s, device=device)
                ov = overlap_frac(xo, L, N)
                Um = (clamped_energy(xo, s, L) / N).median().item()
            print(f"  step {step:4d} overlaps {ov:.3f} (ref {ov_ref:.3f}) clampU/N median {Um:.3f} "
                  f"fkl {fkl.item():.1f} {time.time()-t0:.0f}s", flush=True)
    m.eval()
    with torch.no_grad():
        x, _ = m.sample(1000, s, device=device)
    ov = overlap_frac(x, L, N)
    print(f"FINAL: overlaps {ov:.3f} (ref {ov_ref:.3f})", flush=True)
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))
    for k, p in enumerate([(0, 0), (0, 1), (1, 1)]):
        rc, g = partial_gr(data[:2000], s, L, L / 2, 80, p)
        rc2, g2 = partial_gr(x, s, L, L / 2, 80, p)
        ax[k].plot(rc, g, "k", lw=2.5, label="reference"); ax[k].plot(rc2, g2, "C1", lw=1.6, label="energy-finetuned")
        ax[k].axvline(2 ** (1 / 6), color="grey", ls="--", lw=1)
        ax[k].set_title(f"g_{['A','B'][p[0]]}{['A','B'][p[1]]}(r)"); ax[k].set_xlabel("r"); ax[k].legend(fontsize=8)
    fig.suptitle(f"GridFormer + energy finetune: overlaps {ck['overlaps']:.3f} -> {ov:.3f} (ref {ov_ref:.3f})",
                 fontsize=13)
    fig.tight_layout(); out = os.path.join(ART, "ka_gridformer_energy.png")
    fig.savefig(out, dpi=120); print(f"saved {out}", flush=True)


if __name__ == "__main__":
    main()
