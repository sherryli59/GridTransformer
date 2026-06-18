"""Train the KAGridformer (binned + factorized + displacement + curve, the LJ27 recipe)
on the N=100 KA reference and test if it REPRODUCES THE TRAINING SET -- the in-distribution
gate the absolute-spline flows failed (overlaps 0.73-0.83 vs reference 0.000)."""
from __future__ import annotations
import os, time, math, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, torch
from liquid_coupling_flow.ka_gridformer import KAGridformer
from liquid_coupling_flow.ka_energy import ka_energy
from liquid_coupling_flow.ka_observables import partial_gr
from liquid_coupling_flow.particles import min_image

ART = os.path.join(os.path.dirname(__file__), "artifacts")


def overlap_frac(x, L, N, thresh=0.7):
    _, r = min_image(x, L)
    r = r + torch.eye(N, device=x.device)[None] * 1e3
    return float((r.min(dim=2).values.reshape(-1) < thresh).float().mean())


def main(N=100, steps=5000, device="cuda" if torch.cuda.is_available() else "cpu"):
    ref = torch.load(os.path.join(ART, f"ka_reference_N{N}.pt"), map_location=device)
    data, s, L = ref["x"].to(device), ref["s"].to(device), ref["L"]
    U_ref = (ka_energy(data, s, L) / N).mean().item()
    ov_ref = overlap_frac(data[:512], L, N)
    print(f"reference N={N}: {data.shape[0]} configs, <U>/N {U_ref:.3f}, overlaps {ov_ref:.3f}", flush=True)

    m = KAGridformer(L=L, n_bins=96, d_model=192, n_head=6, n_layer=6, R=32).to(device)
    print(f"gridformer {sum(p.numel() for p in m.parameters())/1e6:.2f}M params", flush=True)
    opt = torch.optim.AdamW(m.parameters(), lr=3e-4, weight_decay=1e-4)
    B, t0 = 128, time.time()
    for step in range(steps):
        idx = torch.randint(0, data.shape[0], (B,), device=device)
        loss = -m.log_prob(data[idx], s).mean()
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 5.0)
        opt.step()
        if step % 500 == 0 or step == steps - 1:
            print(f"  step {step:4d} -logq {loss.item():.1f} (base {2*N*math.log(L):.1f}) {time.time()-t0:.0f}s", flush=True)
    m.eval()
    with torch.no_grad():
        x, _ = m.sample(1000, s, device=device)
    ov = overlap_frac(x, L, N)
    U = (ka_energy(x, s, L) / N); Um = U[torch.isfinite(U)].median().item()
    print(f"SAMPLES: overlaps {ov:.3f} (ref {ov_ref:.3f}), <U>/N median {Um:.3f} (ref {U_ref:.3f})", flush=True)

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))
    for k, p in enumerate([(0, 0), (0, 1), (1, 1)]):
        rc, g = partial_gr(data[:2000], s, L, L / 2, 80, p)
        rc2, g2 = partial_gr(x, s, L, L / 2, 80, p)
        ax[k].plot(rc, g, "k", lw=2.5, label="reference"); ax[k].plot(rc2, g2, "C2", lw=1.6, label="gridformer")
        ax[k].axvline(2 ** (1 / 6), color="grey", ls="--", lw=1)
        ax[k].set_title(f"g_{['A','B'][p[0]]}{['A','B'][p[1]]}(r)"); ax[k].set_xlabel("r"); ax[k].legend(fontsize=8)
    fig.suptitle(f"GridFormer in-distribution N={N}: overlaps {ov:.3f} (ref {ov_ref:.3f}) "
                 f"-- {'REPRODUCES' if ov < 0.05 else 'FAILS'}", fontsize=13)
    fig.tight_layout(); out = os.path.join(ART, "ka_gridformer_indist.png")
    fig.savefig(out, dpi=120); print(f"saved {out}", flush=True)
    torch.save({"state_dict": m.state_dict(), "L": L, "s": s.cpu(), "N": N,
                "overlaps": ov, "U_median": Um}, os.path.join(ART, f"ka_gridformer_N{N}.pt"))


if __name__ == "__main__":
    main()
