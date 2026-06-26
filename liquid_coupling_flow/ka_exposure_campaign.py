"""Final campaign figure: TF/FR clash-by-index + g_BB(peak,spurious) for all arms vs the data reference,
PLUS the full partial g(r) curve comparison (AA/AB/BB) of free-run samples vs the data reference."""
from __future__ import annotations
import os, numpy as np, torch, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from liquid_coupling_flow.ka_exposure_lf import tf_fr_by_index, _load
from liquid_coupling_flow.ka_observables import partial_gr

ART = os.path.join(os.path.dirname(__file__), "artifacts")
SIG = {(0, 0): 1.0, (0, 1): 0.8, (1, 1): 0.88}                 # KA species LJ diameters (sigma_ab)
ARMS = [("baseline", "ka_localframe_N100_20k.pt", "C0"),
        ("noise-null", "ka_localframe_ss_N100.pt", "C7"),
        ("neither(+10k)", "ka_softlabel_N100_neither_s0.0.pt", "C5"),   # one-hot continued fine-tune = the honest control
        ("+B soft(best)", None, "C2"),      # soft_ckpt arg, e.g. ka_softlabel_N100_both_s0.5.pt
        ("+A sched", "ka_sched_N100.pt", "C1"),
        ("+A+B", "ka_combined_N100.pt", "C3")]


def main(N=100, soft_ckpt="ka_softlabel_N100_both_s1.0.pt", device="cuda" if torch.cuda.is_available() else "cpu"):
    fig, ax = plt.subplots(1, 2, figsize=(15, 5.2)); jj = np.arange(1, N); peaks = []
    for tag, ckpt, col in ARMS:
        ckpt = soft_ckpt if tag == "+B soft(best)" else ckpt
        if ckpt is None or not os.path.exists(os.path.join(ART, ckpt)):
            print(f"SKIP {tag}: {ckpt} missing", flush=True); continue
        d = tf_fr_by_index(_load(ckpt, device), N, device)
        ax[0].plot(jj, d["fr"][1:], col, lw=2, label=f"{tag} FR (gap {d['fr'][1:].mean()-d['tf'][1:].mean():.3f})")
        peaks.append((tag, d["gbb_fr"][0], d["gbb_fr"][1], d["gbb_data"][0]))
        print(f"{tag:14s}: FR clash {d['fr'][1:].mean():.3f} closure {d['fr'][2*N//3:].mean():.3f} "
              f"g_BB peak {d['gbb_fr'][0]:.2f} spur {d['gbb_fr'][1]:.3f}", flush=True)
    ax[0].set_xlabel("curve index j"); ax[0].set_ylabel("FR clash"); ax[0].legend(fontsize=8); ax[0].grid(alpha=0.3)
    ax[0].set_title(f"Free-run clash by index (N={N})")
    names = [p[0] for p in peaks]; xs = np.arange(len(names))
    ax[1].bar(xs - 0.2, [p[1] for p in peaks], 0.4, label="g_BB peak (FR)")
    ax[1].axhline(peaks[0][3], color="k", ls="--", lw=1, label="data peak")
    ax[1].bar(xs + 0.2, [p[2] for p in peaks], 0.4, label="spurious(<0.88)")
    ax[1].set_xticks(xs); ax[1].set_xticklabels(names, rotation=20, fontsize=8); ax[1].legend(fontsize=8)
    ax[1].set_title("g_BB peak + spurious vs data")
    out = os.path.join(ART, f"ka_exposure_campaign_N{N}.png"); fig.tight_layout(); fig.savefig(out, dpi=120)
    print(f"saved {out}", flush=True)


@torch.no_grad()
def gr_compare(N=100, soft_ckpt="ka_softlabel_N100_both_s0.5.pt", B=512,
               device="cuda" if torch.cuda.is_available() else "cpu"):
    """Full partial g(r) (AA/AB/BB) of free-run samples vs the data reference, for each local-frame arm.
    Shows WHERE structure matches/smears (contact peak, 2nd shell, excluded-volume hole) beyond the scalar
    peak/spurious. Grey band = r<sigma excluded volume; dotted = LJ contact 2^(1/6)*sigma; g->1 at large r."""
    ref = torch.load(os.path.join(ART, f"ka_reference_N{N}.pt"), map_location=device, weights_only=False)
    s, L, data = ref["s"].to(device).long(), ref["L"], ref["x"].to(device)
    nB = int((s == 1).sum()) if s.dim() == 1 else int((s[0] == 1).sum())
    samples = {"data": (data[:B], s, "k")}
    for tag, ckpt, col in ARMS:
        ckpt = soft_ckpt if tag == "+B soft(best)" else ckpt
        if ckpt is None or not os.path.exists(os.path.join(ART, ckpt)):
            print(f"SKIP {tag}: {ckpt} missing", flush=True); continue
        xg, sg = _load(ckpt, device).sample(B, N, n_B=nB, device=device)
        samples[tag] = (xg, sg, col)
    fig, ax = plt.subplots(1, 3, figsize=(18, 5.2))
    for axi, pair, ttl in ((ax[0], (0, 0), "AA"), (ax[1], (0, 1), "AB"), (ax[2], (1, 1), "BB")):
        axi.axvspan(0, SIG[pair], color="grey", alpha=0.10); axi.axhline(1.0, color="grey", ls="-", lw=0.6)
        axi.axvline(2 ** (1 / 6) * SIG[pair], color="grey", ls=":", lw=1)
        print(f"\n g_{ttl}(r) (sigma={SIG[pair]}, contact={2**(1/6)*SIG[pair]:.2f}):", flush=True)
        for tag, (x, sx, col) in samples.items():
            rc, g = partial_gr(x, sx, L, L / 2, 120, pair)
            axi.plot(rc, g, color=col, lw=2.4 if tag == "data" else 1.4, label=tag)
            pk = int(np.argmax(g))
            print(f"   {tag:14s} peak {g[pk]:.2f} @ r={rc[pk]:.2f}  g(r<{SIG[pair]:.2f}) "
                  f"{float(np.mean(g[rc < SIG[pair]])):.3f}  tail {float(np.mean(g[-15:])):.2f}", flush=True)
        axi.set_xlim(0, L / 2); axi.set_title(f"g_{ttl}(r)"); axi.set_xlabel("r"); axi.set_ylabel("g(r)"); axi.legend(fontsize=8)
    fig.suptitle(f"Partial g(r) N={N}: data vs local-frame arms (free-run)", fontsize=12)
    fig.tight_layout(); out = os.path.join(ART, f"ka_exposure_gr_N{N}.png"); fig.savefig(out, dpi=120)
    print(f"\nsaved {out}", flush=True)


if __name__ == "__main__":
    import sys
    soft = sys.argv[1] if len(sys.argv) > 1 else "ka_softlabel_N100_both_s0.5.pt"
    main(soft_ckpt=soft)
    gr_compare(soft_ckpt=soft)
