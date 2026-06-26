"""Final campaign figure: TF/FR clash-by-index + g_BB(peak,spurious) for all arms vs the data reference."""
from __future__ import annotations
import os, numpy as np, torch, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from liquid_coupling_flow.ka_exposure_lf import tf_fr_by_index, _load

ART = os.path.join(os.path.dirname(__file__), "artifacts")
ARMS = [("baseline", "ka_localframe_N100_20k.pt", "C0"),
        ("noise-null", "ka_localframe_ss_N100.pt", "C7"),
        ("+B soft(best)", None, "C2"),      # set ckpt name from Task 3 winner, e.g. ka_softlabel_N100_both_s1.0.pt
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


if __name__ == "__main__":
    import sys
    main(soft_ckpt=sys.argv[1] if len(sys.argv) > 1 else "ka_softlabel_N100_both_s1.0.pt")
