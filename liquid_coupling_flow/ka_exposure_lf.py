"""Per-index TEACHER-FORCED vs FREE-RUN exposure-bias diagnostic for the causal local-frame model.
TF: place every particle j from the TRUE prefix (model heads on _local context); clash j vs the TRUE
0..j-1. FR: the model's own AR rollout (sample); clash j vs its OWN 0..j-1. The TF->FR gap, growing with
curve index j (worst at closure j>80), IS the exposure bias. Co-reports g_BB peak + spurious(<0.88)."""
from __future__ import annotations
import os, torch, torch.nn.functional as F, numpy as np, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from liquid_coupling_flow.ka_localframe import KALocalFrameModel
from liquid_coupling_flow.ka_observables import partial_gr

ART = os.path.join(os.path.dirname(__file__), "artifacts")


def clash_by_index(cand, prefix, L, thr=0.7):
    """cand [B,N,2] (particle j), prefix [B,N,2] (the 0..j-1 to check j against). Returns clash[N]."""
    N = cand.shape[1]; out = np.zeros(N)
    for j in range(1, N):
        df = cand[:, j:j + 1] - prefix[:, :j]; df = df - L * torch.round(df / L)
        out[j] = float(((df ** 2).sum(-1).min(1).values.sqrt() < thr).float().mean())
    return out


def _gbb(pos, sp, L):
    rc, g = partial_gr(pos, sp, L, min(L / 2, 4.0), 100, (1, 1))
    return float(g.max()), float(g[rc < 0.88].mean())


@torch.no_grad()
def _tf_place(m, xo, so, L, N, device):
    """Teacher-forced placement of every particle j from the TRUE prefix context."""
    sc = m.geo._scaffold(N, device); arc = m._arc_scale(N); B = xo.shape[0]
    context, origin = m._local(xo, so, sc, L, N)
    ba = torch.multinomial(F.softmax(m.head_a(context).reshape(-1, m.n_bins), -1), 1).reshape(B, N)
    bb = torch.multinomial(F.softmax(m.head_b(context + m.bin_a_emb(ba)).reshape(-1, m.n_bins), -1), 1).reshape(B, N)
    a = m._bin_center(ba) + (torch.rand(B, N, device=device) - 0.5) * m.bin_w
    bc = m._bin_center(bb) + (torch.rand(B, N, device=device) - 0.5) * m.bin_w
    return torch.remainder(origin + torch.stack([a, bc], -1) * arc, L)


@torch.no_grad()
def tf_fr_by_index(m, N, device, B=512):
    ref = torch.load(os.path.join(ART, f"ka_reference_N{N}.pt"), map_location=device, weights_only=False)
    s, L, data = ref["s"].to(device).long(), ref["L"], ref["x"].to(device)[:B]
    nB = int((s == 1).sum()) if s.dim() == 1 else int((s[0] == 1).sum())
    order = m.geo._curve_order(data, N)
    xo = torch.gather(data, 1, order[..., None].expand(-1, -1, 2))
    so = torch.gather(s.expand(B, N).clone() if s.dim() == 1 else s[:B], 1, order)
    tf_pos = _tf_place(m, xo, so, L, N, device)                                      # ONE TF sample, reused
    tf = clash_by_index(tf_pos, xo, L)                                               # TF: vs TRUE prefix
    xg, sg = m.sample(B, N, n_B=nB, device=device)
    og = m.geo._curve_order(xg, N)
    xgo = torch.gather(xg, 1, og[..., None].expand(-1, -1, 2)); sgo = torch.gather(sg, 1, og)
    fr = clash_by_index(xgo, xgo, L)                                                  # FR: vs OWN prefix
    sd = s.expand(B, N) if s.dim() == 1 else s[:B]
    return {"tf": tf, "fr": fr, "L": L,
            "gbb_tf": _gbb(tf_pos, so, L),                                            # partial_gr is order-invariant
            "gbb_fr": _gbb(xg, sg, L), "gbb_data": _gbb(data, sd, L)}


def load_compat(m, state_dict):
    """strict-ish load. The factorized heads MUST all be present; only the non-factorized-only heads
    (sp_out_emb, head_sa) may be absent -- the baseline ckpt ka_localframe_N100_20k.pt predates them
    (VERIFIED missing: sp_out_emb.weight, head_sa.weight, head_sa.bias). Any OTHER missing key, or any
    unexpected key (e.g. a future subclass param), is a real mismatch and raises -- this is what guards
    the silent strict=False garbage-load. (Blanket strict=True is wrong here: it would error on the
    baseline ckpt's missing optional heads.)"""
    inc = m.load_state_dict(state_dict, strict=False)
    bad = [k for k in inc.missing_keys if not k.startswith(("sp_out_emb", "head_sa"))]
    assert not bad and not inc.unexpected_keys, (bad, list(inc.unexpected_keys))
    return m


def _load(ckpt, device):
    ck = torch.load(os.path.join(ART, ckpt), map_location=device, weights_only=False)
    m = KALocalFrameModel(rho=ck["rho"], n_bins=ck["n_bins"], knn=ck["knn"],
                          head_mode=ck.get("head_mode", "factorized")).to(device)
    load_compat(m, ck["state_dict"]); m.eval()
    return m


@torch.no_grad()
def main(N=100, device="cuda" if torch.cuda.is_available() else "cpu"):
    arms = [("baseline (one-hot)", "ka_localframe_N100_20k.pt", "C0"),
            ("noise-null (failed)", "ka_localframe_ss_N100.pt", "C7")]
    fig, ax = plt.subplots(1, 1, figsize=(8, 5)); rows = []
    for tag, ckpt, col in arms:
        if not os.path.exists(os.path.join(ART, ckpt)):
            print(f"SKIP {tag}: {ckpt} missing", flush=True); continue
        d = tf_fr_by_index(_load(ckpt, device), N, device); jj = np.arange(1, N)
        early = d["fr"][1:N // 3].mean(); mid = d["fr"][N // 3:2 * N // 3].mean(); late = d["fr"][2 * N // 3:].mean()
        print(f"{tag:22s}: FR clash early {early:.3f} mid {mid:.3f} late(closure) {late:.3f} | "
              f"TF mean {d['tf'][1:].mean():.3f} FR mean {d['fr'][1:].mean():.3f} gap {d['fr'][1:].mean()-d['tf'][1:].mean():.3f} | "
              f"g_BB peak TF {d['gbb_tf'][0]:.2f} FR {d['gbb_fr'][0]:.2f} data {d['gbb_data'][0]:.2f} | "
              f"spur FR {d['gbb_fr'][1]:.3f} data {d['gbb_data'][1]:.3f}", flush=True)
        ax.plot(jj, d["tf"][1:], col, ls="--", lw=1.5, label=f"{tag} TF")
        ax.plot(jj, d["fr"][1:], col, lw=2, label=f"{tag} FR")
        rows.append((tag, d))
    ax.set_xlabel("curve index j"); ax.set_ylabel("fresh-clash rate"); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    ax.set_title(f"Exposure bias (local-frame): TF vs FR clash by index, N={N}")
    out = os.path.join(ART, f"ka_exposure_lf_N{N}.png"); fig.tight_layout(); fig.savefig(out, dpi=120)
    print(f"saved {out}", flush=True)


if __name__ == "__main__":
    main()
