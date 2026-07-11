"""TF-vs-FR decomposition on the current ka3d_cavity_ar checkpoint (the decisive exposure-bias test).

TEACHER-FORCED generation: use the TRUE interior to build every origin/context (no rollout drift),
sample position bins from the heads, place at TF-origin + sampled residual. Measures the PER-STEP
conditional quality alone.  FREE-RUNNING generation: full rollout (model.sample_pair).

If TF is clean (few clashes, high overlap) but FR clashes -> pure exposure-bias drift -> the fix is
the exact-log_prob energy/SMC corrector (campaign lever), NOT more conditional training.
If TF ALSO clashes -> the conditional itself is wrong (scaffold alignment / capacity) -> fix that first.
"""
import torch
import torch.nn.functional as F
from liquid_coupling_flow.ka3d_cavity_ar import KA3DCavityAR, fibonacci_ball_scaffold, cavity_order, _mic
from liquid_coupling_flow.ka3d_cavity_carve import carve
from liquid_coupling_flow.ka3d_pts_observables import core_overlap
from liquid_coupling_flow.ka_energy import ka_energy

dev = "cuda" if torch.cuda.is_available() else "cpu"
d = torch.load("liquid_coupling_flow/artifacts/ka3d_dataset_N512_T0.5.pt", map_location=dev, weights_only=False)
X, S, L = d["x"].to(dev).float(), d["s"].to(dev).long(), float(d["L"])
ck = torch.load("liquid_coupling_flow/artifacts/ka3d_cavity_ar.pt", map_location=dev, weights_only=False)
m = KA3DCavityAR().to(dev); m.load_state_dict(ck["state_dict"]); m.eval()
print(f"loaded ckpt step={ck['step']}", flush=True)
r_ctx = 2.5
torch.manual_seed(0); g = torch.Generator(device=dev).manual_seed(7)


@torch.no_grad()
def tf_generate(m, irel, s_in, bnd, s_bnd, R):
    """Teacher-forced generation: TRUE origins+contexts, sampled position bins, TRUE species."""
    n, mm = irel.shape[0], bnd.shape[0]
    scaffold = fibonacci_ball_scaffold(n, R, m.rho, device=dev, dtype=irel.dtype)
    order = cavity_order(irel, R); xo, so = irel[order], s_in[order]
    combined = torch.cat([bnd, xo]); scomb = torch.cat([s_bnd, so])
    kind = torch.cat([torch.ones(mm, dtype=torch.long, device=dev), torch.zeros(n, dtype=torch.long, device=dev)])
    idxp = torch.arange(mm + n, device=dev)
    valid = idxp[None] < (mm + torch.arange(n, device=dev))[:, None]
    origin = m._origins(combined, valid, scaffold)
    h = m._neighbour_context(combined, scomb, kind, valid, origin, scaffold)
    e = m.sp_out_emb(so)
    ba = torch.multinomial(F.softmax(m.head_a(h + e), -1), 1).squeeze(-1)
    bb = torch.multinomial(F.softmax(m.head_b(h + e + m.bin_a_emb(ba)), -1), 1).squeeze(-1)
    bc = torch.multinomial(F.softmax(m.head_c(h + e + m.bin_a_emb(ba) + m.bin_b_emb(bb)), -1), 1).squeeze(-1)
    off = torch.stack([m._bin_center(ba), m._bin_center(bb), m._bin_center(bc)], -1) \
        + (torch.rand(n, 3, device=dev) - 0.5) * m.bin_w
    return origin + off, so


def energy_of(center, irel, s_in, xout, sout):
    N = irel.shape[0] + xout.shape[0]
    x = torch.empty(1, N, 3, device=dev); s = torch.empty(1, N, device=dev, dtype=torch.long)
    x[0, :irel.shape[0]] = torch.remainder(center + irel, L); x[0, irel.shape[0]:] = torch.remainder(center + xout, L)
    s[0, :irel.shape[0]] = s_in; s[0, irel.shape[0]:] = sout
    return (ka_energy(x, s, L) / N).item()


z = torch.zeros(3, device=dev)
tf_U, fr_U, tf_ov, fr_ov, tf_cl, fr_cl, tru_U = [], [], [], [], [], [], []
with torch.no_grad():
    for ci in range(900, 960):
        xb, sb = X[ci], S[ci]
        center = xb[int(torch.randint(xb.shape[0], (), generator=g, device=dev))].clone()
        R = float([1.6, 2.0, 2.4][ci % 3])
        p = carve(xb, sb, center, R, L)
        if p["n_in"] < 4:
            continue
        irel = _mic(p["x_in"], center, L); ball = _mic(p["x_out"], center, L)
        shell = ball.norm(dim=-1) < R + r_ctx
        bnd, s_bnd = ball[shell], p["s_out"][shell]
        n_B = int((p["s_in"] == 1).sum())
        ut = energy_of(center, irel, p["s_in"], ball, p["s_out"]); tru_U.append(ut)
        # TF
        gi, gs = tf_generate(m, irel, p["s_in"], bnd, s_bnd, R)
        u = energy_of(center, gi, gs, ball, p["s_out"]); tf_U.append(u); tf_cl.append(u > ut + 2)
        tf_ov.append(core_overlap(gi[None], gs[None], irel[None], p["s_in"][None], z, L)[0].item())
        # FR
        gi2, gs2, _ = m.sample_pair(bnd, s_bnd, p["n_in"] - n_B, n_B, R, return_logq=True)
        u2 = energy_of(center, gi2, gs2, ball, p["s_out"]); fr_U.append(u2); fr_cl.append(u2 > ut + 2)
        fr_ov.append(core_overlap(gi2[None], gs2[None], irel[None], p["s_in"][None], z, L)[0].item())

import statistics as st
n = len(tf_U)
print(f"held cavities evaluated: {n}", flush=True)
print(f"  TRUE   U/N median={st.median(tru_U):+.3f}", flush=True)
print(f"  TF-gen U/N median={st.median(tf_U):+.3f}  clashed={sum(tf_cl)}/{n}  overlap={sum(tf_ov)/n:.3f}", flush=True)
print(f"  FR-gen U/N median={st.median(fr_U):+.3f}  clashed={sum(fr_cl)}/{n}  overlap={sum(fr_ov)/n:.3f}", flush=True)
print(f"VERDICT: {'TF clean, FR clashes -> EXPOSURE-BIAS DRIFT -> energy/SMC corrector' if sum(tf_cl) < n*0.3 and sum(fr_cl) > n*0.6 else 'TF also clashes -> CONDITIONAL itself weak (alignment/capacity)' if sum(tf_cl) > n*0.5 else 'mixed'}", flush=True)
