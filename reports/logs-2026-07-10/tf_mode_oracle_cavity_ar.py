"""Decompose the TF-clash source on the final ka3d_cavity_ar checkpoint (no retrain).
For held cavities, teacher-forced (TRUE origins/contexts), place each interior particle at:
  ORACLE = its TRUE residual bin center (+dither)  -> tests whether the BIN RESOLUTION alone
           (0.14 sigma) destroys the true clash-free config;
  MODE   = argmax bin of the conditional           -> tests whether the conditional is correctly
           PEAKED near the truth (clean) or mis-peaked (order/origin problem);
  SAMPLE = multinomial draw                          -> tests tail leakage onto clash bins.
Clean ORACLE + clean MODE + clashing SAMPLE => tail leakage (finer bins / temperature).
Clashing ORACLE => resolution wall.  Clashing MODE (oracle clean) => conditional mis-peaked.
"""
import torch
import torch.nn.functional as F
from liquid_coupling_flow.ka3d_cavity_ar import KA3DCavityAR, fibonacci_ball_scaffold, cavity_order, _mic
from liquid_coupling_flow.ka3d_cavity_carve import carve
from liquid_coupling_flow.ka_energy import ka_energy

dev = "cuda" if torch.cuda.is_available() else "cpu"
d = torch.load("liquid_coupling_flow/artifacts/ka3d_dataset_N512_T0.5.pt", map_location=dev, weights_only=False)
X, S, L = d["x"].to(dev).float(), d["s"].to(dev).long(), float(d["L"])
ck = torch.load("liquid_coupling_flow/artifacts/ka3d_cavity_ar.pt", map_location=dev, weights_only=False)
m = KA3DCavityAR().to(dev); m.load_state_dict(ck["state_dict"]); m.eval()
print(f"ckpt step={ck['step']}  n_bins={m.n_bins} arc_range={m.arc_range} bin_w={m.bin_w:.4f} sigma", flush=True)
r_ctx = 2.5; g = torch.Generator(device=dev).manual_seed(7)


@torch.no_grad()
def context(irel, s_in, bnd, s_bnd, R):
    n, mm = irel.shape[0], bnd.shape[0]
    scaffold = fibonacci_ball_scaffold(n, R, m.rho, device=dev, dtype=irel.dtype)
    order = cavity_order(irel, R); xo, so = irel[order], s_in[order]
    combined = torch.cat([bnd, xo]); scomb = torch.cat([s_bnd, so])
    kind = torch.cat([torch.ones(mm, dtype=torch.long, device=dev), torch.zeros(n, dtype=torch.long, device=dev)])
    idxp = torch.arange(mm + n, device=dev)
    valid = idxp[None] < (mm + torch.arange(n, device=dev))[:, None]
    origin_mask = valid & (idxp[None] >= mm)                       # interior-prefix-only origin (matches model)
    origin = m._origins(combined, origin_mask, scaffold)
    h, Rf = m._frame_context(combined, scomb, kind, valid, origin, scaffold)
    return xo, so, origin, h, Rf


@torch.no_grad()
def frame_to_world(Rf, off_local, origin):
    return origin + torch.einsum('naj,na->nj', Rf, off_local)


@torch.no_grad()
def place(h, Rf, so, origin, mode):
    e = m.sp_out_emb(so)
    la = m.head_a(h + e); lb_of = lambda ba: m.head_b(h + e + m.bin_a_emb(ba))
    lc_of = lambda ba, bb: m.head_c(h + e + m.bin_a_emb(ba) + m.bin_b_emb(bb))
    if mode == "argmax":
        ba = la.argmax(-1); bb = lb_of(ba).argmax(-1); bc = lc_of(ba, bb).argmax(-1)
    else:
        ba = torch.multinomial(F.softmax(la, -1), 1).squeeze(-1)
        bb = torch.multinomial(F.softmax(lb_of(ba), -1), 1).squeeze(-1)
        bc = torch.multinomial(F.softmax(lc_of(ba, bb), -1), 1).squeeze(-1)
    off = torch.stack([m._bin_center(ba), m._bin_center(bb), m._bin_center(bc)], -1) \
        + (torch.rand(h.shape[0], 3, device=dev) - 0.5) * m.bin_w
    return frame_to_world(Rf, off, origin)                        # local-frame offset -> world


def energy_of(center, irel, s_in, xout, sout):
    N = irel.shape[0] + xout.shape[0]
    x = torch.empty(1, N, 3, device=dev); s = torch.empty(1, N, device=dev, dtype=torch.long)
    x[0, :irel.shape[0]] = torch.remainder(center + irel, L); x[0, irel.shape[0]:] = torch.remainder(center + xout, L)
    s[0, :irel.shape[0]] = s_in; s[0, irel.shape[0]:] = sout
    return (ka_energy(x, s, L) / N).item()


import statistics as st
res = {k: {"U": [], "cl": []} for k in ("oracle", "mode", "sample")}
tru = []
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
        ut = energy_of(center, irel, p["s_in"], ball, p["s_out"]); tru.append(ut)
        xo, so, origin, h, Rf = context(irel, p["s_in"], ball[shell], p["s_out"][shell], R)
        # oracle: true residual bins (residual expressed in the LOCAL frame, then mapped back)
        abc = torch.einsum('naj,nj->na', Rf, xo - origin)
        ba, bb, bc = m._bin(abc[:, 0]), m._bin(abc[:, 1]), m._bin(abc[:, 2])
        orc_local = torch.stack([m._bin_center(ba), m._bin_center(bb), m._bin_center(bc)], -1) \
            + (torch.rand(xo.shape[0], 3, device=dev) - 0.5) * m.bin_w
        orc = frame_to_world(Rf, orc_local, origin)
        for name, gi in (("oracle", orc), ("mode", place(h, Rf, so, origin, "argmax")), ("sample", place(h, Rf, so, origin, "sample"))):
            u = energy_of(center, gi, so, ball, p["s_out"])
            res[name]["U"].append(u); res[name]["cl"].append(u > ut + 2)

n = len(tru)
print(f"held cavities={n}  TRUE U/N median={st.median(tru):+.3f}", flush=True)
for k in ("oracle", "mode", "sample"):
    print(f"  {k:7s}: U/N median={st.median(res[k]['U']):+.3f}  clashed={sum(res[k]['cl'])}/{n}", flush=True)
