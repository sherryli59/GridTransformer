"""DECISIVE: overfit 12 cavities (two-stream model), then on the MEMORIZED cavities decompose
generation clash into: ORACLE (true bins), TF-SAMPLE (true prefix/frame/origin, SAMPLED bins =
per-step conditional, NO rollout drift), FR-SAMPLE (full rollout). Determines the failure locus:
  oracle clean + TF-sample clean + FR clashes => ROLLOUT DRIFT (no per-step boundary-feed fix helps;
                                                 lever = energy corrector / scheduled sampling).
  TF-sample also clashes                       => PER-STEP conditional (a feed fix could still help).
"""
import time, statistics as st
import torch
import torch.nn.functional as F
from liquid_coupling_flow.ka3d_cavity_ar import KA3DCavityAR, fibonacci_ball_scaffold, cavity_order, _mic
from liquid_coupling_flow.ka3d_cavity_carve import carve
from liquid_coupling_flow.ka_energy import ka_energy

dev = "cuda" if torch.cuda.is_available() else "cpu"
d = torch.load("liquid_coupling_flow/artifacts/ka3d_dataset_N512_T0.5.pt", map_location=dev, weights_only=False)
X, S, L = d["x"].to(dev).float(), d["s"].to(dev).long(), float(d["L"])
r_ctx, R = 2.5, 2.0
torch.manual_seed(0)

pairs = []
for i in range(12):
    center = X[i][(37 * i + 5) % X.shape[1]].clone()
    p = carve(X[i], S[i], center, R, L)
    if p["n_in"] < 6:
        continue
    irel = _mic(p["x_in"], center, L); ball = _mic(p["x_out"], center, L)
    shell = ball.norm(dim=-1) < R + r_ctx
    nB = int((p["s_in"] == 1).sum())
    pairs.append(dict(irel=irel, s=p["s_in"], bnd=ball[shell], s_bnd=p["s_out"][shell], n=p["n_in"],
                      nA=p["n_in"] - nB, nB=nB, xout=ball, sout=p["s_out"], center=center))

m = KA3DCavityAR().to(dev)
opt = torch.optim.Adam(m.parameters(), lr=3e-4); t0 = time.time()
for step in range(3000):
    idx = torch.randint(len(pairs), (4,))
    loss = sum(-m.log_prob_pair(pairs[i]["irel"], pairs[i]["s"], pairs[i]["bnd"], pairs[i]["s_bnd"], R) / pairs[i]["n"]
               for i in idx) / 4
    opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 5.0); opt.step()
print(f"overfit done -logp/n(last batch)={float(loss):+.3f} ({time.time()-t0:.0f}s)", flush=True)
m.eval()


def full_U(center, irel, s_in, xout, sout):
    N = irel.shape[0] + xout.shape[0]
    x = torch.empty(1, N, 3, device=dev); s = torch.empty(1, N, device=dev, dtype=torch.long)
    x[0, :irel.shape[0]] = torch.remainder(center + irel, L); x[0, irel.shape[0]:] = torch.remainder(center + xout, L)
    s[0, :irel.shape[0]] = s_in; s[0, irel.shape[0]:] = sout
    return (ka_energy(x, s, L) / N).item()


@torch.no_grad()
def tf_place(irel, s_in, bnd, s_bnd, sample):
    """Teacher-forced: TRUE prefix -> true origins/frame/context. sample=False -> oracle (true bins);
    sample=True -> per-step conditional sample. Returns generated interior (center-rel)."""
    n, mm = irel.shape[0], bnd.shape[0]
    scaffold = fibonacci_ball_scaffold(n, R, m.rho, device=dev, dtype=irel.dtype)
    order = cavity_order(irel, R); xo, so = irel[order], s_in[order]
    combined = torch.cat([bnd, xo]); scomb = torch.cat([s_bnd, so])
    kind = torch.cat([torch.ones(mm, dtype=torch.long, device=dev), torch.zeros(n, dtype=torch.long, device=dev)])
    idxp = torch.arange(mm + n, device=dev)
    valid = idxp[None] < (mm + torch.arange(n, device=dev))[:, None]
    origin_mask = valid & (idxp[None] >= mm)
    origin = m._origins(combined, origin_mask, scaffold)
    h, Rf = m._frame_context(combined, scomb, kind, valid, origin, scaffold)
    e = m.sp_out_emb(so)
    if sample:
        ba = torch.multinomial(F.softmax(m.head_a(h + e), -1), 1).squeeze(-1)
        bb = torch.multinomial(F.softmax(m.head_b(h + e + m.bin_a_emb(ba)), -1), 1).squeeze(-1)
        bc = torch.multinomial(F.softmax(m.head_c(h + e + m.bin_a_emb(ba) + m.bin_b_emb(bb)), -1), 1).squeeze(-1)
    else:
        abc = torch.einsum('naj,nj->na', Rf, xo - origin)
        ba, bb, bc = m._bin(abc[:, 0]), m._bin(abc[:, 1]), m._bin(abc[:, 2])
    off = torch.stack([m._bin_center(ba), m._bin_center(bb), m._bin_center(bc)], -1) \
        + (torch.rand(n, 3, device=dev) - 0.5) * m.bin_w
    return origin + torch.einsum('naj,na->nj', Rf, off), so


rows = {k: [] for k in ("oracle", "tf_sample", "fr_sample")}
tru = []
with torch.no_grad():
    for p in pairs:
        ut = full_U(p["center"], p["irel"], p["s"], p["xout"], p["sout"]); tru.append(ut)
        go, so0 = tf_place(p["irel"], p["s"], p["bnd"], p["s_bnd"], sample=False)
        gt, sot = tf_place(p["irel"], p["s"], p["bnd"], p["s_bnd"], sample=True)
        gf, sof = m.sample_pair(p["bnd"], p["s_bnd"], p["nA"], p["nB"], R)
        rows["oracle"].append(full_U(p["center"], go, so0, p["xout"], p["sout"]))
        rows["tf_sample"].append(full_U(p["center"], gt, sot, p["xout"], p["sout"]))
        rows["fr_sample"].append(full_U(p["center"], gf, sof, p["xout"], p["sout"]))

n = len(tru)
print(f"\nmemorized cavities={n}  TRUE U/N median={st.median(tru):+.3f}", flush=True)
for k in ("oracle", "tf_sample", "fr_sample"):
    u = rows[k]; cl = sum(x > st.median(tru) + 2 for x in u)
    print(f"  {k:10s}: U/N median={st.median(u):+.2f}  clashed={cl}/{n}", flush=True)
print("\noracle+tf_sample clean, fr bad => ROLLOUT DRIFT (corrector). tf_sample bad => per-step.", flush=True)
