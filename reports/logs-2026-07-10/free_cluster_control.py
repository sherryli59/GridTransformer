"""CONTROL (user question): is the CAVITY CONDITIONING the problem, or is clash intrinsic to the AR
machinery? Train the SAME 3D local-frame AR on FREE CLUSTERS (interior particles, NO frozen boundary)
and measure oracle/mode/sample intra-cluster clash.

  free cluster CLEAN (mode low-clash)  => the frozen-boundary conditioning breaks generation.
  free cluster CLASHES (mode ~60/60)   => intrinsic machinery problem (order/frame/head), NOT the boundary
                                          (bulk KA localframe = a free-cluster generator, got 8.6% clash).
"""
import time, statistics as st
import torch
import torch.nn.functional as F
from liquid_coupling_flow.ka3d_cavity_ar import KA3DCavityAR, fibonacci_ball_scaffold, cavity_order, _mic
from liquid_coupling_flow.ka3d_cavity_carve import carve
from liquid_coupling_flow.ka_energy import ka_energy

dev = "cuda" if torch.cuda.is_available() else "cpu"
Lbox, R = 100.0, 2.0                                     # big box => intra-cluster LJ only (no PBC/boundary)
d = torch.load("liquid_coupling_flow/artifacts/ka3d_dataset_N512_T0.5.pt", map_location=dev, weights_only=False)
X, S, L = d["x"].to(dev).float(), d["s"].to(dev).long(), float(d["L"])
E3 = torch.zeros(0, 3, device=dev); E1 = torch.zeros(0, dtype=torch.long, device=dev)
torch.manual_seed(0)


def clusters(idxs):
    out = []
    for i in idxs:
        center = X[i][(37 * i + 5) % X.shape[1]].clone()
        p = carve(X[i], S[i], center, R, L)
        if p["n_in"] < 6:
            continue
        irel = _mic(p["x_in"], center, L)
        nB = int((p["s_in"] == 1).sum())
        out.append(dict(irel=irel, s=p["s_in"], n=p["n_in"], nA=p["n_in"] - nB, nB=nB))
    return out


def cluster_U(irel, s):
    return (ka_energy(irel[None], s.long()[None], Lbox) / irel.shape[0]).item()


train = clusters(range(12))
print(f"free-cluster control: {len(train)} clusters, n={[c['n'] for c in train]}, "
      f"true U/N={[round(cluster_U(c['irel'], c['s']), 2) for c in train]}", flush=True)

m = KA3DCavityAR().to(dev)
opt = torch.optim.Adam(m.parameters(), lr=3e-4); t0 = time.time()
for step in range(3000):
    idx = torch.randint(len(train), (4,))
    loss = sum(-m.log_prob_pair(train[i]["irel"], train[i]["s"], E3, E1, R) / train[i]["n"] for i in idx) / 4
    opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 5.0); opt.step()
    if step % 500 == 0 or step == 2999:
        with torch.no_grad():
            full = sum(-m.log_prob_pair(c["irel"], c["s"], E3, E1, R).item() / c["n"] for c in train) / len(train)
        print(f"  step {step:4d}  -logp/n={full:+.3f}  ({time.time()-t0:.0f}s)", flush=True)

m.eval()


@torch.no_grad()
def context(irel, s):
    n = irel.shape[0]
    scaffold = fibonacci_ball_scaffold(n, R, m.rho, device=dev, dtype=irel.dtype)
    order = cavity_order(irel, R); xo, so = irel[order], s[order]
    combined = xo; scomb = so; kind = torch.zeros(n, dtype=torch.long, device=dev)
    idxp = torch.arange(n, device=dev)
    valid = idxp[None] < torch.arange(n, device=dev)[:, None]         # interior prefix (m=0)
    origin = m._origins(combined, valid, scaffold)
    h, Rf = m._frame_context(combined, scomb, kind, valid, origin, scaffold)
    return xo, so, origin, h, Rf


@torch.no_grad()
def place(h, Rf, so, origin, mode):
    e = m.sp_out_emb(so)
    la = m.head_a(h + e)
    if mode == "argmax":
        ba = la.argmax(-1); bb = m.head_b(h + e + m.bin_a_emb(ba)).argmax(-1)
        bc = m.head_c(h + e + m.bin_a_emb(ba) + m.bin_b_emb(bb)).argmax(-1)
    else:
        ba = torch.multinomial(F.softmax(la, -1), 1).squeeze(-1)
        bb = torch.multinomial(F.softmax(m.head_b(h + e + m.bin_a_emb(ba)), -1), 1).squeeze(-1)
        bc = torch.multinomial(F.softmax(m.head_c(h + e + m.bin_a_emb(ba) + m.bin_b_emb(bb)), -1), 1).squeeze(-1)
    off = torch.stack([m._bin_center(ba), m._bin_center(bb), m._bin_center(bc)], -1) \
        + (torch.rand(h.shape[0], 3, device=dev) - 0.5) * m.bin_w
    return origin + torch.einsum('naj,na->nj', Rf, off)


res = {k: {"U": [], "cl": []} for k in ("oracle", "mode", "sample")}
tru = []
with torch.no_grad():
    for c in train:
        xo, so, origin, h, Rf = context(c["irel"], c["s"])
        ut = cluster_U(xo, so); tru.append(ut)
        abc = torch.einsum('naj,nj->na', Rf, xo - origin)
        ba, bb, bc = m._bin(abc[:, 0]), m._bin(abc[:, 1]), m._bin(abc[:, 2])
        orc = origin + torch.einsum('naj,na->nj', Rf, torch.stack([m._bin_center(ba), m._bin_center(bb),
              m._bin_center(bc)], -1) + (torch.rand(xo.shape[0], 3, device=dev) - 0.5) * m.bin_w)
        for name, gi in (("oracle", orc), ("mode", place(h, Rf, so, origin, "argmax")),
                         ("sample", place(h, Rf, so, origin, "sample"))):
            u = cluster_U(gi, so); res[name]["U"].append(u); res[name]["cl"].append(u > ut + 2)

n = len(tru)
print(f"\nfree clusters={n}  TRUE U/N median={st.median(tru):+.3f}", flush=True)
for k in ("oracle", "mode", "sample"):
    print(f"  {k:7s}: U/N median={st.median(res[k]['U']):+.3f}  clashed={sum(res[k]['cl'])}/{n}", flush=True)
print("\nVERDICT: mode clean => boundary conditioning is the problem. mode clashes => intrinsic machinery.", flush=True)
