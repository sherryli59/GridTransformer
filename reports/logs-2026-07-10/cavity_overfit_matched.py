"""MATCHED control: overfit 12 CAVITIES (boundary included) with the SAME model/steps as
free_cluster_control.py (12 free clusters). Removes the overfit-vs-generalize confound so the ONLY
difference is the frozen boundary. Reports mode/sample clash + int<->int vs int<->boundary decomposition.
Compare mode/sample U/N to the free cluster (mode +43, sample +3.5)."""
import time, statistics as st
import torch
import torch.nn.functional as F
from liquid_coupling_flow.ka3d_cavity_ar import KA3DCavityAR, _mic
from liquid_coupling_flow.ka3d_cavity_carve import carve
from liquid_coupling_flow.ka_energy import ka_energy

dev = "cuda" if torch.cuda.is_available() else "cpu"
d = torch.load("liquid_coupling_flow/artifacts/ka3d_dataset_N512_T0.5.pt", map_location=dev, weights_only=False)
X, S, L = d["x"].to(dev).float(), d["s"].to(dev).long(), float(d["L"])
r_ctx, R, CLASH = 2.5, 2.0, 0.8
torch.manual_seed(0)

pairs = []
for i in range(12):                                                      # SAME 12 base configs as free-cluster run
    center = X[i][(37 * i + 5) % X.shape[1]].clone()
    p = carve(X[i], S[i], center, R, L)
    if p["n_in"] < 6:
        continue
    irel = _mic(p["x_in"], center, L); ball = _mic(p["x_out"], center, L)
    shell = ball.norm(dim=-1) < R + r_ctx
    nB = int((p["s_in"] == 1).sum())
    pairs.append(dict(irel=irel, s=p["s_in"], bnd=ball[shell], s_bnd=p["s_out"][shell], n=p["n_in"],
                      nA=p["n_in"] - nB, nB=nB, xout=ball, sout=p["s_out"], center=center))
print(f"matched cavity overfit: {len(pairs)} cavities n={[p['n'] for p in pairs]}", flush=True)


def full_U(center, irel, s_in, xout, sout):
    N = irel.shape[0] + xout.shape[0]
    x = torch.empty(1, N, 3, device=dev); s = torch.empty(1, N, device=dev, dtype=torch.long)
    x[0, :irel.shape[0]] = torch.remainder(center + irel, L); x[0, irel.shape[0]:] = torch.remainder(center + xout, L)
    s[0, :irel.shape[0]] = s_in; s[0, irel.shape[0]:] = sout
    return (ka_energy(x, s, L) / N).item()


m = KA3DCavityAR().to(dev)
opt = torch.optim.Adam(m.parameters(), lr=3e-4); t0 = time.time()
for step in range(3000):
    idx = torch.randint(len(pairs), (4,))
    loss = sum(-m.log_prob_pair(pairs[i]["irel"], pairs[i]["s"], pairs[i]["bnd"], pairs[i]["s_bnd"], R) / pairs[i]["n"]
               for i in idx) / 4
    opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 5.0); opt.step()
    if step % 500 == 0 or step == 2999:
        with torch.no_grad():
            full = sum(-m.log_prob_pair(p["irel"], p["s"], p["bnd"], p["s_bnd"], R).item() / p["n"] for p in pairs) / len(pairs)
        print(f"  step {step:4d}  -logp/n={full:+.3f}  ({time.time()-t0:.0f}s)", flush=True)

m.eval()
tru, mo, sa = [], [], []
ii = ib = tot = 0
with torch.no_grad():
    for p in pairs:
        tru.append(full_U(p["center"], p["irel"], p["s"], p["xout"], p["sout"]))
        gi, gs = m.sample_pair(p["bnd"], p["s_bnd"], p["nA"], p["nB"], R)
        sa.append(full_U(p["center"], gi, gs, p["xout"], p["sout"]))
        # decomposition on the sampled interior
        d_int = (torch.cdist(gi, gi) + torch.eye(gi.shape[0], device=dev) * 1e9).min(1).values
        d_bnd = torch.cdist(gi, p["bnd"]).min(1).values
        ii += int((d_int < CLASH).sum()); ib += int((d_bnd < CLASH).sum()); tot += gi.shape[0]

print(f"\n[MATCHED cavity overfit]  TRUE U/N median={st.median(tru):+.3f}", flush=True)
print(f"  sample U/N median={st.median(sa):+.2f}   (free-cluster overfit was +3.5)", flush=True)
print(f"  sampled interior clashing with INTERIOR : {ii}/{tot} ({100*ii/tot:.1f}%)", flush=True)
print(f"  sampled interior clashing with BOUNDARY : {ib}/{tot} ({100*ib/tot:.1f}%)", flush=True)
print(f"\nVERDICT: cavity sample U/N >> +3.5 AND boundary clashes high => BOUNDARY breaks it. "
      f"cavity ~ +3.5 => matched, boundary not the issue.", flush=True)
