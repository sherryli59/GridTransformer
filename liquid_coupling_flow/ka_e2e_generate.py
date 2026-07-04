"""E2E pipeline stage A+B: generate from the N=100-trained joint flow at target N (default 256; pass N as
argv[1], e.g. `python -m liquid_coupling_flow.ka_e2e_generate 100`), SAVE the raw generation, analyze U/N +
species-oracle disagreement + full partial g(r) vs the PT reference.
Output: artifacts/ka_e2e_gen_N{N}.pt {x, s, U, disagree, meta} + ka_e2e_gen_N{N}.png"""
import os, sys, time, torch, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from liquid_coupling_flow.ipl44.joint_flow import JointSpeciesFlow, random_22_labeling
from liquid_coupling_flow.ka_energy import ka_energy
from liquid_coupling_flow.ka_observables import partial_gr

dev = "cuda"
N = int(sys.argv[1]) if len(sys.argv) > 1 else 256
ART = os.path.join(os.path.dirname(__file__), "artifacts")
JFD = os.path.join(os.path.dirname(__file__), "ipl44", "data")
ref = torch.load(f"{ART}/ka_reference_N{N}_thin.pt", map_location="cpu", weights_only=False)
L = float(ref["L"])
xr = torch.remainder(ref["x"].float(), L); sr = ref["s"].long()
nB = int(sr.sum())
ck = torch.load(f"{JFD}/jf_ka100tt_best.pt", map_location=dev, weights_only=False); cfg = ck["cfg"]
jf = JointSpeciesFlow(n_particles=N, L=L, hidden_nf=cfg["hidden_nf"], n_layers=cfg["n_layers"], two_time=True).to(dev)
jf.load_state_dict(ck["state_dict"]); jf.eval(); jf.knn = 32
S_CANON = torch.zeros(1, N, dtype=torch.long, device=dev); S_CANON[:, :nB] = 1

B = 1024; xs = []; ss = []
t0 = time.time()
while B >= 32:
    try:
        x0 = torch.rand(B, N, 2, device=dev) * L
        s0 = random_22_labeling(B, N, nB, dev)
        xg, sg = jf.sample(x0, s0, n_steps=250)
        xs.append(xg); ss.append(sg)
        break
    except torch.OutOfMemoryError:
        torch.cuda.empty_cache(); B //= 2; print(f"OOM -> B={B}", flush=True)
# top up to >=512 total with the found batch size
while sum(a.shape[0] for a in xs) < 512 and B >= 32:
    x0 = torch.rand(B, N, 2, device=dev) * L
    s0 = random_22_labeling(B, N, nB, dev)
    xg, sg = jf.sample(x0, s0, n_steps=250)
    xs.append(xg); ss.append(sg)
xg = torch.cat(xs); sg = torch.cat(ss)
U = ka_energy(xg, sg, L)
with torch.no_grad():
    Bx = xg.shape[0]; W = []
    for i in range(0, Bx, 128):
        _, lg = jf(torch.ones(min(128, Bx - i), 1, 1, device=dev), xg[i:i + 128],
                   S_CANON.expand(min(128, Bx - i), -1),
                   t_spec=torch.zeros(min(128, Bx - i), 1, 1, device=dev))
        W.append(torch.softmax(lg, -1)[..., 1])
    W = torch.cat(W)
disagree = ((W > 0.5).long() != sg).float().mean()
torch.save({"x": xg.cpu(), "s": sg.cpu(), "U": U.cpu(), "disagree_per_cfg": ((W > 0.5).long() != sg).float().mean(1).cpu(),
            "meta": {"model": "jf_ka100tt_best", "knn": 32, "n_steps": 250, "N": N, "L": L, "gen_s": time.time() - t0}},
           f"{ART}/ka_e2e_gen_N{N}.pt")
Ur = ka_energy(xr[-1024:].to(dev), sr[None].expand(1024, -1).to(dev), L)
print(f"generated {xg.shape[0]} configs at N={N} in {time.time()-t0:.0f}s (batch {B})", flush=True)
print(f"gen  U/N: median {float(U.median())/N:.4f} mean {float(U.mean())/N:.4f} | ref median {float(Ur.median())/N:.4f}", flush=True)
print(f"gen  species-oracle disagreement: {float(disagree):.4f}", flush=True)

fig, ax = plt.subplots(1, 4, figsize=(19, 4.2))
ax[0].hist((Ur / N).cpu().numpy(), bins=45, density=True, alpha=0.6, color="k", label="PT ref")
ax[0].hist((U / N).cpu().numpy(), bins=45, density=True, alpha=0.6, color="tab:red", label="flow gen (raw)")
ax[0].set_title(f"U/N distribution: raw generation vs PT ref (N={N})"); ax[0].legend(fontsize=8)
for j, (pair, nm) in enumerate([((0, 0), "gAA"), ((0, 1), "gAB"), ((1, 1), "gBB")]):
    r, g = partial_gr(xr[-1024:], sr[None].expand(1024, -1).contiguous(), L, rmax=3.5, nbins=100, pair=pair)
    ax[j + 1].plot(r, g, "k-", lw=2, label="PT ref")
    r2, g2 = partial_gr(xg.cpu(), sg.cpu(), L, rmax=3.5, nbins=100, pair=pair)
    ax[j + 1].plot(r2, g2, "r--", label="flow gen (raw)")
    ax[j + 1].set_title(f"{nm}(r)"); ax[j + 1].legend(fontsize=8); ax[j + 1].set_xlabel("r")
plt.tight_layout(); plt.savefig(f"{ART}/ka_e2e_gen_N{N}.png", dpi=110)
print(f"saved {ART}/ka_e2e_gen_N{N}.pt + .png", flush=True)
