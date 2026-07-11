"""Does multi-try (MTM-style: draw N candidates, take the best) already give good acceptance at
K=1, matching the user's prior 2D-KA MTM result? If yes -> no bug, just single-try-vs-MTM mismatch
(+ this 3D model is far less trained than that campaign's). If MTM-of-many STILL fails -> real problem."""
import shutil, statistics as st
from pathlib import Path
import torch
from liquid_coupling_flow.ka3d_cavity_ar import _mic
from liquid_coupling_flow.ka3d_cavity_carve import carve
from liquid_coupling_flow.ka3d_scaffold_ar import KA3DScaffoldAR, label_to_scaffold, fixed_ball_scaffold
from liquid_coupling_flow.ka3d_block import block_log_prob, sample_block
from liquid_coupling_flow.ka_energy import ka_energy

dev = "cuda"
SCR = Path("/tmp/claude-1002/-mnt-ssd-GridTransformer/271167b9-117a-400e-b11f-c2c1bfa0b6df/scratchpad")
ckp = SCR / "diag_mtm_ckpt.pt"; shutil.copy("liquid_coupling_flow/artifacts/ka3d_blob_ar.pt", ckp)
ck = torch.load(ckp, map_location=dev, weights_only=False)
m = KA3DScaffoldAR().to(dev); m.load_state_dict(ck["state_dict"]); m.eval()
print(f"ckpt step={ck['step']}", flush=True)

d = torch.load("liquid_coupling_flow/artifacts/ka3d_dataset_N512_T0.5.pt", map_location=dev, weights_only=False)
X, S, L = d["x"].to(dev).float(), d["s"].to(dev).long(), float(d["L"])
R, beta, NTRY = 2.0, 2.0, 16
gen = torch.Generator(device=dev).manual_seed(5)
E3, E1 = torch.zeros(0, 3, device=dev), torch.zeros(0, dtype=torch.long, device=dev)


def energy_total(xo, so):
    return ka_energy(xo[None], so.long()[None], 100.0)[0]


for K in (1, 4):
    any_positive, best_las = [], []
    with torch.no_grad():
        for ci in range(900, 930):
            center = torch.rand(3, generator=gen, device=dev) * L
            p = carve(X[ci], S[ci], center, R, L)
            if p["n_in"] < K + 2:
                continue
            x, s, _ = label_to_scaffold(_mic(p["x_in"], center, L), p["s_in"], R)
            n = x.shape[0]
            anchors = fixed_ball_scaffold(n, R, dev)
            seed = int(torch.randint(n, (), generator=gen, device=dev))
            idx = (anchors - anchors[seed]).norm(dim=-1).topk(K, largest=False).indices
            blk = torch.zeros(n, dtype=torch.bool, device=dev); blk[idx] = True
            u_old = energy_total(x, s)
            lq_rev = block_log_prob(m, x, s, blk, E3, E1, R)
            las = []
            for _ in range(NTRY):
                xn, sn, lq_fwd = sample_block(m, x, s, blk, E3, E1, R, gen=gen)
                u_new = energy_total(xn, sn)
                las.append(float(-beta * (u_new - u_old) + (lq_rev - lq_fwd)))
            best_las.append(max(las)); any_positive.append(max(las) > 0)
    n_ev = len(best_las)
    print(f"K={K}: sites={n_ev}  P(any-of-{NTRY} accepted)={sum(any_positive)}/{n_ev} "
          f"({100*sum(any_positive)/n_ev:.0f}%)  median-best-log_alpha={st.median(best_las):+.2f}", flush=True)
print("\nIf K=1 any-of-N is high -> MTM viable now, mismatch (not bug) confirmed; "
      "gap will shrink further with more training.", flush=True)
