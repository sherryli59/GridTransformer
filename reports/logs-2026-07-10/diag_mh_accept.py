"""Why is mh_accept pinned at 0%? Mechanical check: is it explained by block_clash being high
(LJ core makes any clash's ΔU astronomically positive -> exp(-b*dU)~0), or is something else
wrong? Compares block sizes k=1 (proven full-cage heat-bath scale) vs k=4 (current eval) on the
LATEST checkpoint, reports the actual log_alpha / dU distribution (not just accept/reject)."""
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
ckp = SCR / "diag_mh_ckpt.pt"; shutil.copy("liquid_coupling_flow/artifacts/ka3d_blob_ar.pt", ckp)
ck = torch.load(ckp, map_location=dev, weights_only=False)
m = KA3DScaffoldAR().to(dev); m.load_state_dict(ck["state_dict"]); m.eval()
print(f"ckpt step={ck['step']}", flush=True)

d = torch.load("liquid_coupling_flow/artifacts/ka3d_dataset_N512_T0.5.pt", map_location=dev, weights_only=False)
X, S, L = d["x"].to(dev).float(), d["s"].to(dev).long(), float(d["L"])
R, beta = 2.0, 2.0
gen = torch.Generator(device=dev).manual_seed(3)
E3, E1 = torch.zeros(0, 3, device=dev), torch.zeros(0, dtype=torch.long, device=dev)


def energy_total(xo, so):
    return ka_energy(xo[None], so.long()[None], 100.0)[0]


for K in (1, 2, 4, 8):
    dUs, log_alphas, accepts, clash_frees = [], [], [], []
    with torch.no_grad():
        for ci in range(900, 920):
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
            xn, sn, lq_fwd = sample_block(m, x, s, blk, E3, E1, R, gen=gen)
            u_new = energy_total(xn, sn)
            dU = float(u_new - u_old)
            la = float(-beta * dU + (lq_rev - lq_fwd))
            dUs.append(dU); log_alphas.append(la); accepts.append(la > 0)
            bi = xn[blk]; dd = torch.cdist(bi, xn); dd[torch.arange(K), idx] = 1e9
            clash_frees.append(bool((dd.min(1).values >= 0.8).all()))
    n_ev = len(dUs)
    print(f"\nK={K}  (n={n_ev} proposals)", flush=True)
    print(f"  dU        : median={st.median(dUs):+.2f}  min={min(dUs):+.2f}  max={max(dUs):+.2e}", flush=True)
    print(f"  log_alpha : median={st.median(log_alphas):+.2f}  best={max(log_alphas):+.2f}", flush=True)
    print(f"  clash-free proposals: {sum(clash_frees)}/{n_ev}  ({100*sum(clash_frees)/n_ev:.0f}%)", flush=True)
    print(f"  P(accept) if drawn once: {sum(a for a in accepts)}/{n_ev}", flush=True)
print("\nRead: if clash-free proposals ALSO have log_alpha<<0 -> not just clash, systematic dU offset. "
      "If clash-free proposals have log_alpha~0 -> pure clash-rate problem, smaller K should already accept.",
      flush=True)
