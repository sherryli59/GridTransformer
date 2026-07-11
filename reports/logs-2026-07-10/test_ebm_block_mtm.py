"""EBM (learned-potential) block-MTM K-sweep on 2D KA, suffix blob, mirroring test_2d_block_mtm.py exactly
so it is apples-to-apples with the factorized baseline (k=4 30%/1.15, k=8 5%/2.70). Also gates EXACTNESS:
the tilted-b block sampler's logq must equal the scorer to fp precision. Pass a checkpoint as argv[1]
(default the fine-tuned EBM); load the warm factorized ckpt to confirm phi~0 reproduces the baseline."""
import statistics as st, sys
import torch
import torch.nn.functional as F
from liquid_coupling_flow.ka_localframe_ebm import KALocalFrameEBM
from liquid_coupling_flow.ka_gridformer import _wrap_pm
from liquid_coupling_flow.ka_energy import ka_energy

dev = "cuda"
ckpath = sys.argv[1] if len(sys.argv) > 1 else "liquid_coupling_flow/artifacts/ka_localframe_ebm_N100.pt"
ck = torch.load(ckpath, map_location=dev, weights_only=False)
m = KALocalFrameEBM(rho=ck["rho"], n_bins=ck["n_bins"], knn=ck["knn"]).to(dev)
m.load_state_dict(ck["state_dict"], strict=False); m.eval()
print(f"loaded {ckpath}", flush=True)
N = 100
ref = torch.load(f"liquid_coupling_flow/artifacts/ka_reference_N{N}.pt", map_location=dev, weights_only=False)
data = ref["x"].to(dev); s0 = ref["s"].to(dev).long(); L = ref["L"]; beta = 2.0
sc = m.geo._scaffold(N, dev); arc = m._arc_scale(N)
bc = m._bin_center(torch.arange(m.n_bins, device=dev))


def block_sample(pos0, sp0, k, B):
    return m.block_sample_suffix(pos0, sp0, k, B, sc, L)


def block_logp(pos_in, sp0, k):
    return m.block_logp_suffix(pos_in, sp0, k, sc, L)


def clash_in_block(pos, k):
    block = pos[N - k:]
    d = pos[None] - block[:, None]; d = d - L * torch.round(d / L)
    dd = d.norm(dim=-1); dd[torch.arange(k), torch.arange(N - k, N)] = 1e9
    return int((dd.min(1).values < 0.8).sum())


def energy(pos, sp):
    return float(ka_energy(pos[None], sp.long()[None], L)[0])


# ---- exactness gate: sampler logq == scorer logp on the SAME sampled block ----
order = m.geo._curve_order(data[0:1], N)[0]; pos0 = data[0][order]; sp0 = s0[order]
pt, lqs = block_sample(pos0, sp0, 6, 8)
lpr = block_logp(pt, sp0, 6)
err = (lqs - lpr).abs().max().item()
print(f"EXACTNESS |logq_sample - logp_score| = {err:.2e}  -> {'PASS' if err < 1e-3 else 'FAIL'}", flush=True)

for k in (4, 8):
    clashfree, nclash, mtm = [], [], []
    for ci in range(20):
        order = m.geo._curve_order(data[ci:ci + 1], N)[0]
        pos0 = data[ci][order]; sp0 = s0[order]
        u_old = energy(pos0, sp0); lq_x = float(block_logp(pos0[None], sp0, k))
        pos_t, lq_t = block_sample(pos0, sp0, k, 16)
        lus = [-beta * u_old - lq_x]
        for t in range(16):
            lus.append(-beta * energy(pos_t[t], sp0) - float(lq_t[t]))
            if t == 0:
                nc = clash_in_block(pos_t[t], k); nclash.append(nc); clashfree.append(nc == 0)
        lu = torch.tensor(lus[1:], device=dev); sfwd = torch.logsumexp(lu, 0)
        js = int(torch.multinomial(torch.softmax(lu, 0), 1)); lu_rev = lu.clone(); lu_rev[js] = lus[0]
        mtm.append(float(torch.rand((), device=dev).log() < (sfwd - torch.logsumexp(lu_rev, 0))))
    n = len(mtm)
    print(f"EBM k={k}: clash-free={100*sum(clashfree)/n:.0f}%  mean #clash={st.mean(nclash):.2f}  "
          f"MTM accept={100*sum(mtm)/n:.0f}%   (factorized: k=4 30%/1.15, k=8 5%/2.70)", flush=True)
