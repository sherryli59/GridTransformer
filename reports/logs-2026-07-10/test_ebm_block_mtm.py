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


@torch.no_grad()
def block_sample(pos0, sp0, k, B):
    pos = pos0[None].expand(B, N, 2).clone(); sp = sp0[None].expand(B, N).clone()
    logq = torch.zeros(B, device=dev)
    for j in range(N - k, N):
        h, origin, nr, nsp, val = m._step_ebm(pos, sp, sc[j], j, L)
        la = F.log_softmax(m.head_a(h), -1); ba = torch.multinomial(la.exp(), 1).squeeze(-1)
        base_b = m.head_b(h + m.bin_a_emb(ba))
        V = m._V_b(ba, nr, nsp, val, sp[:, j], arc, bc)
        lb = F.log_softmax(base_b - V, -1); bb = torch.multinomial(lb.exp(), 1).squeeze(-1)
        logq += la.gather(-1, ba[:, None]).squeeze(-1) + lb.gather(-1, bb[:, None]).squeeze(-1)
        a = m._bin_center(ba) + (torch.rand(B, device=dev) - 0.5) * m.bin_w
        b = m._bin_center(bb) + (torch.rand(B, device=dev) - 0.5) * m.bin_w
        pos[:, j] = torch.remainder(origin + torch.stack([a, b], -1) * arc, L)
    return pos, logq


@torch.no_grad()
def block_logp(pos_in, sp0, k):
    """logq of the block in pos_in (any config) under the same tilted head. Works for B trials."""
    B = pos_in.shape[0]; pos = pos_in.clone(); sp = sp0[None].expand(B, N).clone()
    lp = torch.zeros(B, device=dev)
    for j in range(N - k, N):
        h, origin, nr, nsp, val = m._step_ebm(pos, sp, sc[j], j, L)
        ab = _wrap_pm(pos[:, j] - origin, L) / arc
        ba = m._bin(ab[..., 0]); bb = m._bin(ab[..., 1])
        la = F.log_softmax(m.head_a(h), -1)
        base_b = m.head_b(h + m.bin_a_emb(ba)); V = m._V_b(ba, nr, nsp, val, sp[:, j], arc, bc)
        lb = F.log_softmax(base_b - V, -1)
        lp += la.gather(-1, ba[:, None]).squeeze(-1) + lb.gather(-1, bb[:, None]).squeeze(-1)
    return lp


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
