"""Sanity: run the SAME block-MTM idea on the 2D ka_localframe model (which we KNOW generates well,
TF-clash 8.6%). Regenerate the last-k curve particles (a suffix blob = exact in-distribution tail
conditional) given the first N-k, fixing species. Measure block-clash + CORRECT independence-MTM
acceptance, exactly as in 3D. If 2D is clean here, the approach + metrics are validated and 3D's
broad conditional is the real gap; if 2D also fails, the test itself is wrong."""
import statistics as st
import torch
import torch.nn.functional as F
from liquid_coupling_flow.ka_localframe import KALocalFrameModel
from liquid_coupling_flow.ka_gridformer import _wrap_pm
from liquid_coupling_flow.ka_energy import ka_energy

dev = "cuda"
ck = torch.load("liquid_coupling_flow/artifacts/ka_localframe_N100_20k.pt", map_location=dev, weights_only=False)
m = KALocalFrameModel(rho=ck["rho"], n_bins=ck["n_bins"], knn=ck["knn"]).to(dev)
m.load_state_dict(ck["state_dict"], strict=False); m.eval()
N = 100
ref = torch.load(f"liquid_coupling_flow/artifacts/ka_reference_N{N}.pt", map_location=dev, weights_only=False)
data = ref["x"].to(dev); s0 = ref["s"].to(dev).long(); L = ref["L"]
beta = 2.0

sc = m.geo._scaffold(N, dev); arc = m._arc_scale(N)


@torch.no_grad()
def block_sample(pos0, sp0, k, B):
    """Regenerate the last-k curve particles for B trials (positions only, true species). Returns
    pos [B,N,2], logq [B]."""
    pos = pos0[None].expand(B, N, 2).clone(); sp = sp0[None].expand(B, N).clone()
    logq = torch.zeros(B, device=dev)
    for j in range(N - k, N):
        h, origin = m._step(pos, sp, sc[j], j, L)                 # context from prefix 0..j-1
        la = F.log_softmax(m.head_a(h), -1); ba = torch.multinomial(la.exp(), 1).squeeze(-1)
        lb = F.log_softmax(m.head_b(h + m.bin_a_emb(ba)), -1); bb = torch.multinomial(lb.exp(), 1).squeeze(-1)
        logq += la.gather(-1, ba[:, None]).squeeze(-1) + lb.gather(-1, bb[:, None]).squeeze(-1)
        a = m._bin_center(ba) + (torch.rand(B, device=dev) - 0.5) * m.bin_w
        b = m._bin_center(bb) + (torch.rand(B, device=dev) - 0.5) * m.bin_w
        pos[:, j] = torch.remainder(origin + torch.stack([a, b], -1) * arc, L)
    return pos, logq


@torch.no_grad()
def block_logp_true(pos0, sp0, k):
    """logq of the TRUE last-k block given the true prefix (reverse move)."""
    lp = torch.zeros(1, device=dev); pos = pos0[None]; sp = sp0[None]
    for j in range(N - k, N):
        h, origin = m._step(pos, sp, sc[j], j, L)
        ab = _wrap_pm(pos[:, j] - origin, L) / arc
        ba = m._bin(ab[..., 0]); bb = m._bin(ab[..., 1])
        la = F.log_softmax(m.head_a(h), -1); lb = F.log_softmax(m.head_b(h + m.bin_a_emb(ba)), -1)
        lp += la.gather(-1, ba[:, None]).squeeze(-1) + lb.gather(-1, bb[:, None]).squeeze(-1)
    return float(lp)


def clash_in_block(pos, sp, k):
    """# block particles (< 0.8) of any other, with PBC (2D)."""
    block = pos[N - k:]
    d = pos[None] - block[:, None]; d = d - L * torch.round(d / L)
    dd = d.norm(dim=-1); dd[torch.arange(k), torch.arange(N - k, N)] = 1e9
    return int((dd.min(1).values < 0.8).sum())


def energy(pos, sp):
    return float(ka_energy(pos[None], sp.long()[None], L)[0])


for k in (4, 8):
    clashfree, nclash, mtm = [], [], []
    for ci in range(20):
        order = m.geo._curve_order(data[ci:ci + 1], N)[0]
        pos0 = data[ci][order]; sp0 = (s0 if s0.dim() == 1 else s0[ci])[order] if s0.dim() > 1 else s0[order]
        # correct independence-MTM: 16 trial blocks + current
        u_old = energy(pos0, sp0); lq_x = block_logp_true(pos0, sp0, k)
        pos_t, lq_t = block_sample(pos0, sp0, k, 16)
        lus = [-beta * u_old - lq_x]
        for t in range(16):
            lus.append(-beta * energy(pos_t[t], sp0) - float(lq_t[t]))
            if t == 0:
                nc = clash_in_block(pos_t[t], sp0, k); nclash.append(nc); clashfree.append(nc == 0)
        lu = torch.tensor(lus[1:], device=dev); sfwd = torch.logsumexp(lu, 0)
        js = int(torch.multinomial(torch.softmax(lu, 0), 1)); lu_rev = lu.clone(); lu_rev[js] = lus[0]
        mtm.append(float(torch.rand((), device=dev).log() < (sfwd - torch.logsumexp(lu_rev, 0))))
    n = len(mtm)
    print(f"2D k={k}: clash-free blocks={100*sum(clashfree)/n:.0f}%  mean #clash={st.mean(nclash):.2f}  "
          f"MTM accept={100*sum(mtm)/n:.0f}%   (3D was: clash-free ~3%, MTM 0%)", flush=True)
