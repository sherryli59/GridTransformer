"""Push the EBM block-MTM: K x N_trials acceptance sweep + energy diagnostic to confirm the mechanism
(does the learned potential lower proposal ENERGY / match logq to -bU, not just clash count?). Compares
against the factorized model on the SAME configs. argv: ebm_ckpt factorized_ckpt."""
import statistics as st, sys
import torch
import torch.nn.functional as F
from liquid_coupling_flow.ka_localframe_ebm import KALocalFrameEBM
from liquid_coupling_flow.ka_localframe import KALocalFrameModel
from liquid_coupling_flow.ka_gridformer import _wrap_pm
from liquid_coupling_flow.ka_energy import ka_energy

dev = "cuda"; beta = 2.0; N = 100
ebm_path = sys.argv[1] if len(sys.argv) > 1 else "liquid_coupling_flow/artifacts/ka_localframe_ebm_N100.pt"
fac_path = sys.argv[2] if len(sys.argv) > 2 else "liquid_coupling_flow/artifacts/ka_localframe_N100_20k.pt"
cke = torch.load(ebm_path, map_location=dev, weights_only=False)
me = KALocalFrameEBM(rho=cke["rho"], n_bins=cke["n_bins"], knn=cke["knn"]).to(dev)
me.load_state_dict(cke["state_dict"], strict=False); me.eval()
ckf = torch.load(fac_path, map_location=dev, weights_only=False)
mf = KALocalFrameModel(rho=ckf["rho"], n_bins=ckf["n_bins"], knn=ckf["knn"]).to(dev)
mf.load_state_dict(ckf["state_dict"], strict=False); mf.eval()
ref = torch.load(f"liquid_coupling_flow/artifacts/ka_reference_N{N}.pt", map_location=dev, weights_only=False)
data = ref["x"].to(dev); s0 = ref["s"].to(dev).long(); L = ref["L"]
sc = me.geo._scaffold(N, dev); arc = me._arc_scale(N)
bc = me._bin_center(torch.arange(me.n_bins, device=dev))


def energy(pos, sp):
    return float(ka_energy(pos[None], sp.long()[None], L)[0])


@torch.no_grad()
def sample_ebm(pos0, sp0, k, B):
    pos = pos0[None].expand(B, N, 2).clone(); sp = sp0[None].expand(B, N).clone(); logq = torch.zeros(B, device=dev)
    for j in range(N - k, N):
        h, origin, nr, nsp, val = me._step_ebm(pos, sp, sc[j], j, L)
        la = F.log_softmax(me.head_a(h), -1); ba = torch.multinomial(la.exp(), 1).squeeze(-1)
        V = me._V_b(ba, nr, nsp, val, sp[:, j], arc, bc)
        lb = F.log_softmax(me.head_b(h + me.bin_a_emb(ba)) - V, -1); bb = torch.multinomial(lb.exp(), 1).squeeze(-1)
        logq += la.gather(-1, ba[:, None]).squeeze(-1) + lb.gather(-1, bb[:, None]).squeeze(-1)
        a = me._bin_center(ba) + (torch.rand(B, device=dev) - 0.5) * me.bin_w
        b = me._bin_center(bb) + (torch.rand(B, device=dev) - 0.5) * me.bin_w
        pos[:, j] = torch.remainder(origin + torch.stack([a, b], -1) * arc, L)
    return pos, logq


@torch.no_grad()
def logp_ebm(pos_in, sp0, k):
    B = pos_in.shape[0]; pos = pos_in.clone(); sp = sp0[None].expand(B, N).clone(); lp = torch.zeros(B, device=dev)
    for j in range(N - k, N):
        h, origin, nr, nsp, val = me._step_ebm(pos, sp, sc[j], j, L)
        ab = _wrap_pm(pos[:, j] - origin, L) / arc; ba = me._bin(ab[..., 0]); bb = me._bin(ab[..., 1])
        la = F.log_softmax(me.head_a(h), -1); V = me._V_b(ba, nr, nsp, val, sp[:, j], arc, bc)
        lb = F.log_softmax(me.head_b(h + me.bin_a_emb(ba)) - V, -1)
        lp += la.gather(-1, ba[:, None]).squeeze(-1) + lb.gather(-1, bb[:, None]).squeeze(-1)
    return lp


@torch.no_grad()
def sample_fac(pos0, sp0, k, B):
    pos = pos0[None].expand(B, N, 2).clone(); sp = sp0[None].expand(B, N).clone(); logq = torch.zeros(B, device=dev)
    for j in range(N - k, N):
        h, origin = mf._step(pos, sp, sc[j], j, L)
        la = F.log_softmax(mf.head_a(h), -1); ba = torch.multinomial(la.exp(), 1).squeeze(-1)
        lb = F.log_softmax(mf.head_b(h + mf.bin_a_emb(ba)), -1); bb = torch.multinomial(lb.exp(), 1).squeeze(-1)
        logq += la.gather(-1, ba[:, None]).squeeze(-1) + lb.gather(-1, bb[:, None]).squeeze(-1)
        a = mf._bin_center(ba) + (torch.rand(B, device=dev) - 0.5) * mf.bin_w
        b = mf._bin_center(bb) + (torch.rand(B, device=dev) - 0.5) * mf.bin_w
        pos[:, j] = torch.remainder(origin + torch.stack([a, b], -1) * arc, L)
    return pos, logq


@torch.no_grad()
def logp_fac(pos_in, sp0, k):
    B = pos_in.shape[0]; pos = pos_in.clone(); sp = sp0[None].expand(B, N).clone(); lp = torch.zeros(B, device=dev)
    for j in range(N - k, N):
        h, origin = mf._step(pos, sp, sc[j], j, L)
        ab = _wrap_pm(pos[:, j] - origin, L) / arc; ba = mf._bin(ab[..., 0]); bb = mf._bin(ab[..., 1])
        la = F.log_softmax(mf.head_a(h), -1); lb = F.log_softmax(mf.head_b(h + mf.bin_a_emb(ba)), -1)
        lp += la.gather(-1, ba[:, None]).squeeze(-1) + lb.gather(-1, bb[:, None]).squeeze(-1)
    return lp


def mtm_accept(sample_fn, logp_fn, pos0, sp0, k, Nt):
    u_old = energy(pos0, sp0); lq_x = float(logp_fn(pos0[None], sp0, k))
    pos_t, lq_t = sample_fn(pos0, sp0, k, Nt)
    ut = torch.tensor([energy(pos_t[t], sp0) for t in range(Nt)], device=dev)
    lus = torch.cat([torch.tensor([-beta * u_old - lq_x], device=dev), -beta * ut - lq_t])
    sfwd = torch.logsumexp(lus[1:], 0); js = int(torch.multinomial(torch.softmax(lus[1:], 0), 1))
    lu_rev = lus[1:].clone(); lu_rev[js] = lus[0]
    acc = float(torch.rand((), device=dev).log() < (sfwd - torch.logsumexp(lu_rev, 0)))
    return acc, float(ut.mean()), u_old                     # mean proposal energy, true energy


print(f"true block context: N={N} beta={beta}\n", flush=True)
for k in (4, 6, 8):
    print(f"--- k={k} ---", flush=True)
    for tag, sfn, lfn in (("factorized", sample_fac, logp_fac), ("EBM-potential", sample_ebm, logp_ebm)):
        de = []
        for Nt in (16, 32, 64):
            accs, ep = [], []
            for ci in range(20):
                order = me.geo._curve_order(data[ci:ci + 1], N)[0]; pos0 = data[ci][order]; sp0 = s0[order]
                a, me_, ue = mtm_accept(sfn, lfn, pos0, sp0, k, Nt)
                accs.append(a); ep.append(me_ - ue)
            de.append(f"N{Nt}:{100*sum(accs)/len(accs):.0f}%")
            if Nt == 16:
                dmean = st.mean(ep)
        print(f"  {tag:14s} " + "  ".join(de) + f"   [<U_prop - U_true>={dmean:+.2f}]", flush=True)
