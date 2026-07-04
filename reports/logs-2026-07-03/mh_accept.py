"""Estimate the stationary MH acceptance rate of the k-NN EGNN cluster move.
Reference configs are equilibrium (parallel-tempering) samples ~ pi, so averaging the acceptance prob over them
IS the stationary acceptance rate. Valid state-dependent-proposal MH (cluster slots are move-invariant; the cage
is selected from the current cluster centroid, so the reverse density conditions on cage(x')):
  log alpha = -beta*(U' - U) + log q(x_C | cage(x')) - log q(x'_C | cage(x))
"""
import sys, torch, numpy as np
from liquid_coupling_flow import ka_cluster as KC, ka_cluster_egnn as E
from liquid_coupling_flow.ka_cluster_flow import slot_order, _scaffold
from liquid_coupling_flow.ka_energy import ka_energy

DEV = "cuda"; T = 0.5; BETA = 1.0 / T
integ = sys.argv[1] if len(sys.argv) > 1 else "midpoint"
nst = int(sys.argv[2]) if len(sys.argv) > 2 else 32
B = int(sys.argv[3]) if len(sys.argv) > 3 else 64
torch.manual_seed(0)

sc, L, geo = _scaffold(100, DEV)
ref = torch.load(f"{E.ART}/ka_reference_N100.pt", map_location=DEV, weights_only=False)
s0 = ref["s"].to(DEV).long()
ck_name = sys.argv[4] if len(sys.argv) > 4 else "ka_cluster_egnn_N100.pt"
ck = torch.load(f"{E.ART}/{ck_name}", map_location=DEV, weights_only=False)
# fp32: GeForce FP64 is 1:64; acceptance is energy-dominated so fp32 log_q noise (~1e-3) is irrelevant
P = E.load_flow(ck, DEV); P.n_steps = nst; P.integrator = integ; P.n_picard = 60; P.picard_tol = 1e-12
k = ck["k"]
print(f"model prior-off step {ck['step']} loss {ck['loss_last']:.3f} | integ={integ} n_steps={nst} B={B} beta={BETA}", flush=True)

# a random slice of equilibrium configs, slot-ordered, in double
idx = torch.randperm(ref["x"].shape[0], device=DEV)[:B]
pos, sso = slot_order(ref["x"][idx].to(DEV), s0, geo, 100)

# self-consistency check on this model (sample then re-score x'): confirms log_q is trustworthy for MH
cl0 = KC.cluster_slots(5, sc, k, L)
xchk, lq_s = P.sample(pos, sso, cl0, sc, L)
lq_r = P.log_q(pos, sso, cl0, xchk, sc, L)
print(f"self-consistency  max|logq_sample - logq_score(x')| = {(lq_s - lq_r).abs().max().item():.2e}  "
      f"(cage fixed by pos) -> log_q trustworthy if small", flush=True)

acc_all, dU_all, dq_all, clash_all = [], [], [], []
seeds = list(range(0, 100, 10))
for seed in seeds:
    cl = KC.cluster_slots(seed, sc, k, L)
    xC_cur = pos[:, cl].clone()
    U_cur = ka_energy(pos, sso, L)                                   # [B]
    xC_new, logq_fwd = P.sample(pos, sso, cl, sc, L)                 # log q(x' | cage(x))
    pos_new = pos.clone(); pos_new[:, cl] = xC_new
    U_new = ka_energy(pos_new, sso, L)
    logq_rev = P.log_q(pos_new, sso, cl, xC_cur, sc, L)             # log q(x | cage(x'))  (reverse density)
    dU = U_new - U_cur
    log_alpha = -BETA * dU + logq_rev - logq_fwd
    acc = torch.clamp(log_alpha, max=0.0).exp()                     # min(1, exp(.))
    # clash diagnostic on the proposed cluster (min neighbour distance < 0.7)
    d = xC_new[:, :, None, :] - pos_new[:, None, :, :]; d = d - L * torch.round(d / L)
    r = (d ** 2).sum(-1).sqrt(); r[:, :, cl] += torch.eye(k, device=DEV)[None] * 1e3
    clash = (r.min(-1).values.min(-1).values < 0.7).double()        # [B] any-clash per proposal
    acc_all.append(acc); dU_all.append(dU); dq_all.append(logq_rev - logq_fwd); clash_all.append(clash)

acc = torch.cat(acc_all); dU = torch.cat(dU_all); dq = torch.cat(dq_all); clash = torch.cat(clash_all)
print(f"\n=== MH cluster-move acceptance (prior-off proposal, {len(seeds)} seeds x {B} eq configs = {acc.numel()} moves) ===")
print(f"  mean acceptance      : {acc.mean().item()*100:.2f}%   (median {acc.median().item()*100:.2f}%)")
print(f"  acceptance | no-clash : {acc[clash==0].mean().item()*100:.2f}%   (fraction no-clash {(clash==0).double().mean().item()*100:.1f}%)")
print(f"  acceptance | clash    : {acc[clash==1].mean().item()*100:.3f}%   (fraction clash {(clash==1).double().mean().item()*100:.1f}%)")
print(f"  mean dU (beta*dU)     : {dU.mean().item():.3f}  ({(BETA*dU).mean().item():.3f})")
print(f"  mean logq_rev-logq_fwd: {dq.mean().item():.3f}")
print(f"  P(dU<0)               : {(dU<0).double().mean().item()*100:.1f}%   P(acc>0.5): {(acc>0.5).double().mean().item()*100:.1f}%")
