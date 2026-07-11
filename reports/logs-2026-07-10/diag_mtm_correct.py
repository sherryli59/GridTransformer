"""CORRECT independence-sampler MTM test for COLLECTIVE (K>1) block moves.

Block proposal T(x,y) = q(y_block | retained) does NOT depend on the current block values x_block
(only on the shared retained context) -> it's an independence sampler. For independence samplers,
MTM's acceptance reduces to a REUSABLE-batch form (Liu-Liang-Wong): draw N trial blocks y_1..y_N,
select Y ~ softmax(log_u), where log_u(z) = -beta*U(z) - logq(z|retained). Then

    accept = min(1, S_fwd / S_rev),  S_fwd = sum_j u(y_j),  S_rev = S_fwd - u(Y) + u(x)

This can accept even when EVERY individual trial fails vanilla single-try MH, because it compares
SUMS of importance weights, not any one trial's pass/fail (unlike the flawed "any single trial
accepted" proxy). Tests K=4 and K=8 (collective moves) only -- no K=1.
"""
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
ckp = SCR / "diag_mtm2_ckpt.pt"; shutil.copy("liquid_coupling_flow/artifacts/ka3d_blob_ar.pt", ckp)
ck = torch.load(ckp, map_location=dev, weights_only=False)
m = KA3DScaffoldAR().to(dev); m.load_state_dict(ck["state_dict"]); m.eval()
print(f"ckpt step={ck['step']}", flush=True)

d = torch.load("liquid_coupling_flow/artifacts/ka3d_dataset_N512_T0.5.pt", map_location=dev, weights_only=False)
X, S, L = d["x"].to(dev).float(), d["s"].to(dev).long(), float(d["L"])
R, beta = 2.0, 2.0
gen = torch.Generator(device=dev).manual_seed(9)
E3, E1 = torch.zeros(0, 3, device=dev), torch.zeros(0, dtype=torch.long, device=dev)


def energy_total(xo, so):
    return ka_energy(xo[None], so.long()[None], 100.0)[0]


def mtm_trial(K, N, n_sites):
    """One MTM decision per site: N candidate blocks, correct independence-sampler acceptance."""
    accepts, ratios = [], []
    with torch.no_grad():
        for ci in range(900, 900 + n_sites):
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
            u_old = float(energy_total(x, s))
            lq_x = float(block_log_prob(m, x, s, blk, E3, E1, R))         # logq(current block | retained)
            log_u_x = -beta * u_old - lq_x
            log_u_trials = []
            for _ in range(N):
                xn, sn, lq_y = sample_block(m, x, s, blk, E3, E1, R, gen=gen)
                u_new = float(energy_total(xn, sn))
                log_u_trials.append(-beta * u_new - float(lq_y))
            lu = torch.tensor(log_u_trials, device=dev)
            probs = torch.softmax(lu, 0)
            jstar = torch.multinomial(probs, 1, generator=gen).item()
            s_fwd = torch.logsumexp(lu, 0)
            lu_rev = lu.clone(); lu_rev[jstar] = log_u_x
            s_rev = torch.logsumexp(lu_rev, 0)
            log_ratio = float(s_fwd - s_rev)
            ratios.append(log_ratio)
            accept = torch.rand((), device=dev, generator=gen).log().item() < min(0.0, log_ratio)
            accepts.append(accept)
    return accepts, ratios


for K in (4, 8):
    for N in (8, 32):
        accepts, ratios = mtm_trial(K, N, n_sites=25)
        n_ev = len(accepts)
        print(f"K={K} N={N}: sites={n_ev}  empirical MTM accept={100*sum(accepts)/n_ev:.0f}%  "
              f"median log_ratio={st.median(ratios):+.2f}  best={max(ratios):+.2f}", flush=True)
print("\nThis is the CORRECT MTM test (sum-of-weights, not any-single-trial). "
      "If still ~0% even at N=32, the conditional needs more training before collective MTM works; "
      "if nonzero, MTM is already viable at this K.", flush=True)
