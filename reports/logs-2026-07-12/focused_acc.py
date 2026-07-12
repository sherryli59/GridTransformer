"""HIGH-STATISTICS block-MTM acceptance of the final rho=1.15 cavity model (best ckpt), held chains only.

Fixes the noisy early read: per cavity run M_REP independent MTM instances in ONE batched call
(M_REP*NTRIAL trial proposals -> reshape [M_REP, NTRIAL]) => M_REP accept samples/cavity; accumulate over
many cavities => acc% +- SE. Also an N-TRIAL LADDER (independence-MTM acceptance rises with #trials) to show
the multiple-try lever. pos_temp=1.0 => EXACT (the trustworthy number)."""
import math, torch, statistics as st
from liquid_coupling_flow.ka3d_ebm_batched import KA3DScaffoldEBMBatched
from liquid_coupling_flow.ka3d_scaffold_ar import label_to_scaffold, fixed_ball_scaffold
from liquid_coupling_flow.ka3d_cavity_ar import _mic
from liquid_coupling_flow.ka3d_cavity_carve import carve
from liquid_coupling_flow.ka_energy import ka_energy

dev = "cuda"; RCTX = 2.5; BIGL = 100.0; BETA = 2.0
M_REP = 24                                                            # parallel MTM instances per cavity (M>1)
N_CAV = 12                                                            # held cavities per cell
ck = torch.load("liquid_coupling_flow/artifacts/ka3d_cavity_ebm3ax_rho115_best.pt", map_location=dev, weights_only=False)
m = KA3DScaffoldEBMBatched(cat_bins=128, cat_range=2.5).to(dev)
m.load_state_dict(ck["state_dict"], strict=False); m.eval(); m.use_frame = False
print(f"ckpt step {ck.get('step')} held_nll {ck.get('held_nll'):+.4f}  M_REP={M_REP} N_CAV={N_CAV} beta={BETA}", flush=True)
d = torch.load("liquid_coupling_flow/artifacts/ka3d_dataset_N4096_T0.5_rho1.15.pt", map_location=dev, weights_only=False)
X, S, L = d["x"].to(dev).float(), d["s"].to(dev).long(), float(d["L"])
gen = torch.Generator(device=dev).manual_seed(0)


def energy_b(Xb, Sb, bnd, sb):
    Mn, mm = Xb.shape[0], bnd.shape[0]
    x = torch.cat([Xb, bnd[None].expand(Mn, mm, 3)], 1); s = torch.cat([Sb, sb[None].expand(Mn, mm)], 1)
    return ka_energy(x, s.long(), BIGL)


@torch.no_grad()
def acc_cell(Rr, K, ntrial, pos_temp=1.0):
    """Return (acc%, SE%, median dE/particle, n_accept_samples)."""
    samples = []
    des = []
    ncav = 0
    for ci in range(16):
        c = torch.rand(3, generator=gen, device=dev) * L; p = carve(X[ci], S[ci], c, Rr, L)
        if p["n_in"] < K + 6:
            continue
        xo, so, _ = label_to_scaffold(_mic(p["x_in"], c, L), p["s_in"], Rr)
        xout = _mic(p["x_out"], c, L); bm = xout.norm(dim=-1) < (Rr + RCTX); bnd, sb = xout[bm], p["s_out"][bm]
        n = xo.shape[0]; a = fixed_ball_scaffold(n, Rr, dev); seed = int(torch.randint(n, (), generator=gen, device=dev))
        blk = torch.zeros(n, dtype=torch.bool, device=dev)
        blk[(a - a[seed]).norm(dim=-1).topk(K, largest=False).indices] = True
        Xb = xo[None]; Sb = so[None]
        u0 = (-BETA * energy_b(Xb, Sb, bnd, sb) - m.block_log_prob_b(Xb, Sb, blk, bnd, sb, Rr))[0]   # scalar
        B = M_REP * ntrial
        Xr = xo[None].expand(B, n, 3).contiguous(); Sr = so[None].expand(B, n).contiguous()
        Xp, Sp, lqf = m.sample_block_b(Xr, Sr, blk, bnd, sb, Rr, gen=gen, pos_temp=pos_temp)
        up = (-BETA * energy_b(Xp, Sp, bnd, sb) - lqf).reshape(M_REP, ntrial)       # [M_REP, ntrial]
        des.append(float(((-up * 0 + energy_b(Xp, Sp, bnd, sb).reshape(M_REP, ntrial)).median() -
                          energy_b(Xb, Sb, bnd, sb)[0]) / K))
        sf = torch.logsumexp(up, 1)                                                 # [M_REP]
        probs = torch.softmax(up, 1)
        J = torch.multinomial(probs, 1, generator=gen).squeeze(1)                   # [M_REP]
        lr = up.clone(); lr[torch.arange(M_REP, device=dev), J] = u0
        sr = torch.logsumexp(lr, 1)
        acc = (torch.rand(M_REP, device=dev, generator=gen).log() < (sf - sr)).float()
        samples.extend(acc.tolist())
        ncav += 1
        if ncav >= N_CAV:
            break
    p_hat = st.mean(samples); nsamp = len(samples)
    se = 100 * math.sqrt(max(p_hat * (1 - p_hat), 1e-9) / nsamp)
    return 100 * p_hat, se, st.median(des), nsamp


print("\n=== EXACT (pos_temp=1.0) block-MTM acceptance, N=16 trials ===", flush=True)
print(f"{'':>7}" + "".join(f"  K={k:<16}" for k in (4, 6, 8)), flush=True)
for Rr in (2.0, 2.5, 3.0):
    row = f"R={Rr:<4}"
    for K in (4, 6, 8):
        acc, se, de, ns = acc_cell(Rr, K, 16)
        row += f"  {acc:4.1f}+-{se:3.1f}% (dE{de:+6.1f})"
    print(row + f"  [n={ns}]", flush=True)

print("\n=== N-TRIAL LADDER (K=4, R=2.0, exact): acceptance vs #trials ===", flush=True)
for ntr in (4, 8, 16, 32, 64):
    acc, se, de, ns = acc_cell(2.0, 4, ntr)
    print(f"  N={ntr:3d} trials:  acc {acc:4.1f} +- {se:3.1f}%   (n={ns})", flush=True)

print("\n=== SHARPENED PROPOSAL ENERGY (pos_temp) at K=4 -- dE only (temp<1 acc not exact) ===", flush=True)
for Rr in (2.0, 2.5, 3.0):
    row = f"R={Rr:<4}"
    for pt in (1.0, 0.5, 0.35):
        _, _, de, _ = acc_cell(Rr, 4, 16, pos_temp=pt)
        row += f"  T={pt}:dE{de:+7.1f}"
    print(row, flush=True)
