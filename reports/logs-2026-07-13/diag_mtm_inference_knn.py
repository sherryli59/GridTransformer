"""Does a bigger TILT cage AT INFERENCE (no retrain) improve MTM acceptance for the knn=8-trained model?
The inference-time control for the knn=24 retrain. Set m.knn_pot in {8,16,24,32} -> BOTH sample_block_b
and block_log_prob_b use it (self-consistent -> exact tempered density -> valid I-MTM). Measure exact
block-MTM acceptance (fraction accepted, current state = DATA block) at K=4/8, R=2.0, pos_temp=0.4, N=16
trials, over held cavities. Bigger cage helps zero-shot => the model can use extra context without
retraining; helps only after retrain => it must LEARN to use it (the retrain's real value)."""
import torch, statistics as st
from liquid_coupling_flow.ka3d_ebm_batched import KA3DScaffoldEBMBatched
from liquid_coupling_flow.ka3d_scaffold_ar import label_to_scaffold, fixed_ball_scaffold
from liquid_coupling_flow.ka3d_cavity_ar import _mic
from liquid_coupling_flow.ka3d_cavity_carve import carve
from liquid_coupling_flow.ka_energy import ka_energy

dev = "cuda"; RCTX = 2.5; BIGL = 100.0; BETA = 2.0; N = 16; NCAV = 12; PT = 1.0
m = KA3DScaffoldEBMBatched(cat_bins=128, cat_range=2.5).to(dev)
m.load_state_dict(torch.load("liquid_coupling_flow/artifacts/ka3d_cavity_ebm3ax_rho115_best.pt", map_location=dev, weights_only=False)["state_dict"], strict=False)
m.eval(); m.use_frame = False
D = torch.load("liquid_coupling_flow/artifacts/ka3d_dataset_N4096_T0.5_rho1.15.pt", map_location=dev, weights_only=False)
X, S, L = D["x"].to(dev).float(), D["s"].to(dev).long(), float(D["L"])


def energy_b(Xb, Sb, bnd, sb):
    Mn, mm = Xb.shape[0], bnd.shape[0]
    x = torch.cat([Xb, bnd[None].expand(Mn, mm, 3)], 1); s = torch.cat([Sb, sb[None].expand(Mn, mm)], 1)
    return ka_energy(x.double(), s.long(), BIGL).float()


@torch.no_grad()
def accept(R, K, gen):
    accs, dEs = [], []
    ncav = 0
    for ci in range(16):
        c = torch.rand(3, generator=gen, device=dev) * L; p = carve(X[ci], S[ci], c, R, L)
        if p["n_in"] < K + 6:
            continue
        xo, so, _ = label_to_scaffold(_mic(p["x_in"], c, L), p["s_in"], R)
        xout = _mic(p["x_out"], c, L); bm = xout.norm(dim=-1) < (R + RCTX); bnd, sb = xout[bm], p["s_out"][bm]
        n = xo.shape[0]; a = fixed_ball_scaffold(n, R, dev); seed = int(torch.randint(n, (), generator=gen, device=dev))
        blk = torch.zeros(n, dtype=torch.bool, device=dev)
        blk[(a - a[seed]).norm(dim=-1).topk(K, largest=False).indices] = True
        Xr = xo[None].expand(N, n, 3).contiguous(); Sr = so[None].expand(N, n).contiguous()
        Xp, Sp, lqf = m.sample_block_b(Xr, Sr, blk, bnd, sb, R, gen=gen, pos_temp=PT)
        up = -BETA * energy_b(Xp, Sp, bnd, sb) - lqf                                # trial weights
        Xd = xo[None].contiguous(); Sd = so[None].contiguous()
        u0 = (-BETA * energy_b(Xd, Sd, bnd, sb) - m.block_log_prob_b(Xd, Sd, blk, bnd, sb, R, pos_temp=PT))[0]
        sf = torch.logsumexp(up, 0)
        J = int(torch.multinomial(torch.softmax(up, 0), 1, generator=gen))
        lr = up.clone(); lr[J] = u0; sr = torch.logsumexp(lr, 0)
        accs.append(float(torch.rand((), device=dev, generator=gen).log() < (sf - sr)))
        Ed = energy_b(Xd, Sd, bnd, sb)[0]
        dEs.append(float((energy_b(Xp, Sp, bnd, sb).median() - Ed) / K))
        ncav += 1
        if ncav >= NCAV:
            break
    return 100 * st.mean(accs), st.median(dEs)


print(f"=== exact tempered block-MTM acceptance vs INFERENCE tilt-cage (knn=8 model, pos_temp={PT}, N={N}) ===", flush=True)
for kp in (8, 16, 24, 32):
    m.knn_pot = kp
    row = f"  knn_pot={kp:2d}:"
    for R in (2.0, 2.5):
        for K in (4, 8):
            gen = torch.Generator(device=dev).manual_seed(0)
            acc, de = accept(R, K, gen)
            row += f"  R{R}K{K}: {acc:4.1f}% (dE/p{de:+6.1f})"
    print(row, flush=True)
