"""EARLY READ: block-MTM acceptance of the in-training rho=1.15 cavity model (best ckpt), on HELD-chain
cavities only (the 16 reference chains — never trained on). K x R scan, Liu-Liang-Wong independence-MTM
with N=16 trials, M=8 cavities-in-parallel per (K,R) cell. Also reports the raw proposal-block energy above
the data block (the clash scale) — context for whether accepts are energy-limited or logq-limited."""
import torch, statistics as st
from liquid_coupling_flow.ka3d_ebm_batched import KA3DScaffoldEBMBatched
from liquid_coupling_flow.ka3d_scaffold_ar import label_to_scaffold, fixed_ball_scaffold
from liquid_coupling_flow.ka3d_cavity_ar import _mic
from liquid_coupling_flow.ka3d_cavity_carve import carve
from liquid_coupling_flow.ka_energy import ka_energy

dev = "cuda"; RCTX = 2.5; BIGL = 100.0; BETA = 2.0; NTRIAL = 16; NCAV = 8

ck = torch.load("liquid_coupling_flow/artifacts/ka3d_cavity_ebm3ax_rho115_best.pt", map_location=dev, weights_only=False)
m = KA3DScaffoldEBMBatched(cat_bins=128, cat_range=2.5).to(dev)
m.load_state_dict(ck["state_dict"], strict=False); m.eval(); m.use_frame = False
print(f"ckpt step {ck.get('step')} held_nll {ck.get('held_nll'):+.4f}", flush=True)

d = torch.load("liquid_coupling_flow/artifacts/ka3d_dataset_N4096_T0.5_rho1.15.pt", map_location=dev, weights_only=False)
X, S, L = d["x"].to(dev).float(), d["s"].to(dev).long(), float(d["L"])
gen = torch.Generator(device=dev).manual_seed(0)


def energy_b(Xb, Sb, bnd, sb):
    M, mm = Xb.shape[0], bnd.shape[0]
    x = torch.cat([Xb, bnd[None].expand(M, mm, 3)], 1); s = torch.cat([Sb, sb[None].expand(M, mm)], 1)
    return ka_energy(x, s.long(), BIGL)


@torch.no_grad()
def mtm_accept(Rr, K):
    accs, dEs = [], []
    ncav = 0
    for ci in range(16):                                                   # snapshot-1 configs of held chains
        c = torch.rand(3, generator=gen, device=dev) * L; p = carve(X[ci], S[ci], c, Rr, L)
        if p["n_in"] < K + 6:
            continue
        xo, so, _ = label_to_scaffold(_mic(p["x_in"], c, L), p["s_in"], Rr)
        xout = _mic(p["x_out"], c, L); bm = xout.norm(dim=-1) < (Rr + RCTX); bnd, sb = xout[bm], p["s_out"][bm]
        n = xo.shape[0]; a = fixed_ball_scaffold(n, Rr, dev); seed = int(torch.randint(n, (), generator=gen, device=dev))
        blk = torch.zeros(n, dtype=torch.bool, device=dev)
        blk[(a - a[seed]).norm(dim=-1).topk(K, largest=False).indices] = True
        Xb = xo[None].expand(1, n, 3).contiguous(); Sb = so[None].expand(1, n).contiguous()
        Xr, Sr = Xb.expand(NTRIAL, n, 3).contiguous(), Sb.expand(NTRIAL, n).contiguous()
        Xp, Sp, lqf = m.sample_block_b(Xr, Sr, blk, bnd, sb, Rr, gen=gen)
        up = (-BETA * energy_b(Xp, Sp, bnd, sb) - lqf)                      # u(y_j) [NTRIAL]
        u0 = (-BETA * energy_b(Xb, Sb, bnd, sb) - m.block_log_prob_b(Xb, Sb, blk, bnd, sb, Rr))[0]
        sf = torch.logsumexp(up, 0)
        J = int(torch.multinomial(torch.softmax(up, 0), 1, generator=gen))
        lr = up.clone(); lr[J] = u0; sr = torch.logsumexp(lr, 0)
        accs.append(float(torch.rand((), device=dev, generator=gen).log() < (sf - sr)))
        E_data = energy_b(Xb, Sb, bnd, sb)[0]
        dEs.append(float((energy_b(Xp, Sp, bnd, sb).median() - E_data) / K))
        ncav += 1
        if ncav >= NCAV:
            break
    return 100 * st.mean(accs), st.median(dEs), ncav


print(f"\nblock-MTM acceptance (N={NTRIAL} trials, {NCAV} held cavities/cell, beta={BETA}):", flush=True)
print(f"{'':>8}" + "".join(f"  K={k:<12}" for k in (4, 8, 12)), flush=True)
for Rr in (2.0, 2.5, 3.0):
    row = f"R={Rr:<5}"
    for K in (4, 8, 12):
        acc, de, nc = mtm_accept(Rr, K)
        row += f"  {acc:3.0f}% (dE/p {de:+7.1f})"
    print(row, flush=True)
