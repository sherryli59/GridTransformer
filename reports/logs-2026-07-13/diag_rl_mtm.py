"""Does RL+demand's declash translate to exact MTM ACCEPTANCE? MTM uses the TRUE (uncapped) block energy,
so residual hard overlaps (terminal spike) can keep it ~0 even with lower CAPPED energy. Exact tempered
block-MTM (current=data block), 5-seed, K=4/6/8, R=2.5, base (block-cond) vs RL+demand (step500). Also
reports the true (uncapped) median block dE/particle -- the thing acceptance actually sees."""
import torch, statistics as st
from liquid_coupling_flow.ka3d_ebm_batched import KA3DScaffoldEBMBatched
from liquid_coupling_flow.ka3d_scaffold_ar import label_to_scaffold, fixed_ball_scaffold
from liquid_coupling_flow.ka3d_cavity_ar import _mic
from liquid_coupling_flow.ka3d_cavity_carve import carve
from liquid_coupling_flow.ka_energy import ka_energy

dev = "cuda"; RCTX = 2.5; R = 2.5; BIGL = 100.0; BETA = 2.0; M = 16; NCAV = 12
BASE = "liquid_coupling_flow/artifacts/ka3d_cavity_ebm3ax_rho115_blockcond_knn24_best.pt"
RLD = "liquid_coupling_flow/artifacts/ka3d_cavity_ebm3ax_rho115_rl_demand_knn24.pt"
D = torch.load("liquid_coupling_flow/artifacts/ka3d_dataset_N4096_T0.5_rho1.15.pt", map_location=dev, weights_only=False)
X, S, L = D["x"].to(dev).float(), D["s"].to(dev).long(), float(D["L"])


def load(path, use_demand):
    m = KA3DScaffoldEBMBatched(cat_bins=128, cat_range=2.5, knn_pot=24, use_demand=use_demand).to(dev)
    m.load_state_dict(torch.load(path, map_location=dev, weights_only=False)["state_dict"], strict=False)
    m.eval(); m.use_frame = False
    return m


def energy_b(Xb, Sb, bnd, sb):
    Mn, mm = Xb.shape[0], bnd.shape[0]
    x = torch.cat([Xb, bnd[None].expand(Mn, mm, 3)], 1); s = torch.cat([Sb, sb[None].expand(Mn, mm)], 1)
    return ka_energy(x.double(), s.long(), BIGL).float()


@torch.no_grad()
def mtm(m, K, gen):
    accs, des = [], []
    ncav = 0
    for ci in range(16):
        c = torch.rand(3, generator=gen, device=dev) * L; p = carve(X[ci], S[ci], c, R, L)
        if p["n_in"] < K + 6:
            continue
        xo, so, _ = label_to_scaffold(_mic(p["x_in"], c, L), p["s_in"], R)
        xout = _mic(p["x_out"], c, L); bm = xout.norm(dim=-1) < (R + RCTX); bnd, sb = xout[bm], p["s_out"][bm]
        n = xo.shape[0]; a = fixed_ball_scaffold(n, R, dev, xo.dtype, ordering=m.scaffold_order)
        seed = int(torch.randint(n, (), generator=gen, device=dev))
        blk = torch.zeros(n, dtype=torch.bool, device=dev)
        blk[(a - a[seed]).norm(dim=-1).topk(K, largest=False).indices] = True
        Xr = xo[None].expand(M, n, 3).contiguous(); Sr = so[None].expand(M, n).contiguous()
        Xp, Sp, lqf = m.sample_block_b(Xr, Sr, blk, bnd, sb, R, gen=gen)
        Ep = energy_b(Xp, Sp, bnd, sb); E0 = energy_b(xo[None], so[None], bnd, sb)[0]
        up = -BETA * Ep - lqf
        u0 = (-BETA * E0 - m.block_log_prob_b(xo[None], so[None], blk, bnd, sb, R))[0]
        sf = torch.logsumexp(up, 0)
        J = int(torch.multinomial(torch.softmax(up, 0), 1, generator=gen)); lr = up.clone(); lr[J] = u0
        accs.append(float(torch.rand((), device=dev, generator=gen).log() < (sf - torch.logsumexp(lr, 0))))
        des.append(float((Ep.median() - E0) / K))
        ncav += 1
        if ncav >= NCAV:
            break
    return 100 * st.mean(accs), st.median(des)


mb = load(BASE, False); mr = load(RLD, True)
print(f"=== exact tempered block-MTM acceptance, base vs RL+demand (step500), R={R}, 5-seed ===", flush=True)
print(f"{'K':>3} | {'base accept':>18} {'base dE/p':>10} | {'RL accept':>18} {'RL dE/p':>10}", flush=True)
for K in (4, 6, 8):
    ab = [mtm(mb, K, torch.Generator(device=dev).manual_seed(s)) for s in range(5)]
    ar = [mtm(mr, K, torch.Generator(device=dev).manual_seed(s)) for s in range(5)]
    accb = [x[0] for x in ab]; accr = [x[0] for x in ar]
    print(f"{K:>3} | {st.mean(accb):6.1f} +/- {st.pstdev(accb):4.1f}%   {st.median([x[1] for x in ab]):>+10.1f} | "
          f"{st.mean(accr):6.1f} +/- {st.pstdev(accr):4.1f}%   {st.median([x[1] for x in ar]):>+10.1f}", flush=True)
