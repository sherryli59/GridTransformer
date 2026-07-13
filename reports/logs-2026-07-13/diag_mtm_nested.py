"""Does the NESTED min-sep mask (+ bigger cage) help MTM ACCEPTANCE? Never measured -- prior MTM runs
used the per-axis head with no mask. Here: coarse head, exact tempered block-MTM (current = data block),
three configs -- (knn8, no mask), (knn24, no mask), (knn24, NESTED mask cut=0.9). knn_pot sets BOTH the
tilt cage AND the nested-mask cage, so mask sees 24 neighbours and sample==score stays exact (verified
1.4e-14 earlier). pos_temp=1.0 (measurable; pos_temp=0.4 is bimodal-0). Tests whether declashing the
placement moves acceptance, or whether the +22-nat structural floor (ceiling test) holds regardless."""
import torch, statistics as st
from liquid_coupling_flow.ka3d_coarse_head import KA3DScaffoldEBMCoarse
from liquid_coupling_flow.ka3d_scaffold_ar import label_to_scaffold, fixed_ball_scaffold
from liquid_coupling_flow.ka3d_cavity_ar import _mic
from liquid_coupling_flow.ka3d_cavity_carve import carve
from liquid_coupling_flow.ka_energy import ka_energy

dev = "cuda"; RCTX = 2.5; BIGL = 100.0; BETA = 2.0; N = 16; NCAV = 12; PT = 1.0
m = KA3DScaffoldEBMCoarse(cat_bins=128, cat_range=2.5).to(dev)
m.load_state_dict(torch.load("liquid_coupling_flow/artifacts/ka3d_cavity_coarse_best.pt", map_location=dev, weights_only=False)["state_dict"], strict=False)
m.eval(); m.use_frame = False
D = torch.load("liquid_coupling_flow/artifacts/ka3d_dataset_N4096_T0.5_rho1.15.pt", map_location=dev, weights_only=False)
X, S, L = D["x"].to(dev).float(), D["s"].to(dev).long(), float(D["L"])


def energy_b(Xb, Sb, bnd, sb):
    Mn, mm = Xb.shape[0], bnd.shape[0]
    x = torch.cat([Xb, bnd[None].expand(Mn, mm, 3)], 1); s = torch.cat([Sb, sb[None].expand(Mn, mm)], 1)
    return ka_energy(x.double(), s.long(), BIGL).float()


@torch.no_grad()
def accept(R, K, kw, gen):
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
        Xp, Sp, lqf = m.sample_block_b(Xr, Sr, blk, bnd, sb, R, gen=gen, pos_temp=PT, **kw)
        up = -BETA * energy_b(Xp, Sp, bnd, sb) - lqf
        Xd = xo[None].contiguous(); Sd = so[None].contiguous()
        u0 = (-BETA * energy_b(Xd, Sd, bnd, sb) - m.block_log_prob_b(Xd, Sd, blk, bnd, sb, R, pos_temp=PT, **kw))[0]
        sf = torch.logsumexp(up, 0)
        J = int(torch.multinomial(torch.softmax(up, 0), 1, generator=gen))
        lr = up.clone(); lr[J] = u0; sr = torch.logsumexp(lr, 0)
        accs.append(float(torch.rand((), device=dev, generator=gen).log() < (sf - sr)))
        dEs.append(float((energy_b(Xp, Sp, bnd, sb).median() - energy_b(Xd, Sd, bnd, sb)[0]) / K))
        ncav += 1
        if ncav >= NCAV:
            break
    return 100 * st.mean(accs), st.median(dEs)


print(f"=== coarse-head block-MTM acceptance: nested mask effect (pos_temp={PT}, N={N}, {NCAV} cav) ===", flush=True)
for knn, kw, label in [(8, {}, "knn8  no-mask"),
                       (24, {}, "knn24 no-mask"),
                       (24, {"min_sep": 0.9, "nested": True}, "knn24 NESTED-mask")]:
    m.knn_pot = knn
    row = f"  {label:18s}:"
    for R in (2.0,):
        for K in (4, 8):
            gen = torch.Generator(device=dev).manual_seed(0)
            acc, de = accept(R, K, kw, gen)
            row += f"  R{R}K{K}: {acc:4.1f}% (dE/p{de:+7.1f})"
    print(row, flush=True)
