"""VERDICT for the knn=24 tilt RETRAIN (the top lever). The in-loop metric (single-seed, N=16, K=4 only)
was too noisy and never tested K=8. Here: exact tempered block-MTM acceptance for three models, K=4 AND
K=8, R=2.0, averaged over SEEDS x cavities for error bars --
  (A) baseline knn=8   model @ inference knn_pot=8   (ka3d_cavity_ebm3ax_rho115_best.pt)
  (B) baseline knn=8   model @ inference knn_pot=24  (zero-shot bigger cage -- the inference control)
  (C) retrained knn=24 model @ inference knn_pot=24  (ka3d_cavity_ebm3ax_rho115_knn24_best.pt)
If (C) > (B) the retrain LEARNED to use the cage (its real value). If (C) ~ (B) ~ (A) the +1/particle
structural floor holds and the lever is null. pos_temp=1.0 (measurable). Also reports median dE/particle
of accepted-vs-data block (the energy the proposal pays)."""
import torch, statistics as st
from liquid_coupling_flow.ka3d_ebm_batched import KA3DScaffoldEBMBatched
from liquid_coupling_flow.ka3d_scaffold_ar import label_to_scaffold, fixed_ball_scaffold
from liquid_coupling_flow.ka3d_cavity_ar import _mic
from liquid_coupling_flow.ka3d_cavity_carve import carve
from liquid_coupling_flow.ka_energy import ka_energy

dev = "cuda"; RCTX = 2.5; BIGL = 100.0; BETA = 2.0; N = 16; NCAV = 12; PT = 1.0
BASE = "liquid_coupling_flow/artifacts/ka3d_cavity_ebm3ax_rho115_best.pt"
KNN24 = "liquid_coupling_flow/artifacts/ka3d_cavity_ebm3ax_rho115_knn24_best.pt"
D = torch.load("liquid_coupling_flow/artifacts/ka3d_dataset_N4096_T0.5_rho1.15.pt", map_location=dev, weights_only=False)
X, S, L = D["x"].to(dev).float(), D["s"].to(dev).long(), float(D["L"])


def load(path):
    m = KA3DScaffoldEBMBatched(cat_bins=128, cat_range=2.5).to(dev)
    m.load_state_dict(torch.load(path, map_location=dev, weights_only=False)["state_dict"], strict=False)
    m.eval(); m.use_frame = False
    return m


def energy_b(Xb, Sb, bnd, sb):
    Mn, mm = Xb.shape[0], bnd.shape[0]
    x = torch.cat([Xb, bnd[None].expand(Mn, mm, 3)], 1); s = torch.cat([Sb, sb[None].expand(Mn, mm)], 1)
    return ka_energy(x.double(), s.long(), BIGL).float()


@torch.no_grad()
def accept(m, R, K, gen):
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
        up = -BETA * energy_b(Xp, Sp, bnd, sb) - lqf
        Xd = xo[None].contiguous(); Sd = so[None].contiguous()
        u0 = (-BETA * energy_b(Xd, Sd, bnd, sb) - m.block_log_prob_b(Xd, Sd, blk, bnd, sb, R, pos_temp=PT))[0]
        sf = torch.logsumexp(up, 0)
        J = int(torch.multinomial(torch.softmax(up, 0), 1, generator=gen))
        lr = up.clone(); lr[J] = u0; sr = torch.logsumexp(lr, 0)
        accs.append(float(torch.rand((), device=dev, generator=gen).log() < (sf - sr)))
        dEs.append(float((energy_b(Xp, Sp, bnd, sb).median() - energy_b(Xd, Sd, bnd, sb)[0]) / K))
        ncav += 1
        if ncav >= NCAV:
            break
    return 100 * st.mean(accs), st.median(dEs)


SEEDS = (0, 1, 2, 3, 4)
print(f"=== knn=24 RETRAIN verdict: exact tempered block-MTM, {len(SEEDS)} seeds x {NCAV} cav, pos_temp={PT} ===", flush=True)
print(f"    (mean +/- std over seeds; dE = median accepted-block energy per particle vs data block)", flush=True)
configs = [("A baseline knn8  @infer8 ", load(BASE),  8),
           ("B baseline knn8  @infer24", None,        24),   # reuse A's weights, bigger inference cage
           ("C RETRAIN  knn24 @infer24", load(KNN24), 24)]
mB = configs[0][1]  # baseline weights for config B
for label, m, kp in configs:
    mm = mB if m is None else m
    mm.knn_pot = kp
    for K in (4, 8):
        accs, des = [], []
        for sd in SEEDS:
            gen = torch.Generator(device=dev).manual_seed(sd)
            a, d = accept(mm, 2.0, K, gen)
            accs.append(a); des.append(d)
        mean = st.mean(accs); sdv = st.pstdev(accs)
        print(f"  {label}  K{K}:  {mean:5.1f} +/- {sdv:4.1f}%   dE/p {st.median(des):+7.1f}", flush=True)
