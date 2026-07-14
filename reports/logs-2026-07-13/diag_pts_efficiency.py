"""QUICK: does better proposal quality (RL+demand) buy PTS-sampler efficiency? Lean AIS bridge per cavity:
full-regen initial pop from the model (allmask), tempered path pi_lam ~ q_model^(1-lam) * e^(-lam*beta*U),
K-block MTM + K=1 heat-bath mutations (same model), quartic schedule. Efficiency = ESS sustained through the
anneal (higher = fewer wasted particles = shorter effective path) + final population energy/clash. Compare
base (block-cond) vs RL+demand. N=4096 R=2.5, M=32, T=12, few cavities. Exactness not the point here (both
use each model's own exact q); this is a RELATIVE efficiency read."""
import torch, statistics as st
from liquid_coupling_flow.ka3d_ebm_batched import KA3DScaffoldEBMBatched
from liquid_coupling_flow.ka3d_scaffold_ar import label_to_scaffold, fixed_ball_scaffold
from liquid_coupling_flow.ka3d_cavity_ar import _mic
from liquid_coupling_flow.ka3d_cavity_carve import carve
from liquid_coupling_flow.ka_energy import ka_energy, SIGMA

dev = "cuda"; RCTX = 2.5; R = 2.5; BIGL = 100.0; BETA = 2.0; M = 32; T = 12; NCAV = 6; CUT = 0.85
SIG = torch.tensor(SIGMA, device=dev)
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


def ess(logw):
    w = torch.softmax(logw, 0)
    return float(1.0 / (w * w).sum() / len(logw))                                  # ESS fraction [0,1]


def clash_frac(Xb, Sb, bnd, sb):
    allx = torch.cat([Xb, bnd[None].expand(Xb.shape[0], -1, 3)], 1); alls = torch.cat([Sb, sb[None].expand(Xb.shape[0], -1)], 1)
    n = Xb.shape[1]; d = torch.cdist(Xb, allx); sg = SIG[Sb[:, :, None], alls[:, None, :]]; cl = d < CUT * sg
    cl[:, torch.arange(n), torch.arange(n)] = False
    return float(cl[:, :, :n].sum((1, 2)).float().mean() / n)


@torch.no_grad()
def ais(m, xo, so, bnd, sb, R, gen):
    n = xo.shape[0]; allm = torch.ones(n, dtype=torch.bool, device=dev)
    Xb, Sb, _ = m.sample_block_b(xo[None].expand(M, n, 3).clone(), so[None].expand(M, n).clone(), allm, bnd, sb, R, gen=gen)
    Ecur = energy_b(Xb, Sb, bnd, sb); q0 = m.block_log_prob_b(Xb, Sb, allm, bnd, sb, R)
    lam = torch.linspace(0, 1, T + 1, device=dev) ** 4
    logw = torch.zeros(M, device=dev); esses = []
    for t in range(1, T + 1):
        logw = logw + float(lam[t] - lam[t - 1]) * (-BETA * Ecur - q0)
        esses.append(ess(logw))
        if ess(logw) < 0.5:
            w = torch.softmax(logw, 0); idx = torch.multinomial(w, M, replacement=True, generator=gen)
            Xb, Sb, Ecur, q0 = Xb[idx], Sb[idx], Ecur[idx], q0[idx]; logw = torch.zeros(M, device=dev)
        lt = float(lam[t])
        for _ in range(2):                                                          # K=8 block moves
            a = fixed_ball_scaffold(n, R, dev, xo.dtype, ordering=m.scaffold_order)
            seed = int(torch.randint(n, (), generator=gen, device=dev)); blk = torch.zeros(n, dtype=torch.bool, device=dev)
            blk[(a - a[seed]).norm(dim=-1).topk(8, largest=False).indices] = True
            lqr = m.block_log_prob_b(Xb, Sb, blk, bnd, sb, R); Xn, Sn, lqf = m.sample_block_b(Xb, Sb, blk, bnd, sb, R, gen=gen)
            En = energy_b(Xn, Sn, bnd, sb); q0n = m.block_log_prob_b(Xn, Sn, allm, bnd, sb, R)
            la = (1 - lt) * (q0n - q0) - lt * BETA * (En - Ecur) + lqr - lqf
            acc = torch.rand(M, device=dev, generator=gen).log() < la
            Xb = torch.where(acc[:, None, None], Xn, Xb); Sb = torch.where(acc[:, None], Sn, Sb)
            Ecur = torch.where(acc, En, Ecur); q0 = torch.where(acc, q0n, q0)
    return esses, float(Ecur.median()), clash_frac(Xb, Sb, bnd, sb)


mb = load(BASE, False); mr = load(RLD, True)
res = {"base": {"ess": [], "E": [], "cl": []}, "RL": {"ess": [], "E": [], "cl": []}}
gen = torch.Generator(device=dev).manual_seed(0); ncav = 0
for ci in range(16):
    c = torch.rand(3, generator=gen, device=dev) * L; p = carve(X[ci], S[ci], c, R, L)
    if p["n_in"] < 20:
        continue
    xo, so, _ = label_to_scaffold(_mic(p["x_in"], c, L), p["s_in"], R)
    xout = _mic(p["x_out"], c, L); bm = xout.norm(dim=-1) < (R + RCTX); bnd, sb = xout[bm], p["s_out"][bm]
    E0 = energy_b(xo[None], so[None], bnd, sb)[0]
    for name, m in (("base", mb), ("RL", mr)):
        es, Em, cl = ais(m, xo, so, bnd, sb, R, torch.Generator(device=dev).manual_seed(100 + ci))
        res[name]["ess"].append(es); res[name]["E"].append(float((Em - E0) / xo.shape[0])); res[name]["cl"].append(cl)
    ncav += 1
    if ncav >= NCAV:
        break

print(f"=== PTS-sampler efficiency: AIS bridge, base vs RL+demand ({ncav} cav, M={M}, T={T}) ===", flush=True)
print(f"  ESS fraction per rung (mean over cav):", flush=True)
print(f"    rung : " + " ".join(f"{t:>4}" for t in range(1, T + 1)), flush=True)
for name in ("base", "RL"):
    mean_ess = [st.mean([e[t] for e in res[name]["ess"]]) for t in range(T)]
    print(f"    {name:>4} : " + " ".join(f"{v:>4.2f}" for v in mean_ess), flush=True)
print(f"  final pop dE/particle:  base {st.mean(res['base']['E']):+.2f}   RL {st.mean(res['RL']['E']):+.2f}", flush=True)
print(f"  final pop clash/particle: base {st.mean(res['base']['cl']):.3f}   RL {st.mean(res['RL']['cl']):.3f}", flush=True)
print(f"  mean ESS over anneal:   base {st.mean([st.mean(e) for e in res['base']['ess']]):.3f}   "
      f"RL {st.mean([st.mean(e) for e in res['RL']['ess']]):.3f}", flush=True)
