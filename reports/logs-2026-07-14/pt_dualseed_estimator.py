"""DUAL-SEED PT ESTIMATOR for cavity PTS q(R) (option 2 -- the production path). Exact by construction:
thermal replicas pi_r ~ e^{-beta_r U} (NO model density in any bath), AR suffix+local block moves as the
per-replica kernel (exact MH: log a = -beta_r dU + lqr - lqf), standard replica exchange, and the stack-
agreement certificate: stack A seeded at the reference (from above), stack B seeded from the STRATIFIED
alien pool (from below -- candidates incl. true basin members found by the draw->prescreen pipeline, hot
replicas from bulk draws). PT is init-independent at convergence; seeds only accelerate; A==B at the cold
rung = converged, and the shared value is q(R). Incremental per-sweep saves (checkpoint directive).
beta ladder 0.4->2.0 linear x NR=10; 2 chains/stack; SW sweeps of (1 suffix + 2 local) + exchange each sweep.
Cold-replica q~ logged every 25 sweeps. lam05 Rext model as proposal, R=2.0, cavities 0/1."""
import sys, functools, time
import torch
sys.path.insert(0, "reports/logs-2026-07-13")
from ka3d_smc_sweep import energy_b, blob_mask
from liquid_coupling_flow.ka3d_ebm_batched import KA3DScaffoldEBMBatched
from liquid_coupling_flow.ka3d_scaffold_ar import label_to_scaffold, fixed_ball_scaffold
from liquid_coupling_flow.ka3d_cavity_ar import _mic
from liquid_coupling_flow.ka3d_cavity_carve import carve
torch.set_grad_enabled(False)

dev = "cuda"; RCTX = 2.5; BETA_T = 2.0; L_BOX = 0.368; RHO = 1.149; BULK = RHO * L_BOX ** 3
K = 8; R = 2.0; POOL = 2048; BATCH = 512; ART = "liquid_coupling_flow/artifacts"
NR = 10; NCH = 2; SW = 250; REPORT = 25; BURN_FRAC = 0.4
OUT = "reports/logs-2026-07-14/pt_dualseed_estimator.pt"
m = KA3DScaffoldEBMBatched(cat_bins=128, cat_range=2.5, knn_pot=24, use_demand=True).to(dev)
m.load_state_dict(torch.load(f"{ART}/ka3d_cavity_ebm3ax_rho115_rl_Rext_knn24_lam05_best.pt",
                             map_location=dev, weights_only=False)["state_dict"], strict=False)
m.eval(); m.use_frame = False
D = torch.load(f"{ART}/ka3d_dataset_N4096_T0.5_rho1.15.pt", map_location=dev, weights_only=False)
X, S, L = D["x"].to(dev).float(), D["s"].to(dev).long(), float(D["L"])
BETAS = torch.linspace(0.4, BETA_T, NR, device=dev)                      # replica ladder


@functools.lru_cache(maxsize=16)
def n_boxes(l, R):
    g = torch.arange(-int(R / l) - 1, int(R / l) + 2) * l + l / 2
    gx, gy, gz = torch.meshgrid(g, g, g, indexing="ij")
    return int((gx ** 2 + gy ** 2 + gz ** 2 < R * R).sum())


def box_id_set(x, l, R):
    mm = x.norm(dim=-1) < R; ijk = torch.floor(x[mm] / l).long(); off = ijk - ijk.min(0).values
    span = off.max(0).values + 1
    return set((off[:, 0] * span[1] * span[2] + off[:, 1] * span[2] + off[:, 2]).tolist())


def qtil_one(a, xin, R):
    return len(box_id_set(a, L_BOX, R) & box_id_set(xin, L_BOX, R)) / (L_BOX ** 3 * n_boxes(L_BOX, R)) - BULK


results = {}
gen = torch.Generator(device=dev).manual_seed(0); ncav = 0
for ci in range(24):
    c = torch.rand(3, generator=gen, device=dev) * L; p = carve(X[ci], S[ci], c, R, L)
    if p["n_in"] < 14:
        continue
    t0 = time.time()
    xo, so, _ = label_to_scaffold(_mic(p["x_in"], c, L), p["s_in"], R)
    xin = _mic(p["x_in"], c, L)
    xout = _mic(p["x_out"], c, L); bm = xout.norm(dim=-1) < (R + RCTX); bnd, sb = xout[bm], p["s_out"][bm]
    n = xo.shape[0]; allm = torch.ones(n, dtype=torch.bool, device=dev)
    a = fixed_ball_scaffold(n, R, dev, xo.dtype, ordering=m.scaffold_order)
    g = torch.Generator(device=dev).manual_seed(700 + ci)
    # --- stratified pool for stack-B seeds ---
    allX, allS, allq = [], [], []
    for b0 in range(0, POOL, BATCH):
        nb = min(BATCH, POOL - b0)
        Xa, Sa, _ = m.sample_block_b(xo[None].expand(nb, n, 3).clone(), so[None].expand(nb, n).clone(),
                                     allm, bnd, sb, R, gen=g)
        Xc = Xa.cpu(); allq += [qtil_one(Xc[k], xin.cpu(), R) for k in range(nb)]
        allX.append(Xc); allS.append(Sa.cpu())
    allX = torch.cat(allX); allS = torch.cat(allS); allq = torch.tensor(allq)
    top = allq.topk(NCH).indices                                          # best candidates -> B cold seeds
    rnd = torch.randperm(POOL)[:NR * NCH]                                 # bulk draws -> B hot/mid seeds
    # --- build stacks: state [2, NR, NCH, n, 3]; stack 0 = A(data), 1 = B(stratified alien) ---
    Xrep = torch.empty(2, NR, NCH, n, 3, device=dev); Srep = torch.empty(2, NR, NCH, n, dtype=torch.long, device=dev)
    Xrep[0] = xo[None, None]; Srep[0] = so[None, None]                    # stack A: all replicas at reference
    for r in range(NR):
        for ch in range(NCH):
            idx = int(top[ch]) if r == NR - 1 else int(rnd[r * NCH + ch]) # B: cold=candidates, rest=bulk
            Xrep[1, r, ch] = allX[idx].to(dev); Srep[1, r, ch] = allS[idx].to(dev)
    Xf = Xrep.reshape(-1, n, 3); Sf = Srep.reshape(-1, n)
    U = energy_b(Xf, Sf, bnd, sb)
    beta_f = BETAS[None, :, None].expand(2, NR, NCH).reshape(-1)
    ex_acc = torch.zeros(NR - 1); ex_try = torch.zeros(NR - 1); traj = []
    print(f"=== cav {ci} (n={n}) PT: {NR} replicas x {NCH} chains x 2 stacks, {SW} sweeps "
          f"(B cold seeds q~ {[round(float(allq[t]),2) for t in top]}) ===", flush=True)
    for sw in range(1, SW + 1):
        for kind in ("suf", "loc", "loc"):                                # replica mutation
            if kind == "suf":
                ks = int(torch.randint(n // 2, 3 * n // 4 + 1, (), generator=g, device=dev))
                mask = torch.zeros(n, dtype=torch.bool, device=dev); mask[n - ks:] = True
            else:
                mask = blob_mask(n, K, a, g)
            lqr = m.block_log_prob_b(Xf, Sf, mask, bnd, sb, R); Xn, Sn, lqf = m.sample_block_b(Xf, Sf, mask, bnd, sb, R, gen=g)
            En = energy_b(Xn, Sn, bnd, sb)
            la = -beta_f * (En - U) + lqr - lqf                           # thermal MH, no q0 in bath
            acc = torch.rand(len(Xf), device=dev, generator=g).log() < la
            Xf = torch.where(acc[:, None, None], Xn, Xf); Sf = torch.where(acc[:, None], Sn, Sf)
            U = torch.where(acc, En, U)
        Xrep = Xf.view(2, NR, NCH, n, 3); Srep = Sf.view(2, NR, NCH, n); Urep = U.view(2, NR, NCH)
        par = sw % 2                                                       # exchange, alternating parity
        for r in range(par, NR - 1, 2):
            dlog = (BETAS[r] - BETAS[r + 1]) * (Urep[:, r] - Urep[:, r + 1])   # [2, NCH]
            swp = torch.rand(2, NCH, device=dev, generator=g).log() < dlog
            ex_acc[r] += float(swp.float().sum()); ex_try[r] += swp.numel()
            for st in range(2):
                for ch in range(NCH):
                    if swp[st, ch]:
                        Xrep[st, [r, r + 1], ch] = Xrep[st, [r + 1, r], ch]
                        Srep[st, [r, r + 1], ch] = Srep[st, [r + 1, r], ch]
                        Urep[st, [r, r + 1], ch] = Urep[st, [r + 1, r], ch]
        Xf = Xrep.reshape(-1, n, 3); Sf = Srep.reshape(-1, n); U = Urep.reshape(-1)
        if sw % REPORT == 0:
            qA = [qtil_one(Xrep[0, -1, ch].cpu(), xin.cpu(), R) for ch in range(NCH)]
            qB = [qtil_one(Xrep[1, -1, ch].cpu(), xin.cpu(), R) for ch in range(NCH)]
            traj.append({"sweep": sw, "qA": qA, "qB": qB,
                         "U_cold_A": float(Urep[0, -1].mean()), "U_cold_B": float(Urep[1, -1].mean())})
            print(f"  sw {sw:>4}: cold q~ A {[f'{q:+.2f}' for q in qA]}  B {[f'{q:+.2f}' for q in qB]}  "
                  f"U/n A {float(Urep[0,-1].mean())/n:+.2f} B {float(Urep[1,-1].mean())/n:+.2f}", flush=True)
            results[(ci, "traj")] = traj; torch.save(results, OUT)         # incremental save
    nburn = int(len(traj) * BURN_FRAC)
    qAf = [q for rrow in traj[nburn:] for q in rrow["qA"]]; qBf = [q for rrow in traj[nburn:] for q in rrow["qB"]]
    import statistics as st
    agree = abs(st.mean(qAf) - st.mean(qBf))
    results[(ci, "final")] = {"qA": st.mean(qAf), "qA_sd": st.pstdev(qAf), "qB": st.mean(qBf),
                              "qB_sd": st.pstdev(qBf), "agree_gap": agree,
                              "exchange_acc": (ex_acc / ex_try.clamp(min=1)).tolist(),
                              "wall_s": time.time() - t0}
    torch.save(results, OUT)
    print(f"  FINAL (post-burn): A {st.mean(qAf):+.3f}+-{st.pstdev(qAf):.3f}  B {st.mean(qBf):+.3f}+-{st.pstdev(qBf):.3f}"
          f"  gap {agree:.3f}  exch_acc {[f'{x:.2f}' for x in (ex_acc/ex_try.clamp(min=1)).tolist()]}"
          f"  ({time.time()-t0:.0f}s)", flush=True)
    ncav += 1
    if ncav >= 2:
        break
print(f"saved -> {OUT}", flush=True)
print("CERTIFICATE: gap ~< within-stack sd => CONVERGED; the shared value = q(R) (box observable).", flush=True)
