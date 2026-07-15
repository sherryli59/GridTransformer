"""COLD block-MH acceptance for the mW scaffold transformer kernel (the go/no-go for using it as the SMC/MC
mutation). Thermal bath e^{-beta U} (NO model density in the bath -> no reverse-ODE curse; the AR block density
is one cheap forward). Move: regenerate a full cavity interior (allmask) with sample_block_b; exact MH
  log a = -beta*dU_block(du_block) + logq_rev - logq_fwd            (single-try)
plus the Liu I-MTM multi-try variant (select best-of-8 by w = -beta*dU - logq). Measured on EQUILIBRATED bank
configs at beta_target=1/T_STAR (worst case: any move disturbs equilibrium). KA glass gave ~0-1%; hand-off
predicts mW better. Report acceptance + mean dU per proposal vs R, and du_block cost units."""
import sys, statistics as st
import torch
from liquid_coupling_flow.mw.mw_scaffold import make_mw_scaffold, carve_mw_cavity
from liquid_coupling_flow.mw.mw_energy import du_block, T_STAR, EnergyEvalCounter, count_energy_evals
torch.set_grad_enabled(False)

dev = torch.device("cuda"); BETA = 1.0 / T_STAR
CK = "liquid_coupling_flow/mw/artifacts/mw_scaffold_runs/mw_scaffold_N512_varK_rl_lam002_b8.pt"
BANK = "liquid_coupling_flow/mw/artifacts/mw_ref_N512_traj_train_v1.pt"
ck = torch.load(CK, map_location=dev, weights_only=False)
m = make_mw_scaffold(cat_bins=ck["cat_bins"], cat_range=ck["cat_range"], knn=ck["knn"],
                     knn_bnd=ck["knn_bnd"], knn_pot=ck["knn_pot"], bnd_cutoff=ck["bnd_cutoff"]).to(dev)
m.load_state_dict(ck["state_dict"]); m.eval(); m.use_frame = False
L = float(ck["L"])
obj = torch.load(BANK, map_location="cpu", weights_only=False)
X = (obj["cfgs"] if isinstance(obj, dict) and "cfgs" in obj else obj).float()
X = X[-16:].to(dev)                                             # late frames = well-equilibrated
M_TRY = 8; NCAV = 6; NPROP = 16
print(f"=== mW cold block-MH acceptance (beta={BETA:.3f}, ckpt step {ck['step']}, RL lam={ck.get('rl_lambda')}) ===", flush=True)
print(f"{'R':>5} {'K':>4} | {'acc_1try':>8} {'acc_mtm8':>8} | {'dU/prop':>8} {'dU_acc':>7} | {'lqr-lqf':>8}", flush=True)
for R in (1.25, 1.55, 2.0):
    a1, amtm, dus, dua, dlq = [], [], [], [], []
    gen = torch.Generator(device=dev).manual_seed(0)
    counter = EnergyEvalCounter()
    with count_energy_evals(counter):
        ncav = 0
        for ci in range(64):
            x = X[ci % len(X)]
            center = torch.rand(3, generator=gen, device=dev) * L
            try:
                c = carve_mw_cavity(x, center, R, L, boundary_buffer=2.0)
            except ValueError:
                continue
            K = c["xo"].shape[0]
            if K < 4:
                continue
            xo, so, bnd, sb, bm = c["xo"], c["so"], c["bnd"], c["s_bnd"], c["block_mask"]
            order = c["label_order"]; idx_in = c["pair"]["idx_in"]; ctr = c["pair"]["center"]
            lqr = m.block_log_prob_b(xo[None], so[None], bm, bnd, sb, R)[0]
            Xn, Sn, lqf = m.sample_block_b(xo[None].expand(NPROP, K, 3).clone(),
                                           so[None].expand(NPROP, K).clone(), bm, bnd, sb, R, gen=gen)
            raw = Xn.new_empty(Xn.shape); raw[:, order] = Xn                 # scaffold -> raw interior order
            x_new = torch.remainder(ctr[None, None] + raw, L)                # box coords [NPROP,K,3]
            dU = du_block(x[None].expand(NPROP, -1, 3), idx_in, x_new, L)    # [NPROP]
            la1 = -BETA * dU + lqr - lqf
            acc1 = torch.rand(NPROP, device=dev, generator=gen).log() < la1
            a1.append(float(acc1.float().mean()))
            dus.append(float(dU.mean()))
            if acc1.any():
                dua.append(float(dU[acc1].mean()))
            dlq.append(float((lqr - lqf).mean()))
            # I-MTM with the SAME NPROP draws: forward best-of-M_TRY from the first M_TRY; reverse = fresh
            lwf = (-BETA * dU - lqf)[:M_TRY]
            J = int(torch.multinomial(torch.softmax(lwf, 0), 1, generator=gen))
            lwr = (-BETA * dU - lqf)[M_TRY:2 * M_TRY - 1] if NPROP >= 2 * M_TRY - 1 else lwf[:M_TRY - 1]
            lw_cur = -lqr                                                     # dU(current)=0
            la_m = torch.logsumexp(lwf, 0) - torch.logsumexp(torch.cat([lwr, lw_cur[None]]), 0)
            amtm.append(float(torch.rand((), device=dev, generator=gen).log() < la_m))
            ncav += 1
            if ncav >= NCAV:
                break
    print(f"{R:>5} {K:>4} | {st.mean(a1):>8.3f} {st.mean(amtm):>8.3f} | {st.mean(dus):>8.2f} "
          f"{(st.mean(dua) if dua else float('nan')):>7.2f} | {st.mean(dlq):>8.2f}", flush=True)
print(f"\ncost note: one du_block on [B={NPROP}] costs counter units {counter.as_dict()['cost_units']:.0f} total this R", flush=True)
print("acc>~5% at any R => transformer kernel LIVE for the thermal SMC/MC (Phase D arm 3).", flush=True)
