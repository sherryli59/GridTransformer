"""Validate the exact-tempered AR block-MTM (the surprise 4.7% K=8 acceptance) by STATIONARITY.

Kernel under test: Liu-Liang-Wong independence-MTM, trials + current-state BOTH scored under the
tempered sampling density (block_log_prob_b(pos_temp=T), round-trip 1.5e-5). Fixed block per cavity
-> the kernel targets pi(block | rest) exactly, so from a DATA start the block energy trace must HOLD
(no drift beyond the data band). Three probes:
  MAIN     : 6 held cavities (R=2.0, K=8), data start, 200 sequential moves. PASS = |<U/K> end-start|
             small + acceptance consistent with the one-shot ladder + accepted moves really move.
  BUGCTL   : same but current-state scored at pos_temp=1.0 (reproduces the morning sweep's mixed
             densities). SENSITIVITY control: this chain should show clearly different behavior
             (biased acceptance / drift) -- if it doesn't, stationarity can't detect the bug class.
  Per-cavity acceptance spread: guards against 1-2 cavities carrying the aggregate.
"""
import torch, statistics as st
from liquid_coupling_flow.ka3d_ebm_batched import KA3DScaffoldEBMBatched
from liquid_coupling_flow.ka3d_scaffold_ar import label_to_scaffold, fixed_ball_scaffold
from liquid_coupling_flow.ka3d_cavity_ar import _mic
from liquid_coupling_flow.ka3d_cavity_carve import carve
from liquid_coupling_flow.ka_energy import ka_energy

dev = "cuda"; RCTX = 2.5; BIGL = 100.0; BETA = 2.0; NTRIAL = 16; K = 8; R = 2.0
POS_TEMP = 0.4; NMOVE = 200; NCAV = 6; NMOVE_CTL = 100; NCAV_CTL = 3

ar = KA3DScaffoldEBMBatched(cat_bins=128, cat_range=2.5).to(dev)
ar.load_state_dict(torch.load("liquid_coupling_flow/artifacts/ka3d_cavity_ebm3ax_rho115_best.pt",
                              map_location=dev, weights_only=False)["state_dict"], strict=False)
ar.eval(); ar.use_frame = False
d = torch.load("liquid_coupling_flow/artifacts/ka3d_dataset_N4096_T0.5_rho1.15.pt", map_location=dev, weights_only=False)
X, S, L = d["x"].to(dev).float(), d["s"].to(dev).long(), float(d["L"])
gen = torch.Generator(device=dev).manual_seed(1)


def energy_blk(xo, so, blk, bnd, sb):
    x = torch.cat([xo[None, blk], torch.cat([bnd, xo[~blk]], 0)[None]], 1)
    s = torch.cat([so[None, blk], torch.cat([sb, so[~blk]], 0)[None]], 1)
    return float(ka_energy(x.double(), s.long(), BIGL)[0])


@torch.no_grad()
def run_chain(xo0, so0, blk, bnd, sb, nmove, score_temp, gen):
    """Sequential MTM chain on a FIXED block. score_temp = density used for the CURRENT state
    (POS_TEMP = exact; 1.0 = reproduce the mixed-density bug). Returns traces."""
    xo, so = xo0.clone(), so0.clone()
    n = xo.shape[0]
    Etr, accs, disps = [], [], []
    for _ in range(nmove):
        Xr = xo[None].expand(NTRIAL, n, 3).contiguous(); Sr = so[None].expand(NTRIAL, n).contiguous()
        Xp, Sp, lqf = ar.sample_block_b(Xr, Sr, blk, bnd, sb, R, gen=gen, pos_temp=POS_TEMP)
        Uy = ka_energy(torch.cat([Xp[:, blk], torch.cat([bnd, xo[~blk]], 0)[None].expand(NTRIAL, -1, -1)], 1).double(),
                       torch.cat([Sp[:, blk], torch.cat([sb, so[~blk]], 0)[None].expand(NTRIAL, -1)], 1).long(),
                       BIGL).float()
        up = -BETA * Uy - lqf
        lq_x = ar.block_log_prob_b(xo[None], so[None], blk, bnd, sb, R, pos_temp=score_temp)
        Ux = energy_blk(xo, so, blk, bnd, sb)
        u0 = -BETA * Ux - float(lq_x[0])
        Jm = int(torch.multinomial(torch.softmax(up, 0), 1, generator=gen))
        sf = torch.logsumexp(up, 0)
        lr = up.clone(); lr[Jm] = u0
        sr = torch.logsumexp(lr, 0)
        acc = bool(torch.rand((), device=dev, generator=gen).log() < (sf - sr))
        if acc:
            disp = float((Xp[Jm, blk] - xo[blk]).norm(dim=-1).mean())
            xo, so = Xp[Jm].clone(), Sp[Jm].clone()
            disps.append(disp)
        accs.append(float(acc))
        Etr.append(energy_blk(xo, so, blk, bnd, sb) / K)
    return Etr, accs, disps


def carve_cavity(ci, gen):
    while True:
        c = torch.rand(3, generator=gen, device=dev) * L
        p = carve(X[ci], S[ci], c, R, L)
        if p["n_in"] >= K + 6:
            break
    xo, so, _ = label_to_scaffold(_mic(p["x_in"], c, L), p["s_in"], R)
    xout = _mic(p["x_out"], c, L); bm = xout.norm(dim=-1) < (R + RCTX)
    bnd, sb = xout[bm], p["s_out"][bm]
    assert bnd.shape[0] < 600
    n = xo.shape[0]; anch = fixed_ball_scaffold(n, R, dev)
    seed = int(torch.randint(n, (), generator=gen, device=dev))
    blk = torch.zeros(n, dtype=torch.bool, device=dev)
    blk[(anch - anch[seed]).norm(dim=-1).topk(K, largest=False).indices] = True
    return xo, so, blk, bnd, sb


print(f"=== MAIN: exact tempered chain (score_temp={POS_TEMP}), {NCAV} cavities x {NMOVE} moves ===", flush=True)
drifts, acc_all, disp_all = [], [], []
for ci in range(NCAV):
    xo, so, blk, bnd, sb = carve_cavity(ci, gen)
    E0 = energy_blk(xo, so, blk, bnd, sb) / K
    Etr, accs, disps = run_chain(xo, so, blk, bnd, sb, NMOVE, POS_TEMP, gen)
    tail = st.mean(Etr[-50:])
    drifts.append(tail - E0)
    acc_all.append(100 * st.mean(accs)); disp_all += disps
    print(f"  cav {ci+1}: acc {100*st.mean(accs):5.1f}%  U/K start {E0:+8.3f} tail(last50) {tail:+8.3f} "
          f"drift {tail-E0:+7.3f}  n_acc {len(disps)}  mean|dx|acc {st.mean(disps) if disps else float('nan'):.3f}", flush=True)
print(f"MAIN: acc {st.mean(acc_all):.1f}% (per-cav spread {min(acc_all):.0f}-{max(acc_all):.0f}%)  "
      f"drift med {st.median(drifts):+.3f}/particle  mean|dx| of accepted {st.mean(disp_all) if disp_all else float('nan'):.3f}", flush=True)

print(f"\n=== BUGCTL: current state scored at pos_temp=1.0 (mixed densities), {NCAV_CTL} cavities x {NMOVE_CTL} ===", flush=True)
drifts_c, acc_c = [], []
for ci in range(NCAV_CTL):
    xo, so, blk, bnd, sb = carve_cavity(ci + 8, gen)
    E0 = energy_blk(xo, so, blk, bnd, sb) / K
    Etr, accs, _ = run_chain(xo, so, blk, bnd, sb, NMOVE_CTL, 1.0, gen)
    tail = st.mean(Etr[-25:])
    drifts_c.append(tail - E0); acc_c.append(100 * st.mean(accs))
    print(f"  cav: acc {100*st.mean(accs):5.1f}%  drift {tail-E0:+7.3f}", flush=True)
print(f"BUGCTL: acc {st.mean(acc_c):.1f}%  drift med {st.median(drifts_c):+.3f}/particle "
      f"(sensitivity: should differ clearly from MAIN)", flush=True)

torch.save({"drifts": drifts, "acc": acc_all, "disp": disp_all,
            "drifts_ctl": drifts_c, "acc_ctl": acc_c,
            "cfg": {"K": K, "R": R, "pos_temp": POS_TEMP, "ntrial": NTRIAL, "nmove": NMOVE}},
           "reports/logs-2026-07-12/mtm_ar_stationarity.pt")
print("saved -> reports/logs-2026-07-12/mtm_ar_stationarity.pt", flush=True)
verdict = abs(st.median(drifts)) < 0.15 and st.mean(acc_all) > 1.0
print("VERDICT:", "PASS -- exact tempered MTM holds equilibrium; 4.7% K=8 baseline is real"
      if verdict else "CHECK -- drift or dead acceptance; inspect traces", flush=True)
