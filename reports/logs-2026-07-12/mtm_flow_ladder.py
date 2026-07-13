"""Flow-MTM K-ladder: the decisive deployment number for the cavity EGNN block-flow corrector.

Liu-Liang-Wong independence-MTM on held-chain cavities, three arms sharing one harness:
  AR    : trials = (gated) AR draws, weights use the exact TEMPERED sampling density
          (block_log_prob_b(pos_temp=T) round-trip verified in test_tempered_score.py: 1.5e-5 median).
          NOTE the morning sweep's pos_temp<1 rows mixed densities (~6-nat current-state error) — this
          harness supersedes them.
  FLOW  : trials = flow-corrected (gated) AR draws, weights use the composed exact log-q
          (logq_ar_tempered + logdet). Current-state term via REVERSE flow: x0_rev, ld_rev =
          flow(x, reverse=True); logq(x) = block_log_prob_b(x0_rev, pos_temp) - ld_rev (fwd/rev logdet
          cancellation 4e-7, test_egnn3d_flow.out). Reverse-solve failure or gate-failed reverse image
          -> move REJECTED (q_gated(x)=0: valid, conservative; counted separately).
  Gate  : optional truncated base (ka3d_gated_base). Z_A cancels in every ratio (same cage both sides).
dopri5 AssertionError on a trial batch -> that batch's trials get weight -inf (outside support after
guard), never a crash. Current state = the DATA block (equilibrium acceptance, the regime that matters
for the PTS/SMC deployment).

Usage:
  python mtm_flow_ladder.py --arm ar   --pos_temp 0.4                       # AR baseline (no ODE)
  python mtm_flow_ladder.py --arm flow --flow_ckpt <path> [--gate_cut 0.55] # flow arm
"""
import argparse
import torch, statistics as st
from liquid_coupling_flow.ka3d_ebm_batched import KA3DScaffoldEBMBatched
from liquid_coupling_flow.ka3d_cavity_egnn import CavityBlockFlow
from liquid_coupling_flow.ka3d_gated_base import GatedARBase, gate_pass
from liquid_coupling_flow.ka3d_scaffold_ar import label_to_scaffold, fixed_ball_scaffold
from liquid_coupling_flow.ka3d_cavity_ar import _mic
from liquid_coupling_flow.ka3d_cavity_carve import carve
from liquid_coupling_flow.ka_energy import ka_energy

P = argparse.ArgumentParser()
P.add_argument("--arm", choices=["ar", "flow"], required=True)
P.add_argument("--flow_ckpt", default="liquid_coupling_flow/artifacts/ka3d_cavity_egnn_flow_best.pt")
P.add_argument("--gate_cut", type=float, default=None)
P.add_argument("--pos_temp", type=float, default=0.4)
P.add_argument("--ntrial", type=int, default=16)
P.add_argument("--nrep", type=int, default=8, help="independent MTM moves per cavity (fresh trials)")
P.add_argument("--ncav", type=int, default=8)
P.add_argument("--Ks", type=int, nargs="+", default=[4, 8])
P.add_argument("--R", type=float, default=2.0)
P.add_argument("--beta", type=float, default=2.0)
P.add_argument("--device", default="cuda")
P.add_argument("--out_tag", default="")
P.add_argument("--ode_max_steps", type=int, default=1000,
               help="deployment dopri5 budget: healthy solves need a few hundred; stiff proposals get "
                    "rejected fast instead of grinding to 10k (70-min-first-move lesson)")
P.add_argument("--ode_double", action="store_true", help="double-precision ODE (default float32)")
args = P.parse_args()
dev = args.device; RCTX = 2.5; BIGL = 100.0; N_CAGE = 48; DUMMY_R = 50.0

ar = KA3DScaffoldEBMBatched(cat_bins=128, cat_range=2.5).to(dev)
ar.load_state_dict(torch.load("liquid_coupling_flow/artifacts/ka3d_cavity_ebm3ax_rho115_best.pt",
                              map_location=dev, weights_only=False)["state_dict"], strict=False)
ar.eval(); ar.use_frame = False
base = GatedARBase(ar, cut=args.gate_cut) if args.gate_cut is not None else ar

flow = None
FLOW_DTYPE = torch.float64 if args.ode_double else torch.float32
if args.arm == "flow":
    ck = torch.load(args.flow_ckpt, map_location=dev, weights_only=False)
    a = ck["args"]
    flow = CavityBlockFlow(k=a["k"], n_cage=a["n_cage"], r_c=a.get("r_c", 2.5), hidden_nf=a["hidden_nf"],
                           n_layers=a["n_layers"], n_species=2, max_neighbors=a["max_neighbors"],
                           rep_prior=a.get("rep_prior", False),
                           ode_rtol=1e-6 if args.ode_double else 1e-5,
                           ode_atol=1e-6 if args.ode_double else 1e-5,
                           max_steps=args.ode_max_steps).to(dev).to(FLOW_DTYPE)
    flow.load_state_dict(ck["state_dict"]); flow.eval()
    print(f"flow ckpt step {ck['step']} held_fm {ck['held_loss']:.4f} "
          f"(hid={a['hidden_nf']} L={a['n_layers']} gate_cut(train)={a.get('gate_cut')})", flush=True)

d = torch.load("liquid_coupling_flow/artifacts/ka3d_dataset_N4096_T0.5_rho1.15.pt", map_location=dev, weights_only=False)
X, S, L = d["x"].to(dev).float(), d["s"].to(dev).long(), float(d["L"])
gen = torch.Generator(device=dev).manual_seed(0)


def fixed_size_cage(cage_x, sp_cage, centroid, n_cage):
    Mn, m = cage_x.shape[0], cage_x.shape[1]
    if m >= n_cage:
        idx = torch.topk((cage_x[0] - centroid).norm(dim=-1), n_cage, largest=False).indices
        return cage_x[:, idx], sp_cage[:, idx]
    pad = n_cage - m
    dpos = torch.tensor([DUMMY_R, 0., 0.], device=dev, dtype=cage_x.dtype)[None, None].expand(Mn, pad, 3)
    return torch.cat([cage_x, dpos], 1), torch.cat([sp_cage, torch.zeros(Mn, pad, dtype=sp_cage.dtype, device=dev)], 1)


def energy_b(Xb, Sb, bnd, sb):
    Mn, mm = Xb.shape[0], bnd.shape[0]
    x = torch.cat([Xb, bnd[None].expand(Mn, mm, 3)], 1); s = torch.cat([Sb, sb[None].expand(Mn, mm)], 1)
    return ka_energy(x.double(), s.long(), BIGL)


@torch.no_grad()
def one_move(xo, so, blk, bnd, sb, K, gen):
    """One MTM move from the DATA state. Returns (accepted, diag dict) or None (move skipped)."""
    n = xo.shape[0]
    Xr = xo[None].expand(args.ntrial, n, 3).contiguous(); Sr = so[None].expand(args.ntrial, n).contiguous()
    Xp, Sp, lqf = base.sample_block_b(Xr, Sr, blk, bnd, sb, args.R, gen=gen, pos_temp=args.pos_temp)
    trial_ok = base.last_pass_mask.clone() if isinstance(base, GatedARBase) else torch.ones(
        args.ntrial, dtype=torch.bool, device=dev)
    logdet_f = torch.zeros(args.ntrial, device=dev)
    if args.arm == "flow":
        blk_x = Xp[:, blk]; blk_s = Sp[:, blk]
        cage_x = torch.cat([bnd[None].expand(args.ntrial, -1, -1), Xp[:, ~blk]], 1)
        cage_s = torch.cat([sb[None].expand(args.ntrial, -1), Sp[:, ~blk]], 1)
        centroid = xo[blk].mean(0)
        cfx, cfs = fixed_size_cage(cage_x, cage_s, centroid, N_CAGE)
        try:
            y_blk, ld = flow.flow(blk_x.to(FLOW_DTYPE), cfx.to(FLOW_DTYPE), blk_s, cfs, reverse=False)
            Xp = Xp.clone(); Xp[:, blk] = y_blk.float()
            logdet_f = ld.float()
        except AssertionError as e:                       # stiff batch: all trials out of support
            return None, {"skip": f"fwd-ODE {e}"}
    # trial weights u(y) = -beta U(y) - logq(y); logq(y) = lq_ar_tempered + logdet_f
    Uy = energy_b(Xp[:, blk].float(), Sp[:, blk], torch.cat([bnd, xo[~blk]], 0),
                  torch.cat([sb, so[~blk]], 0)).float()
    up = -args.beta * Uy - (lqf + logdet_f)
    up[~trial_ok] = -torch.inf
    if not torch.isfinite(up).any():
        return None, {"skip": "no in-support trial"}
    # current (data) state weight
    Xb0 = xo[None].contiguous(); Sb0 = so[None].contiguous()
    if args.arm == "flow":
        cage0 = torch.cat([bnd[None], xo[None, ~blk]], 1)
        cage0s = torch.cat([sb[None], so[None, ~blk]], 1)
        cfx0, cfs0 = fixed_size_cage(cage0, cage0s, xo[blk].mean(0), N_CAGE)
        try:
            x0_rev, ld_rev = flow.flow(xo[None, blk].to(FLOW_DTYPE), cfx0.to(FLOW_DTYPE), so[None, blk], cfs0, reverse=True)
        except AssertionError as e:
            return False, {"rej": f"rev-ODE {e}"}       # q(x)=undefined -> reject conservatively
        x0f = x0_rev.float()
        if args.gate_cut is not None and not gate_pass(x0f, so[None, blk], cage0, cage0s, args.gate_cut)[0]:
            return False, {"rej": "reverse image outside gated support"}
        Xrev = xo[None].clone(); Xrev[:, blk] = x0f
        try:
            # reverse image can exit the open ball (seen with the rep_prior field): AR density there is 0
            # -> q(current)=0 -> conservative rejection, same class as the gate-fail branch above
            lq_x = ar.block_log_prob_b(Xrev, Sb0, blk, bnd, sb, args.R, pos_temp=args.pos_temp) - ld_rev.float()
        except ValueError as e:
            return False, {"rej": f"reverse image outside ball ({e})"}
    else:
        if args.gate_cut is not None:
            cage0 = torch.cat([bnd[None], xo[None, ~blk]], 1)
            cage0s = torch.cat([sb[None], so[None, ~blk]], 1)
            if not gate_pass(xo[None, blk], so[None, blk], cage0, cage0s, args.gate_cut)[0]:
                return False, {"rej": "current state outside gated support"}
        lq_x = ar.block_log_prob_b(Xb0, Sb0, blk, bnd, sb, args.R, pos_temp=args.pos_temp)
    Ux = energy_b(xo[None, blk], so[None, blk], torch.cat([bnd, xo[~blk]], 0),
                  torch.cat([sb, so[~blk]], 0)).float()
    u0 = (-args.beta * Ux - lq_x)[0]
    sf = torch.logsumexp(up, 0)
    Jm = int(torch.multinomial(torch.softmax(up, 0), 1, generator=gen))
    lr = up.clone(); lr[Jm] = u0
    sr = torch.logsumexp(lr, 0)
    acc = bool(torch.rand((), device=dev, generator=gen).log() < (sf - sr))
    dE = float((Uy[Jm] - Ux[0]) / K)
    return acc, {"dE_sel": dE, "sf_sr": float(sf - sr), "u0": float(u0),
                 "up_max": float(up.max()), "Jm": Jm}


results = {}
moves = []
for K in args.Ks:
    accs, rejs, skips, dEs = [], 0, 0, []
    ncav = 0
    for ci in range(16):
        c = torch.rand(3, generator=gen, device=dev) * L; p = carve(X[ci], S[ci], c, args.R, L)
        if p["n_in"] < K + 6:
            continue
        xo, so, _ = label_to_scaffold(_mic(p["x_in"], c, L), p["s_in"], args.R)
        xout = _mic(p["x_out"], c, L); bm = xout.norm(dim=-1) < (args.R + RCTX)
        bnd, sb = xout[bm], p["s_out"][bm]
        assert bnd.shape[0] < 600, f"frame sanity: n_bnd={bnd.shape[0]}"
        n = xo.shape[0]; anch = fixed_ball_scaffold(n, args.R, dev)
        seed = int(torch.randint(n, (), generator=gen, device=dev))
        blk = torch.zeros(n, dtype=torch.bool, device=dev)
        blk[(anch - anch[seed]).norm(dim=-1).topk(K, largest=False).indices] = True
        cav_accs = []
        for rep in range(args.nrep):
            acc, info = one_move(xo, so, blk, bnd, sb, K, gen)
            if acc is None:
                skips += 1
                continue
            if "rej" in info:
                rejs += 1
            accs.append(float(acc)); cav_accs.append(float(acc))
            if "dE_sel" in info:
                dEs.append(info["dE_sel"])
            moves.append({"K": K, "ci": ci, "rep": rep, "acc": bool(acc),
                          "blk_idx": blk.nonzero(as_tuple=True)[0].tolist(),
                          "blk_species": so[blk].tolist(), **info})
        if cav_accs:
            print(f"    cav ci={ci}: acc {100*st.mean(cav_accs):5.1f}%  ({int(sum(cav_accs))}/{len(cav_accs)})",
                  flush=True)
        ncav += 1
        if ncav >= args.ncav:
            break
    acc_pct = 100 * st.mean(accs) if accs else float("nan")
    err = 100 * (st.pstdev(accs) / max(len(accs), 1) ** 0.5) if len(accs) > 1 else float("nan")
    de_med = st.median(dEs) if dEs else float("nan")
    results[K] = {"acc": acc_pct, "err": err, "n": len(accs), "rej_guard": rejs, "skips": skips,
                  "dE_sel_med": de_med, "ncav": ncav}
    print(f"K={K:2d}: acc {acc_pct:5.1f}+-{err:.1f}%  (n={len(accs)}, guard-rej {rejs}, skips {skips})  "
          f"dE/p(selected) med {de_med:+8.2f}  [{ncav} cavities]", flush=True)

tag = f"_{args.out_tag}" if args.out_tag else ""
out = f"reports/logs-2026-07-12/mtm_flow_ladder_{args.arm}{tag}.pt"
torch.save({"results": results, "args": vars(args), "moves": moves}, out)
print(f"saved -> {out}", flush=True)
