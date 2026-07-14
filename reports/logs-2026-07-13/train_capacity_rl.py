"""FREE-RUN ENERGY LOSS (REINFORCE): the objective TF-MLE lacks. TF trains q(block|TRUE context) where room
is already left; it never penalizes free-run drift, so it can't teach 'leave room for later ranks'. Here we
add a policy-gradient term that minimizes the ENERGY of the model's OWN free-run block -- clash energy is
concentrated in the late ranks, so this directly teaches early members to leave room.

Energy blows up (LJ r->0 ~ 1e7) -> THREE guards: (1) PER-PAIR capped LJ (a clash contributes `cap`, not
1e7); (2) STANDARDIZED advantage (r-mean)/std over M_rl samples (leave-one-out-ish baseline, scale-free);
(3) TF ANCHOR (keep MLE-on-data so RL can't collapse to under-packed low-energy garbage). REINFORCE needs
only the reward VALUE, so the cap costs no gradient fidelity: loss_rl = mean( adv.detach() * logq ), logq =
block_log_prob_b(generated) WITH grad (exact re-score of the free-run sample). loss = loss_TF + lam*loss_rl.

Warm block-cond FT knn24. Primary metric = free-run blob clash/energy DOWN (NLL won't move). Eval saves
best on free-run capped-energy. K_rl blobs (the problem case)."""
import argparse, time, sys, statistics as st
from pathlib import Path
import torch
sys.path.insert(0, "reports/logs-2026-07-13")
from train_block_cond import build_pool, clamp_ball, rand_rot, block_mask, draw_kind, KINDS, WEIGHTS
from liquid_coupling_flow.ka3d_ebm_batched import KA3DScaffoldEBMBatched
from liquid_coupling_flow.ka3d_scaffold_ar import label_to_scaffold, fixed_ball_scaffold
from liquid_coupling_flow.ka_energy import SIGMA, EPS

ART = "liquid_coupling_flow/artifacts"
dev0 = "cuda" if torch.cuda.is_available() else "cpu"
SIG = torch.tensor(SIGMA, device=dev0); EP = torch.tensor(EPS, device=dev0)
CUT = 0.85


def capped_energy(x, s, cap):
    """REPULSIVE-overlap energy [M]: sum of per-pair clamp(u_LJ, min=0, max=cap). Only clashes (r<sigma,
    u>0) contribute; attractive contacts (u<0) -> 0, so this is a clash-specific penalty NOT contaminated by
    well depth (minimizing it declashes, does NOT reward over-packing). Cap bounds the r->0 blow-up."""
    M, N, _ = x.shape
    sig = SIG[s[:, :, None], s[:, None, :]]; eps = EP[s[:, :, None], s[:, None, :]]
    d2 = torch.cdist(x, x).clamp_min(1e-6) ** 2
    d2 = d2.masked_fill(torch.eye(N, device=x.device, dtype=torch.bool), 1e12)
    inv6 = (sig ** 2 / d2) ** 3
    e = 4 * eps * (inv6 ** 2 - inv6)                                               # raw LJ (unshifted)
    return 0.5 * e.clamp(min=0.0, max=cap).sum(dim=(1, 2))                         # [M] repulsive-only, capped


def clash_frac(x, s, blk_idx, bnd, sb):
    xi, si = x[blk_idx], s[blk_idx]
    allx = torch.cat([x, bnd]); alls = torch.cat([s, sb])
    d = torch.cdist(xi, allx); sg = SIG[si[:, None], alls[None, :]]; cl = d < CUT * sg
    for j, ii in enumerate(blk_idx.tolist()):
        cl[j, ii] = False
    return float(cl.sum(1).float().mean())


@torch.no_grad()
def evaluate(m, held, dev, cap, K_rl=8, M_rl=8):
    m.eval(); nll, en, cf = [], [], []
    for ci, c in enumerate(held[:32]):
        xo, so, _ = label_to_scaffold(clamp_ball(c["xin"], c["R"]), c["sin"], c["R"])
        bnd, sb, R, n = c["bnd"], c["sbnd"], c["R"], c["n"]
        a = fixed_ball_scaffold(n, R, dev, xo.dtype, ordering=m.scaffold_order)
        if n < K_rl + 6:
            continue
        eg = torch.Generator(device=dev).manual_seed(1000 + ci)
        sd = int(torch.randint(n, (), generator=eg, device=dev)); blk = torch.zeros(n, dtype=torch.bool, device=dev)
        blk[(a - a[sd]).norm(dim=-1).topk(K_rl, largest=False).indices] = True
        nll.append(float(-m.block_log_prob_b(xo[None], so[None], blk, bnd, sb, R)[0] / K_rl))
        Xn, Sn, _ = m.sample_block_b(xo[None].expand(M_rl, n, 3).clone(), so[None].expand(M_rl, n).clone(),
                                     blk, bnd, sb, R, gen=eg)
        en.append(float(capped_energy(torch.cat([Xn, bnd[None].expand(M_rl, -1, 3)], 1),
                                      torch.cat([Sn, sb[None].expand(M_rl, -1)], 1), cap).mean()))
        bi = blk.nonzero().squeeze(1)
        cf.append(st.mean([clash_frac(Xn[k], Sn[k], bi, bnd, sb) for k in range(M_rl)]))
    m.train()
    return st.mean(nll), st.mean(en), st.mean(cf)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=8000); ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--m-rl", type=int, default=8); ap.add_argument("--k-rl", type=int, nargs="+", default=[4, 8])
    ap.add_argument("--lam", type=float, default=1.0); ap.add_argument("--cap", type=float, default=5.0)
    ap.add_argument("--lr", type=float, default=5e-5); ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--patience", type=int, default=12)
    ap.add_argument("--radii", type=float, nargs="+", default=[2.0, 2.5, 3.0]); ap.add_argument("--per-frame", type=int, default=60)
    ap.add_argument("--use-demand", action="store_true")
    ap.add_argument("--warm", default=f"{ART}/ka3d_cavity_ebm3ax_rho115_blockcond_knn24_best.pt")
    ap.add_argument("--out", default=f"{ART}/ka3d_cavity_ebm3ax_rho115_rl_knn24.pt")
    ap.add_argument("--device", default=dev0); a = ap.parse_args(); dev = a.device
    torch.manual_seed(0); gen = torch.Generator(device=dev).manual_seed(1)
    dt = torch.load(f"{ART}/ka3d_train_N4096_T0.5_rho1.15.pt", map_location=dev, weights_only=False)
    Xt, St, L = dt["x"].to(dev).float(), dt["s"].to(dev).long(), float(dt["L"])
    dh = torch.load(f"{ART}/ka3d_dataset_N4096_T0.5_rho1.15.pt", map_location=dev, weights_only=False)
    train = build_pool(Xt, St, L, range(Xt.shape[0]), a.radii, 2.5, gen, per=a.per_frame)
    held = build_pool(dh["x"].to(dev).float(), dh["s"].to(dev).long(), L, range(16), a.radii, 2.5, gen, per=8)
    print(f"RL(free-run energy) FT: train={len(train)} held={len(held)} lam={a.lam} cap={a.cap} M_rl={a.m_rl} "
          f"K_rl={tuple(a.k_rl)} use_demand={a.use_demand}", flush=True)
    m = KA3DScaffoldEBMBatched(cat_bins=128, cat_range=2.5, knn_pot=24, use_demand=a.use_demand).to(dev)
    ck = torch.load(a.warm, map_location=dev, weights_only=False)
    miss, _ = m.load_state_dict(ck["state_dict"], strict=False); m.use_frame = False
    print(f"warm {a.warm}: {len(miss)} new", flush=True)
    opt = torch.optim.AdamW(m.parameters(), lr=a.lr, weight_decay=1e-4)
    t0 = time.time(); m.train(); best = float("inf"); stale = 0
    n0, e0, c0 = evaluate(m, held, dev, a.cap, a.k_rl[-1], a.m_rl)
    print(f"warm-start: blob{a.k_rl[-1]} NLL {n0:+.4f}  free-run capE {e0:.3f}  clash/p {c0:.3f}", flush=True)
    for step in range(a.steps + 1):
        ids = torch.randint(len(train), (a.batch,), generator=gen, device=dev)
        opt.zero_grad(); tf_acc = rl_acc = 0.0
        for i in ids:                                     # backward PER CONFIG (grad accum) -> ~1 graph live
            c = train[int(i)]; Q = rand_rot(gen, dev); n = c["n"]
            xo, so, _ = label_to_scaffold(clamp_ball(c["xin"] @ Q.T, c["R"]), c["sin"], c["R"])
            bnd = c["bnd"] @ Q.T; sb = c["sbnd"]; R = c["R"]
            a_sc = fixed_ball_scaffold(n, R, dev, xo.dtype, ordering=m.scaffold_order)
            mk = block_mask(draw_kind(gen, dev), n, R, a_sc, gen, dev)                            # TF anchor
            loss_tf = -m.block_log_prob_b(xo[None], so[None], mk, bnd, sb, R)[0] / int(mk.sum())
            K = int(a.k_rl[torch.randint(len(a.k_rl), (), generator=gen, device=dev)])            # RL blob
            loss_rl = xo.new_zeros(())
            if n >= K + 4:
                seed = int(torch.randint(n, (), generator=gen, device=dev)); blk = torch.zeros(n, dtype=torch.bool, device=dev)
                blk[(a_sc - a_sc[seed]).norm(dim=-1).topk(K, largest=False).indices] = True
                with torch.no_grad():
                    Xn, Sn, _ = m.sample_block_b(xo[None].expand(a.m_rl, n, 3).clone(),
                                                 so[None].expand(a.m_rl, n).clone(), blk, bnd, sb, R, gen=gen)
                    rew = capped_energy(torch.cat([Xn, bnd[None].expand(a.m_rl, -1, 3)], 1),
                                        torch.cat([Sn, sb[None].expand(a.m_rl, -1)], 1), a.cap)
                    adv = (rew - rew.mean()) / (rew.std() + 1e-6)
                logq = m.block_log_prob_b(Xn, Sn, blk, bnd, sb, R)                                # [M_rl] grad
                loss_rl = (adv * logq).mean()
            ((loss_tf + a.lam * loss_rl) / a.batch).backward()
            tf_acc += float(loss_tf) / a.batch; rl_acc += float(loss_rl) / a.batch
        torch.nn.utils.clip_grad_norm_(m.parameters(), 5.0); opt.step()
        if step % a.eval_every == 0:
            nll, en, cf = evaluate(m, held, dev, a.cap, a.k_rl[-1], a.m_rl)
            print(f"step {step:5d} tf {tf_acc:+.3f} rl {rl_acc:+.3f} | "
                  f"blob NLL {nll:+.4f} free-run capE {en:.3f}({en-e0:+.3f}) clash/p {cf:.3f}({cf-c0:+.3f}) "
                  f"({time.time()-t0:.0f}s)", flush=True)
            torch.save({"state_dict": m.state_dict(), "step": step, "capE": en, "clash": cf,
                        "knn_pot": 24, "use_demand": a.use_demand, "cat_bins": 128, "cat_range": 2.5}, a.out)
            if en < best - 1e-3:
                best = en; stale = 0
                torch.save({"state_dict": m.state_dict(), "step": step, "capE": en, "clash": cf, "knn_pot": 24,
                            "use_demand": a.use_demand, "cat_bins": 128, "cat_range": 2.5}, a.out.replace(".pt", "_best.pt"))
            else:
                stale += 1
                if stale >= a.patience:
                    print(f"EARLY STOP step {step}: free-run capE flat {a.patience} evals (best {best:.3f})", flush=True)
                    break
    print(f"saved {a.out} (best free-run capE {best:.3f})", flush=True)


if __name__ == "__main__":
    main()
