"""STAGE 1 of the capacity-conditioning plan: fine-tune with the remaining-demand embedding
(use_demand=True). Same mixed-block objective as the block-cond FT, but through the BATCHED
block_log_prob_b path so the demand feature is active in training (and identical to sampling => exact).
Warm-start the block-cond FT knn24 ckpt; the demand_emb is zero-init so warm-start == that ckpt exactly.
SINGLE new signal (attribution). Eval: blob8 NLL (target) + allmask NLL (base guard) + single. Best on
blob8. Reuses helpers from train_block_cond.py."""
import argparse, time, sys, statistics as st
from pathlib import Path
import torch
sys.path.insert(0, "reports/logs-2026-07-13")
from train_block_cond import build_pool, clamp_ball, rand_rot, block_mask, draw_kind, KINDS, WEIGHTS
from liquid_coupling_flow.ka3d_ebm_batched import KA3DScaffoldEBMBatched
from liquid_coupling_flow.ka3d_scaffold_ar import label_to_scaffold, fixed_ball_scaffold

ART = "liquid_coupling_flow/artifacts"


@torch.no_grad()
def evaluate(m, held, dev):
    m.eval(); out = {"all": [], "blob8": [], "single": []}
    for ci, c in enumerate(held[:64]):
        xo, so, _ = label_to_scaffold(clamp_ball(c["xin"], c["R"]), c["sin"], c["R"])
        bnd, sb, R, n = c["bnd"], c["sbnd"], c["R"], c["n"]
        a = fixed_ball_scaffold(n, R, dev, xo.dtype, ordering=m.scaffold_order)
        eg = torch.Generator(device=dev).manual_seed(1000 + ci)
        am = torch.ones(n, dtype=torch.bool, device=dev)
        out["all"].append(float(-m.block_log_prob_b(xo[None], so[None], am, bnd, sb, R)[0] / n))
        if n > 9:
            sd = int(torch.randint(n, (), generator=eg, device=dev)); bl = torch.zeros(n, dtype=torch.bool, device=dev)
            bl[(a - a[sd]).norm(dim=-1).topk(8, largest=False).indices] = True
            out["blob8"].append(float(-m.block_log_prob_b(xo[None], so[None], bl, bnd, sb, R)[0] / 8))
        sg = torch.zeros(n, dtype=torch.bool, device=dev); sg[int(torch.randint(n, (), generator=eg, device=dev))] = True
        out["single"].append(float(-m.block_log_prob_b(xo[None], so[None], sg, bnd, sb, R)[0] / 1))
    m.train()
    return {k: st.mean(v) for k, v in out.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=20000); ap.add_argument("--batch", type=int, default=6)
    ap.add_argument("--patience", type=int, default=10); ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--radii", type=float, nargs="+", default=[1.6, 2.0, 2.5, 3.0])
    ap.add_argument("--r-ctx", type=float, default=2.5); ap.add_argument("--per-frame", type=int, default=60)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--warm", default=f"{ART}/ka3d_cavity_ebm3ax_rho115_blockcond_knn24_best.pt")
    ap.add_argument("--out", default=f"{ART}/ka3d_cavity_ebm3ax_rho115_demand_knn24.pt")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args(); dev = a.device
    torch.manual_seed(0); gen = torch.Generator(device=dev).manual_seed(1)
    dt = torch.load(f"{ART}/ka3d_train_N4096_T0.5_rho1.15.pt", map_location=dev, weights_only=False)
    Xt, St, L = dt["x"].to(dev).float(), dt["s"].to(dev).long(), float(dt["L"])
    dh = torch.load(f"{ART}/ka3d_dataset_N4096_T0.5_rho1.15.pt", map_location=dev, weights_only=False)
    Xh, Sh = dh["x"].to(dev).float(), dh["s"].to(dev).long()
    train = build_pool(Xt, St, L, range(Xt.shape[0]), a.radii, a.r_ctx, gen, per=a.per_frame)
    held = build_pool(Xh, Sh, L, range(16), a.radii, a.r_ctx, gen, per=8)
    print(f"STAGE1 demand FT: train={len(train)} held={len(held)} mixture={dict(zip(KINDS, WEIGHTS.tolist()))}", flush=True)
    m = KA3DScaffoldEBMBatched(cat_bins=128, cat_range=2.5, knn_pot=24, use_demand=True).to(dev)
    start = 0
    if a.resume and Path(a.out).exists():
        ck = torch.load(a.out, map_location=dev, weights_only=False)
        m.load_state_dict(ck["state_dict"], strict=False); start = int(ck.get("step", 0))
        print(f"RESUME {a.out} @ step {start}", flush=True)
    else:
        ck = torch.load(a.warm, map_location=dev, weights_only=False)
        miss, unexp = m.load_state_dict(ck["state_dict"], strict=False)
        print(f"warm {a.warm}: {len(miss)} new (demand_emb), {len(unexp)} unused", flush=True)
    m.use_frame = False
    opt = torch.optim.AdamW(m.parameters(), lr=a.lr, weight_decay=1e-4)
    t0 = time.time(); m.train(); best = float("inf"); stale = 0
    base0 = evaluate(m, held, dev)
    print(f"warm-start (demand zero-init): all {base0['all']:+.4f} blob8 {base0['blob8']:+.4f} single {base0['single']:+.4f}", flush=True)
    for step in range(start, a.steps + 1):
        ids = torch.randint(len(train), (a.batch,), generator=gen, device=dev)
        loss = Xt.new_zeros(())
        for i in ids:
            c = train[int(i)]; Q = rand_rot(gen, dev); n = c["n"]
            xo, so, _ = label_to_scaffold(clamp_ball(c["xin"] @ Q.T, c["R"]), c["sin"], c["R"])
            bnd = c["bnd"] @ Q.T
            a_sc = fixed_ball_scaffold(n, c["R"], dev, xo.dtype, ordering=m.scaffold_order)
            mk = block_mask(draw_kind(gen, dev), n, c["R"], a_sc, gen, dev)
            loss = loss - m.block_log_prob_b(xo[None], so[None], mk, bnd, c["sbnd"], c["R"])[0] / int(mk.sum())
        loss = loss / a.batch
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 5.0); opt.step()
        if step % a.eval_every == 0:
            e = evaluate(m, held, dev)
            print(f"step {step:5d} loss {loss.item():+.4f} | all {e['all']:+.4f}({e['all']-base0['all']:+.3f}) "
                  f"blob8 {e['blob8']:+.4f}({e['blob8']-base0['blob8']:+.3f}) single {e['single']:+.4f} "
                  f"({time.time()-t0:.0f}s)", flush=True)
            torch.save({"state_dict": m.state_dict(), "step": step, "held": e, "knn_pot": 24,
                        "use_demand": True, "cat_bins": 128, "cat_range": 2.5}, a.out)
            if e["blob8"] < best - 1e-3:
                best = e["blob8"]; stale = 0
                torch.save({"state_dict": m.state_dict(), "step": step, "held": e, "knn_pot": 24,
                            "use_demand": True, "cat_bins": 128, "cat_range": 2.5}, a.out.replace(".pt", "_best.pt"))
            else:
                stale += 1
                if stale >= a.patience:
                    print(f"EARLY STOP step {step}: blob8 flat {a.patience} evals (best {best:+.4f})", flush=True)
                    break
    print(f"saved {a.out} (best blob8 NLL {best:+.4f})", flush=True)


if __name__ == "__main__":
    main()
