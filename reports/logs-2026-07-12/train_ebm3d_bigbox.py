"""BIG-BOX RETRAIN of the 3D learned-potential cavity infiller at the NEW home state point (N=4096
Bapst-derived reference, T=0.5 rho=1.1486): the foundation model for NN-augmented cavity sampling -- PT
seed generation (basin diversity), K-block MTM proposals, and the exact-log_q cavity free-energy program.
Warm-started from the rho=1.2 ebm3ax (small density shift, fine-tune adapts); radii extended into the
DECAY regime (R<=3.0) which the old box could not host. Original: 07-11/train_ebm3d_cavity.py.
Trains on carved cavities WITH a frozen
boundary shell. Warm-start the frameless free-cluster cat head; the boundary stream + potential + R_embed
learn during fine-tune. Multi-radius so R generalizes. Loss = -log_prob_pair/n (block-conditional MLE).
Checkpoints every 500 steps (durability). Eval = block-MTM acceptance + NLL on held cavities."""
import argparse, math, time, statistics as st
from pathlib import Path
import torch
from liquid_coupling_flow.ka3d_scaffold_ebm import KA3DScaffoldEBM
from liquid_coupling_flow.ka3d_scaffold_ar import label_to_scaffold, fixed_ball_scaffold
from liquid_coupling_flow.ka3d_cavity_ar import _mic
from liquid_coupling_flow.ka3d_cavity_carve import carve
from liquid_coupling_flow.ka_energy import ka_energy

ART = "liquid_coupling_flow/artifacts"


def clamp_ball(x, R, eps=1e-5):
    """Radially clamp any point with |x|>=R into the OPEN ball (rotation matmul can nudge a boundary
    particle |x|~R just outside, which ball_unsquash rejects). Only touches near-boundary points."""
    n = x.norm(dim=-1, keepdim=True)
    return x * (R * (1.0 - eps) / n.clamp_min(1e-12)).clamp(max=1.0)


def rand_rot(gen, dev):
    A = torch.randn(3, 3, generator=gen, device=dev); Q, Rm = torch.linalg.qr(A)
    Q = Q * torch.sign(torch.diagonal(Rm))[None]
    if torch.det(Q) < 0:
        Q[:, 0] = -Q[:, 0]
    return Q


def build_pool(X, S, L, frames, radii, r_ctx, gen, per=3, nmin=8):
    pool = []
    for f in frames:
        for _ in range(per):
            c = torch.rand(3, generator=gen, device=X.device) * L
            R = float(radii[int(torch.randint(len(radii), (), generator=gen, device=X.device))])
            p = carve(X[f], S[f], c, R, L)
            if p["n_in"] < nmin:
                continue
            xin = _mic(p["x_in"], c, L); xout = _mic(p["x_out"], c, L)
            bm = xout.norm(dim=-1) < (R + r_ctx)
            pool.append({"xin": xin, "sin": p["s_in"], "bnd": xout[bm], "sbnd": p["s_out"][bm],
                         "R": R, "n": int(p["n_in"])})
    return pool


def energy_cav(xo, so, bnd, s_bnd):
    x = torch.cat([xo, bnd], 0); s = torch.cat([so, s_bnd], 0)
    return float(ka_energy(x[None], s.long()[None], 100.0)[0])              # big box: isolated cavity energy


def blob(n, R, K, gen, dev):
    a = fixed_ball_scaffold(n, R, dev); seed = int(torch.randint(n, (), generator=gen, device=dev))
    m = torch.zeros(n, dtype=torch.bool, device=dev)
    m[(a - a[seed]).norm(dim=-1).topk(min(K, n), largest=False).indices] = True
    return m


@torch.no_grad()
def evaluate(model, held, gen, dev, beta=2.0, skip_mtm=False):
    model.eval(); nll, acc = [], []
    for c in held[:16]:
        xo, so, _ = label_to_scaffold(clamp_ball(c["xin"], c["R"]), c["sin"], c["R"])
        bnd, s_bnd, R, n = c["bnd"], c["sbnd"], c["R"], c["n"]
        nll.append(float(-model.log_prob_pair(xo, so, bnd, s_bnd, R, preordered=True) / n))
        if skip_mtm:
            continue
        blk = blob(n, R, 4, gen, dev)
        u0 = energy_cav(xo, so, bnd, s_bnd); lqx = float(model.block_log_prob(xo, so, blk, bnd, s_bnd, R))
        lus = [-beta * u0 - lqx]
        for _ in range(16):
            xn, sn, lq = model.sample_block(xo, so, blk, bnd, s_bnd, R, gen=gen)
            lus.append(-beta * energy_cav(xn, sn, bnd, s_bnd) - float(lq))
        lu = torch.tensor(lus[1:], device=dev); sf = torch.logsumexp(lu, 0)
        js = int(torch.multinomial(torch.softmax(lu, 0), 1, generator=gen)); lr = lu.clone(); lr[js] = lus[0]
        acc.append(float(torch.rand((), device=dev, generator=gen).log() < (sf - torch.logsumexp(lr, 0))))
    model.train()
    if skip_mtm:
        return st.mean(nll), float("nan")
    return st.mean(nll), 100 * sum(acc) / len(acc)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=40000); ap.add_argument("--batch", type=int, default=6)
    ap.add_argument("--patience", type=int, default=8, help="stop after this many evals w/o held-NLL improvement")
    ap.add_argument("--lr", type=float, default=1.5e-4); ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--radii", type=float, nargs="+", default=[1.6, 2.0, 2.5, 3.0])  # big box: R<=6.4 possible; 3.0 caps step cost
    ap.add_argument("--r-ctx", type=float, default=2.5); ap.add_argument("--train-frames", type=int, default=44)
    ap.add_argument("--per-frame", type=int, default=60)
    ap.add_argument("--resume", action="store_true", help="continue from --out checkpoint (fresh optimizer)")
    ap.add_argument("--freeze-phi", action="store_true",
                    help="keep the potential at zero (base cat-head + boundary + R_embed) -> fair ablation baseline")
    ap.add_argument("--coarse", action="store_true",
                    help="use KA3DScaffoldEBMCoarse (coarse-cell-then-fine-bin position head) instead of the "
                         "sequential per-axis head; MTM eval metric is skipped (unbatched sample_block/"
                         "block_log_prob are NOT overridden for this head and would be inexact)")
    ap.add_argument("--warm", default=f"{ART}/ka3d_cavity_ebm3ax.pt")
    ap.add_argument("--out", default=f"{ART}/ka3d_cavity_ebm3ax_rho115.pt")
    ap.add_argument("--knn-pot", type=int, default=8, help="tilt/potential cage size (default 8; 24-32 = bigger local energy context)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args(); dev = a.device
    if a.coarse and a.out == f"{ART}/ka3d_cavity_ebm3ax_rho115.pt":
        a.out = f"{ART}/ka3d_cavity_coarse.pt"
    torch.manual_seed(0); gen = torch.Generator(device=dev).manual_seed(1)
    # CHAIN-LEVEL SPLIT, zero leakage: train = 112 Bapst train-split chains; held = the 16 test-split
    # chains behind the PTS reference dataset (model never sees the configs whose cavities we measure).
    dt = torch.load(f"{ART}/ka3d_train_N4096_T0.5_rho1.15.pt", map_location=dev, weights_only=False)
    Xt, St, L = dt["x"].to(dev).float(), dt["s"].to(dev).long(), float(dt["L"])
    dh = torch.load(f"{ART}/ka3d_dataset_N4096_T0.5_rho1.15.pt", map_location=dev, weights_only=False)
    Xh, Sh = dh["x"].to(dev).float(), dh["s"].to(dev).long()
    assert abs(float(dh["L"]) - L) < 1e-6
    train = build_pool(Xt, St, L, range(Xt.shape[0]), a.radii, a.r_ctx, gen, per=a.per_frame)
    held = build_pool(Xh, Sh, L, range(16), a.radii, a.r_ctx, gen, per=8)          # snapshot-1 configs only
    print(f"3D cavity-EBM: train={len(train)} held={len(held)} radii={tuple(a.radii)} dev={dev}", flush=True)
    if a.coarse:
        from liquid_coupling_flow.ka3d_coarse_head import KA3DScaffoldEBMCoarse
        m = KA3DScaffoldEBMCoarse(cat_bins=128, cat_range=2.5, knn_pot=a.knn_pot).to(dev)
    else:
        m = KA3DScaffoldEBM(cat_bins=128, cat_range=2.5, knn_pot=a.knn_pot).to(dev)
    print(f"tilt cage knn_pot = {m.knn_pot}", flush=True)
    start = 0
    if a.resume and Path(a.out).exists():
        ck = torch.load(a.out, map_location=dev, weights_only=False)
        m.load_state_dict(ck["state_dict"], strict=False); m.use_frame = False
        start = int(ck.get("step", 0))
        print(f"RESUME {a.out} @ step {start} (fresh optimizer)", flush=True)
    else:
        ck = torch.load(a.warm, map_location=dev, weights_only=False)
        miss, unexp = m.load_state_dict(ck["state_dict"], strict=False); m.use_frame = False
        print(f"warm {a.warm}: {len(miss)} new, {len(unexp)} unused; frameless", flush=True)
        if a.coarse and miss:
            print(f"  new params (up to 12): {list(miss)[:12]}", flush=True)
    if a.freeze_phi:
        params = [p for name, p in m.named_parameters() if name.split(".")[0] not in ("phi", "phi_a", "phi_b")]
        print(f"FREEZE-PHI baseline: potential stays 0 (V_c==0); training {len(params)} param tensors", flush=True)
    else:
        params = list(m.parameters())
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=1e-4)
    t0 = time.time(); m.train()
    best_nll = float("inf"); stale = 0
    for step in range(start, a.steps + 1):
        ids = torch.randint(len(train), (a.batch,), generator=gen, device=dev)
        loss = Xt.new_zeros(())
        for i in ids:
            c = train[int(i)]; Q = rand_rot(gen, dev)
            xo, so, _ = label_to_scaffold(clamp_ball(c["xin"] @ Q.T, c["R"]), c["sin"], c["R"])
            bnd = c["bnd"] @ Q.T
            loss = loss - m.log_prob_pair(xo, so, bnd, c["sbnd"], c["R"], preordered=True) / c["n"]
        loss = loss / a.batch
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 5.0); opt.step()
        if step % a.eval_every == 0:
            nll, acc = evaluate(m, held, gen, dev, skip_mtm=a.coarse)
            gap = loss.item() - nll                                       # train-vs-held gap (overfit monitor)
            print(f"step {step:5d} loss {loss.item():+.4f}  held_nll {nll:+.4f}  gap {gap:+.4f}  "
                  f"MTM(k4,N16) {acc:.0f}%  ({time.time()-t0:.0f}s)", flush=True)
            torch.save({"state_dict": m.state_dict(), "step": step, "radii": a.radii, "r_ctx": a.r_ctx,
                        "cat_bins": 128, "cat_range": 2.5, "held_nll": nll}, a.out)
            if nll < best_nll - 1e-3:
                best_nll = nll; stale = 0
                torch.save({"state_dict": m.state_dict(), "step": step, "radii": a.radii, "r_ctx": a.r_ctx,
                            "cat_bins": 128, "cat_range": 2.5, "held_nll": nll},
                           a.out.replace(".pt", "_best.pt"))
            else:
                stale += 1
                if stale >= a.patience:
                    print(f"EARLY STOP at step {step}: held_nll flat for {a.patience} evals "
                          f"(best {best_nll:+.4f})", flush=True)
                    break
    print(f"saved {a.out} (best held_nll {best_nll:+.4f} -> {a.out.replace('.pt', '_best.pt')})", flush=True)


if __name__ == "__main__":
    main()
