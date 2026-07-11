"""Train the 3D learned-potential cavity infiller (KA3DScaffoldEBM) on carved cavities WITH a frozen
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
def evaluate(model, held, gen, dev, beta=2.0):
    model.eval(); nll, acc = [], []
    for c in held[:16]:
        xo, so, _ = label_to_scaffold(c["xin"], c["sin"], c["R"])
        bnd, s_bnd, R, n = c["bnd"], c["sbnd"], c["R"], c["n"]
        nll.append(float(-model.log_prob_pair(xo, so, bnd, s_bnd, R, preordered=True) / n))
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
    return st.mean(nll), 100 * sum(acc) / len(acc)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=8000); ap.add_argument("--batch", type=int, default=6)
    ap.add_argument("--lr", type=float, default=1.5e-4); ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--radii", type=float, nargs="+", default=[1.6, 2.0, 2.4])  # 2R+r_cut<=L=7.53 -> R<=~2.5
    ap.add_argument("--r-ctx", type=float, default=2.5); ap.add_argument("--train-frames", type=int, default=900)
    ap.add_argument("--freeze-phi", action="store_true",
                    help="keep the potential at zero (base cat-head + boundary + R_embed) -> fair ablation baseline")
    ap.add_argument("--warm", default=f"{ART}/ka3d_blob_noframe.pt")
    ap.add_argument("--out", default=f"{ART}/ka3d_cavity_ebm.pt")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args(); dev = a.device
    torch.manual_seed(0); gen = torch.Generator(device=dev).manual_seed(1)
    d = torch.load(f"{ART}/ka3d_dataset_N512_T0.5.pt", map_location=dev, weights_only=False)
    X, S, L = d["x"].to(dev).float(), d["s"].to(dev).long(), float(d["L"])
    sp = min(a.train_frames, X.shape[0] - 1)
    train = build_pool(X, S, L, range(sp), a.radii, a.r_ctx, gen)
    held = build_pool(X, S, L, range(sp, X.shape[0]), a.radii, a.r_ctx, gen, per=1)
    print(f"3D cavity-EBM: train={len(train)} held={len(held)} radii={tuple(a.radii)} dev={dev}", flush=True)
    ck = torch.load(a.warm, map_location=dev, weights_only=False)
    m = KA3DScaffoldEBM(cat_bins=128, cat_range=2.5).to(dev)
    miss, unexp = m.load_state_dict(ck["state_dict"], strict=False); m.use_frame = False
    print(f"warm {a.warm}: {len(miss)} new, {len(unexp)} unused; frameless", flush=True)
    if a.freeze_phi:
        params = [p for name, p in m.named_parameters() if not name.startswith("phi.")]
        print(f"FREEZE-PHI baseline: potential stays 0 (V_c==0); training {len(params)} param tensors", flush=True)
    else:
        params = list(m.parameters())
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=1e-4)
    t0 = time.time(); m.train()
    for step in range(a.steps + 1):
        ids = torch.randint(len(train), (a.batch,), generator=gen, device=dev)
        loss = X.new_zeros(())
        for i in ids:
            c = train[int(i)]; Q = rand_rot(gen, dev)
            xo, so, _ = label_to_scaffold(c["xin"] @ Q.T, c["sin"], c["R"])
            bnd = c["bnd"] @ Q.T
            loss = loss - m.log_prob_pair(xo, so, bnd, c["sbnd"], c["R"], preordered=True) / c["n"]
        loss = loss / a.batch
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 5.0); opt.step()
        if step % a.eval_every == 0:
            nll, acc = evaluate(m, held, gen, dev)
            print(f"step {step:5d} loss {loss.item():+.4f}  held_nll {nll:+.4f}  MTM(k4,N16) {acc:.0f}%  "
                  f"({time.time()-t0:.0f}s)", flush=True)
            torch.save({"state_dict": m.state_dict(), "step": step, "radii": a.radii, "r_ctx": a.r_ctx,
                        "cat_bins": 128, "cat_range": 2.5}, a.out)
    print(f"saved {a.out}", flush=True)


if __name__ == "__main__":
    main()
