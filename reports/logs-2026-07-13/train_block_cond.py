"""BLOCK-CONDITIONAL FINE-TUNE (#1): make the AR model good at arbitrary-block conditionals, not just the
full Morton sequence. MEASURED motivation: the zero-shot blob conditional is ~3x worse than the byte-exact
trained Morton-suffix control (diag_blob_ood: internal clash 0.133 vs 0.043, dE 2004 vs 693), because
training only ever optimized the full-sequence NLL so q(block|retained,boundary) is an untrained
recombination of prefix conditionals. Fix: train -log q(block | retained + boundary) for a MIXTURE of the
exact block types the SMC/MTM kernels use, so every mutation is in-distribution.

SINGLE CHANGE (attribution discipline): warm-start the knn=8 baseline; NO reservation tilt, NO relaxed
targets in this run. allmask is IN the mixture and equals the original full-AR NLL exactly (verified
diff 0.0) -> the base objective is protected as one mixture component, not bolted on.

Mixture (per config): allmask 0.30 (base protect + full-regen move), random blob 0.45 (K~U[2,Kmax], the
MTM/SMC block), single-site 0.15 (K=1 heat-bath), Morton-tail 0.10 (K~U[2,Kmax], the free in-dist move).
Loss = -block_log_prob / n_block (per-particle scale, balanced across block sizes). block_log_prob is the
exact differentiable density used by eval; MLE needs only the score path (no sampling -> roundtrip
irrelevant). Rotation aug per config (matches base trainer). Checkpoints every 500 (durability); best on
held BLOB-K8 NLL (the target), with full-AR NLL printed every eval to guard against base regression."""
import argparse, time, statistics as st
from pathlib import Path
import torch
from liquid_coupling_flow.ka3d_scaffold_ebm import KA3DScaffoldEBM
from liquid_coupling_flow.ka3d_scaffold_ar import label_to_scaffold, fixed_ball_scaffold
from liquid_coupling_flow.ka3d_cavity_ar import _mic
from liquid_coupling_flow.ka3d_cavity_carve import carve

ART = "liquid_coupling_flow/artifacts"


def clamp_ball(x, R, eps=1e-5):
    n = x.norm(dim=-1, keepdim=True)
    return x * (R * (1.0 - eps) / n.clamp_min(1e-12)).clamp(max=1.0)


def rand_rot(gen, dev):
    A = torch.randn(3, 3, generator=gen, device=dev); Q, Rm = torch.linalg.qr(A)
    Q = Q * torch.sign(torch.diagonal(Rm))[None]
    if torch.det(Q) < 0:
        Q[:, 0] = -Q[:, 0]
    return Q


def build_pool(X, S, L, frames, radii, r_ctx, gen, per=60, nmin=8):
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


def block_mask(kind, n, R, anchors, gen, dev, Kmax=16):
    """Construct a block mask of the given kind. anchors [n,3] = scaffold anchors (morton order)."""
    mk = torch.zeros(n, dtype=torch.bool, device=dev)
    Kc = min(Kmax, n - 1)
    if kind == "all":
        mk[:] = True
    elif kind == "single":
        mk[int(torch.randint(n, (), generator=gen, device=dev))] = True
    elif kind == "tail":
        K = int(torch.randint(2, Kc + 1, (), generator=gen, device=dev))
        mk[n - K:] = True
    else:  # blob: K nearest anchors to a random seed
        K = int(torch.randint(2, Kc + 1, (), generator=gen, device=dev))
        seed = int(torch.randint(n, (), generator=gen, device=dev))
        mk[(anchors - anchors[seed]).norm(dim=-1).topk(K, largest=False).indices] = True
    return mk


KINDS = ["all", "blob", "single", "tail"]
WEIGHTS = torch.tensor([0.30, 0.45, 0.15, 0.10])


def draw_kind(gen, dev):
    return KINDS[int(torch.multinomial(WEIGHTS.to(dev), 1, generator=gen))]


@torch.no_grad()
def evaluate(model, held, gen, dev):
    """Return dict of per-particle NLL for allmask (base), blob-K8, tail-K8, single. Deterministic blocks."""
    model.eval(); out = {"all": [], "blob8": [], "tail8": [], "single": []}
    for ci, c in enumerate(held[:64]):
        xo, so, _ = label_to_scaffold(clamp_ball(c["xin"], c["R"]), c["sin"], c["R"])
        bnd, sb, R, n = c["bnd"], c["sbnd"], c["R"], c["n"]
        a = fixed_ball_scaffold(n, R, dev, xo.dtype, ordering=model.scaffold_order)
        eg = torch.Generator(device=dev).manual_seed(1000 + ci)               # fixed blocks per cavity
        am = torch.ones(n, dtype=torch.bool, device=dev)
        out["all"].append(float(-model.block_log_prob(xo, so, am, bnd, sb, R) / n))
        if n > 9:
            sd = int(torch.randint(n, (), generator=eg, device=dev)); bl = torch.zeros(n, dtype=torch.bool, device=dev)
            bl[(a - a[sd]).norm(dim=-1).topk(8, largest=False).indices] = True
            out["blob8"].append(float(-model.block_log_prob(xo, so, bl, bnd, sb, R) / 8))
            tl = torch.zeros(n, dtype=torch.bool, device=dev); tl[n - 8:] = True
            out["tail8"].append(float(-model.block_log_prob(xo, so, tl, bnd, sb, R) / 8))
        sg = torch.zeros(n, dtype=torch.bool, device=dev); sg[int(torch.randint(n, (), generator=eg, device=dev))] = True
        out["single"].append(float(-model.block_log_prob(xo, so, sg, bnd, sb, R) / 1))
    model.train()
    return {k: st.mean(v) for k, v in out.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=30000); ap.add_argument("--batch", type=int, default=6)
    ap.add_argument("--patience", type=int, default=10); ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--radii", type=float, nargs="+", default=[1.6, 2.0, 2.5, 3.0])
    ap.add_argument("--r-ctx", type=float, default=2.5); ap.add_argument("--per-frame", type=int, default=60)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--warm", default=f"{ART}/ka3d_cavity_ebm3ax_rho115_best.pt")
    ap.add_argument("--out", default=f"{ART}/ka3d_cavity_ebm3ax_rho115_blockcond.pt")
    ap.add_argument("--knn-pot", type=int, default=8)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args(); dev = a.device
    torch.manual_seed(0); gen = torch.Generator(device=dev).manual_seed(1)
    dt = torch.load(f"{ART}/ka3d_train_N4096_T0.5_rho1.15.pt", map_location=dev, weights_only=False)
    Xt, St, L = dt["x"].to(dev).float(), dt["s"].to(dev).long(), float(dt["L"])
    dh = torch.load(f"{ART}/ka3d_dataset_N4096_T0.5_rho1.15.pt", map_location=dev, weights_only=False)
    Xh, Sh = dh["x"].to(dev).float(), dh["s"].to(dev).long()
    train = build_pool(Xt, St, L, range(Xt.shape[0]), a.radii, a.r_ctx, gen, per=a.per_frame)
    held = build_pool(Xh, Sh, L, range(16), a.radii, a.r_ctx, gen, per=8)
    print(f"block-cond FT: train={len(train)} held={len(held)} radii={tuple(a.radii)} "
          f"mixture={dict(zip(KINDS, WEIGHTS.tolist()))} dev={dev}", flush=True)
    m = KA3DScaffoldEBM(cat_bins=128, cat_range=2.5, knn_pot=a.knn_pot).to(dev)
    start = 0
    if a.resume and Path(a.out).exists():
        ck = torch.load(a.out, map_location=dev, weights_only=False)
        m.load_state_dict(ck["state_dict"], strict=False); start = int(ck.get("step", 0))
        print(f"RESUME {a.out} @ step {start}", flush=True)
    else:
        ck = torch.load(a.warm, map_location=dev, weights_only=False)
        m.load_state_dict(ck["state_dict"], strict=False)
        print(f"warm {a.warm}", flush=True)
    m.use_frame = False
    opt = torch.optim.AdamW(m.parameters(), lr=a.lr, weight_decay=1e-4)
    t0 = time.time(); m.train(); best = float("inf"); stale = 0
    base0 = evaluate(m, held, gen, dev)
    print(f"warm-start held NLL: all {base0['all']:+.4f}  blob8 {base0['blob8']:+.4f}  "
          f"tail8 {base0['tail8']:+.4f}  single {base0['single']:+.4f}", flush=True)
    for step in range(start, a.steps + 1):
        ids = torch.randint(len(train), (a.batch,), generator=gen, device=dev)
        loss = Xt.new_zeros(())
        for i in ids:
            c = train[int(i)]; Q = rand_rot(gen, dev); n = c["n"]
            xo, so, _ = label_to_scaffold(clamp_ball(c["xin"] @ Q.T, c["R"]), c["sin"], c["R"])
            bnd = c["bnd"] @ Q.T
            a_sc = fixed_ball_scaffold(n, c["R"], dev, xo.dtype, ordering=m.scaffold_order)
            mk = block_mask(draw_kind(gen, dev), n, c["R"], a_sc, gen, dev)
            loss = loss - m.block_log_prob(xo, so, mk, bnd, c["sbnd"], c["R"]) / int(mk.sum())
        loss = loss / a.batch
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 5.0); opt.step()
        if step % a.eval_every == 0:
            e = evaluate(m, held, gen, dev)
            dblob = e["blob8"] - base0["blob8"]; dall = e["all"] - base0["all"]
            print(f"step {step:5d} loss {loss.item():+.4f} | all {e['all']:+.4f}({dall:+.3f}) "
                  f"blob8 {e['blob8']:+.4f}({dblob:+.3f}) tail8 {e['tail8']:+.4f} single {e['single']:+.4f} "
                  f"({time.time()-t0:.0f}s)", flush=True)
            torch.save({"state_dict": m.state_dict(), "step": step, "held": e, "knn_pot": a.knn_pot,
                        "cat_bins": 128, "cat_range": 2.5}, a.out)
            if e["blob8"] < best - 1e-3:
                best = e["blob8"]; stale = 0
                torch.save({"state_dict": m.state_dict(), "step": step, "held": e, "knn_pot": a.knn_pot,
                            "cat_bins": 128, "cat_range": 2.5}, a.out.replace(".pt", "_best.pt"))
            else:
                stale += 1
                if stale >= a.patience:
                    print(f"EARLY STOP step {step}: blob8 NLL flat {a.patience} evals (best {best:+.4f})", flush=True)
                    break
    print(f"saved {a.out} (best blob8 NLL {best:+.4f})", flush=True)


if __name__ == "__main__":
    main()
