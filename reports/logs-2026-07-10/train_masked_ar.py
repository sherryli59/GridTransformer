"""Masked any-subset training for the fixed-scaffold AR: regenerate a RANDOM block given the rest
(full-cage). One model serves every k: k=n is the one-shot warm start, small k are the exact
block-MH mutation. Loss = -log q(block | retained + boundary)/k with a log-uniform block size (plus
explicit full-generation). The proven bottleneck (block_mh_demo: 0% MH acceptance with the OOD
full-order model) is exactly what full-cage training fixes.

Free clusters first (no boundary) as the architecture gate; boundary added once this passes.
Key eval metrics: held block-nll, block-conditional clash (should ->0), and from-TRUE block-MH
acceptance (should -> high; was 0% for the OOD model).
"""
from __future__ import annotations
import argparse, math, statistics as st, time
from pathlib import Path
import torch
from liquid_coupling_flow.ka3d_cavity_ar import _mic
from liquid_coupling_flow.ka3d_cavity_carve import carve
from liquid_coupling_flow.ka3d_scaffold_ar import (KA3DScaffoldAR, KA3DScaffoldCatAR, label_to_scaffold,
                                                   fixed_ball_scaffold, ball_unsquash, ball_squash)
from liquid_coupling_flow.ka3d_block import block_log_prob, sample_block, block_mh_step
from liquid_coupling_flow.ka_energy import ka_energy


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=Path, default=Path("liquid_coupling_flow/artifacts/ka3d_dataset_N512_T0.5.pt"))
    p.add_argument("--artifact", type=Path, default=Path("liquid_coupling_flow/artifacts/ka3d_masked_ar.pt"))
    p.add_argument("--steps", type=int, default=20000)
    p.add_argument("--eval-every", type=int, default=1000)
    p.add_argument("--batch", type=int, default=6)
    p.add_argument("--train-frames", type=int, default=900)
    p.add_argument("--radii", type=float, nargs="+", default=[1.6, 2.0, 2.4])
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--flow-bins", type=int, default=12,
                   help="RQ-spline knots per axis; 12 hit the resolution floor (width 0.41 vs cage 0.15), mw fix=32")
    p.add_argument("--flow-tail", type=float, default=5.0,
                   help="spline tail_bound; must match the soft-map target (|u| 99%=1.87) -> 2.75, not 5")
    p.add_argument("--head", choices=["spline", "categorical"], default="spline",
                   help="categorical = 2D-proven sharp peak (fixes the broad-mode clash)")
    p.add_argument("--cat-bins", type=int, default=128); p.add_argument("--cat-range", type=float, default=2.5)
    p.add_argument("--no-frame", dest="frame", action="store_false", default=True,
                   help="drop the Gram-Schmidt local frame -> global orientation (needs --rot-aug); 2D recipe")
    p.add_argument("--rot-aug", action="store_true", help="random 3D rotation + relabel each step (for --no-frame)")
    p.add_argument("--d-model", type=int, default=128); p.add_argument("--n-head", type=int, default=4)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def build_pool(X, S, L, frames, radii, gen, per=2):
    pool = []
    for f in frames:
        for _ in range(per):
            center = torch.rand(3, generator=gen, device=X.device) * L
            R = float(radii[int(torch.randint(len(radii), (), generator=gen, device=X.device))])
            c = carve(X[f], S[f], center, R, L)
            if c["n_in"] < 6:
                continue
            x, s, _ = label_to_scaffold(_mic(c["x_in"], center, L), c["s_in"], R)
            pool.append({"x": x, "s": s, "R": R, "n": x.shape[0]})
    return pool


def rand_blob(n, R, gen, dev):
    """Spatially-contiguous block: the k anchors nearest a random seed anchor (k=n => warm start)."""
    if torch.rand((), generator=gen, device=dev) < 0.15:
        k = n                                                    # full generation (warm start)
    else:
        k = int(round(math.exp(torch.rand((), generator=gen, device=dev).item() * math.log(max(n, 2)))))
        k = max(1, min(n, k))
    anchors = fixed_ball_scaffold(n, R, dev)
    seed = int(torch.randint(n, (), generator=gen, device=dev))
    idx = (anchors - anchors[seed]).norm(dim=-1).topk(k, largest=False).indices
    m = torch.zeros(n, dtype=torch.bool, device=dev)
    m[idx] = True
    return m


def rand_rotation(gen, dev):
    A = torch.randn(3, 3, generator=gen, device=dev)
    Q, Rm = torch.linalg.qr(A)
    Q = Q * torch.sign(torch.diagonal(Rm))[None]
    if torch.det(Q) < 0:
        Q[:, 0] = -Q[:, 0]
    return Q


def blob_mask(n, R, k, gen, dev):
    """Fixed-size k-blob: k anchors nearest a random seed."""
    anchors = fixed_ball_scaffold(n, R, dev)
    seed = int(torch.randint(n, (), generator=gen, device=dev))
    idx = (anchors - anchors[seed]).norm(dim=-1).topk(min(k, n), largest=False).indices
    m = torch.zeros(n, dtype=torch.bool, device=dev); m[idx] = True
    return m


def energy_total(xo, so):
    return ka_energy(xo[None], so.long()[None], 100.0)[0]


@torch.no_grad()
def evaluate(model, held, gen, dev):
    model.eval()
    empty_x, empty_s = torch.zeros(0, 3, device=dev), torch.zeros(0, dtype=torch.long, device=dev)
    bnll, bclash, full_u, acc_list = [], [], [], []
    for pair in held:
        x, s, R, n = pair["x"], pair["s"], pair["R"], pair["n"]
        k = min(4, n)
        blk = blob_mask(n, R, k, gen, dev)
        bnll.append(float(-block_log_prob(model, x, s, blk, empty_x, empty_s, R) / k))
        xn, sn, _ = sample_block(model, x, s, blk, empty_x, empty_s, R, gen=gen)
        bi = xn[blk]                                             # regenerated block particles
        d = torch.cdist(bi, xn)                                  # [k,n] block-particle -> all particles
        d[torch.arange(bi.shape[0]), torch.nonzero(blk).squeeze(1)] = 1e9   # exclude self
        bclash.append(float((d.min(1).values < 0.8).float().mean()))
        n_B = int((s == 1).sum())
        fx, fs = model.sample_pair(empty_x, empty_s, n - n_B, n_B, R)
        full_u.append(float(energy_total(fx, fs)) / n)
        # from-true MH acceptance (k=4, 3 sweeps)
        xx, ss = x.clone(), s.clone(); acc = mv = 0
        for _ in range(3):
            for _ in range(max(1, n // 4)):
                b = blob_mask(n, R, 4, gen, dev)
                xx, ss, a, _ = block_mh_step(model, xx, ss, b, empty_x, empty_s, R, 2.0, energy_total, gen)
                acc += int(a); mv += 1
        acc_list.append(acc / max(mv, 1))
    # teacher-forced MODE error (|argmax placement - true|); the sharpness metric we track
    merr = []
    for pair in held[:8]:
        x, s, R, n = pair["x"], pair["s"], pair["R"], pair["n"]
        anch = fixed_ball_scaffold(n, R, dev); ay, _ = ball_unsquash(anch, R)
        kind = torch.zeros(n, dtype=torch.long, device=dev); idx = torch.arange(n, device=dev)
        for jj in range(2, n):
            h, frame = model._frame_context(x, s, kind, (idx < jj)[None], anch[jj:jj+1], anch[jj:jj+1],
                                            slot_feat=model._slot_features(anch[jj:jj+1], jj, n, R))
            u = model.flow.mode(h + model.sp_out_emb(s[jj:jj+1]))
            y = ay[jj] + torch.einsum("naj,na->nj", frame, u)[0]
            merr.append(float((ball_squash(y, R)[0] - x[jj]).norm()))
    model.train()
    return {"block_nll": st.mean(bnll), "block_clash": st.mean(bclash), "mode_err": st.median(merr),
            "full_U_median": st.median(full_u), "mh_accept": st.mean(acc_list)}


def main():
    a = parse(); dev = torch.device(a.device)
    torch.manual_seed(0); gen = torch.Generator(device=dev).manual_seed(1)
    d = torch.load(a.dataset, map_location=dev, weights_only=False)
    X, S, L = d["x"].float(), d["s"].long(), float(d["L"])
    split = min(a.train_frames, X.shape[0] - 1)
    train = build_pool(X, S, L, range(split), a.radii, gen)
    held = build_pool(X, S, L, range(split, X.shape[0]), a.radii, gen, per=1)[:20]
    print(f"masked train={len(train)} held={len(held)} radii={tuple(a.radii)} dev={dev}", flush=True)
    if a.head == "categorical":
        model = KA3DScaffoldCatAR(cat_bins=a.cat_bins, cat_range=a.cat_range, d_model=a.d_model, n_head=a.n_head).to(dev)
    else:
        model = KA3DScaffoldAR(flow_bins=a.flow_bins, flow_tail=a.flow_tail, d_model=a.d_model, n_head=a.n_head).to(dev)
    model.use_frame = a.frame
    print(f"head={a.head} use_frame={a.frame} rot_aug={a.rot_aug}", flush=True)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    empty_x, empty_s = X.new_empty(0, 3), S.new_empty(0)
    start, series, best = 0, [], math.inf
    if a.resume and a.artifact.exists():
        st8 = torch.load(a.artifact, map_location=dev, weights_only=False)
        model.load_state_dict(st8["state_dict"]); opt.load_state_dict(st8["optimizer"])
        start = int(st8["step"]) + 1; series = st8.get("series", []); best = float(st8.get("best", math.inf))
        print(f"resumed step={start}", flush=True)
    a.artifact.parent.mkdir(parents=True, exist_ok=True); t0 = time.time()
    for step in range(start, a.steps + 1):
        ids = torch.randint(len(train), (a.batch,), generator=gen, device=dev)
        loss = X.new_zeros(())
        for i in ids:
            pr = train[int(i)]
            x, s = pr["x"], pr["s"]
            if a.rot_aug:                                          # rotate + relabel (learn rotation-equivariance)
                x, s, _ = label_to_scaffold(x @ rand_rotation(gen, dev).T, s, pr["R"])
            blk = rand_blob(pr["n"], pr["R"], gen, dev)
            loss = loss - block_log_prob(model, x, s, blk, empty_x, empty_s, pr["R"]) / int(blk.sum())
        loss = loss / a.batch
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()
        if step % a.eval_every == 0 or step == a.steps:
            m = evaluate(model, held, gen, dev); m["step"] = step; series.append(m)
            print(f"step {step:6d} block_nll={m['block_nll']:+.3f} mode_err={m['mode_err']:.3f} "
                  f"block_clash={100*m['block_clash']:.1f}% mh_accept={100*m['mh_accept']:.1f}% "
                  f"full_U={m['full_U_median']:+.2f} ({time.time()-t0:.0f}s)", flush=True)
            state = {"state_dict": model.state_dict(), "optimizer": opt.state_dict(), "step": step,
                     "series": series, "best": min(best, m["block_nll"]), "config": vars(a)}
            torch.save(state, a.artifact)
            if m["block_nll"] < best:
                best = m["block_nll"]; torch.save(state, a.artifact.with_name(a.artifact.stem + "_best.pt"))


if __name__ == "__main__":
    main()
