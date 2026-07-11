"""Block-conditional training WITH the frozen boundary (the real cavity target, not the free-cluster
marginal). Each block move regenerates a blob of interior particles conditioned on the frozen boundary
shell AND the retained interior. Tests the hypothesis that the dense wall sharpens the conditional
(free cluster plateaued at block_clash~45%, mode-err~0.45; 2D ref 0.12).

Key metrics vs the free-cluster run: block_clash (vs interior AND boundary), mode-err, and the
CORRECT independence-MTM K=4 acceptance (the actual value metric; single-try MH was 0%)."""
from __future__ import annotations
import argparse, math, statistics as st, time
from pathlib import Path
import torch
import torch.nn.functional as F
from liquid_coupling_flow.ka3d_cavity_ar import _mic
from liquid_coupling_flow.ka3d_cavity_carve import carve
from liquid_coupling_flow.ka3d_scaffold_ar import (KA3DScaffoldAR, label_to_scaffold, fixed_ball_scaffold,
                                                   ball_unsquash)
from liquid_coupling_flow.ka3d_block import block_log_prob, sample_block
from liquid_coupling_flow.ka_energy import ka_energy

BIGBOX = 100.0


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=Path, default=Path("liquid_coupling_flow/artifacts/ka3d_dataset_N512_T0.5.pt"))
    p.add_argument("--artifact", type=Path, default=Path("liquid_coupling_flow/artifacts/ka3d_cavity_block.pt"))
    p.add_argument("--steps", type=int, default=20000); p.add_argument("--eval-every", type=int, default=1000)
    p.add_argument("--batch", type=int, default=6); p.add_argument("--train-frames", type=int, default=900)
    p.add_argument("--radii", type=float, nargs="+", default=[1.6, 2.0, 2.4]); p.add_argument("--r-ctx", type=float, default=2.5)
    p.add_argument("--flow-bins", type=int, default=48); p.add_argument("--flow-tail", type=float, default=2.75)
    p.add_argument("--lr", type=float, default=3e-4); p.add_argument("--resume", action="store_true")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def build_pool(X, S, L, frames, radii, r_ctx, gen, per=2):
    pool = []
    for f in frames:
        for _ in range(per):
            center = torch.rand(3, generator=gen, device=X.device) * L
            R = float(radii[int(torch.randint(len(radii), (), generator=gen, device=X.device))])
            c = carve(X[f], S[f], center, R, L)
            if c["n_in"] < 6:
                continue
            x, s, _ = label_to_scaffold(_mic(c["x_in"], center, L), c["s_in"], R)
            ball = _mic(c["x_out"], center, L); shell = ball.norm(dim=-1) < R + r_ctx
            nB = int((s == 1).sum())
            pool.append({"x": x, "s": s, "R": R, "n": x.shape[0], "bnd": ball[shell], "s_bnd": c["s_out"][shell],
                         "nA": x.shape[0] - nB, "nB": nB})
    return pool


def rand_blob(n, R, gen, dev):
    if torch.rand((), generator=gen, device=dev) < 0.15:
        k = n
    else:
        k = max(1, min(n, int(round(math.exp(torch.rand((), generator=gen, device=dev).item() * math.log(max(n, 2)))))))
    a = fixed_ball_scaffold(n, R, dev); seed = int(torch.randint(n, (), generator=gen, device=dev))
    idx = (a - a[seed]).norm(dim=-1).topk(k, largest=False).indices
    m = torch.zeros(n, dtype=torch.bool, device=dev); m[idx] = True; return m


def blob_mask(n, R, k, gen, dev):
    a = fixed_ball_scaffold(n, R, dev); seed = int(torch.randint(n, (), generator=gen, device=dev))
    idx = (a - a[seed]).norm(dim=-1).topk(min(k, n), largest=False).indices
    m = torch.zeros(n, dtype=torch.bool, device=dev); m[idx] = True; return m


def cavity_energy(irel, s_int, bnd, s_bnd):
    x = torch.cat([irel, bnd]); ss = torch.cat([s_int, s_bnd])
    return ka_energy(x[None], ss.long()[None], BIGBOX)[0]


@torch.no_grad()
def evaluate(model, held, gen, dev):
    model.eval()
    bnll, bclash, modeerr, mtm_acc = [], [], [], []
    for pr in held:
        x, s, R, n, bnd, s_bnd = pr["x"], pr["s"], pr["R"], pr["n"], pr["bnd"], pr["s_bnd"]
        k = min(4, n); blk = blob_mask(n, R, k, gen, dev)
        bnll.append(float(-block_log_prob(model, x, s, blk, bnd, s_bnd, R) / k))
        xn, sn, _ = sample_block(model, x, s, blk, bnd, s_bnd, R, gen=gen)
        bi = xn[blk]; allp = torch.cat([xn, bnd])                       # clash vs interior AND boundary
        d = torch.cdist(bi, allp); d[torch.arange(k), torch.nonzero(blk).squeeze(1)] = 1e9
        bclash.append(float((d.min(1).values < 0.8).float().mean()))
        # correct independence-MTM K=4, N=16
        u_old = float(cavity_energy(x, s, bnd, s_bnd)); lq_x = float(block_log_prob(model, x, s, blk, bnd, s_bnd, R))
        lus = [-2.0 * u_old - lq_x]                                      # current state weight (index 0)
        for _ in range(16):
            xt, stt, lqy = sample_block(model, x, s, blk, bnd, s_bnd, R, gen=gen)
            lus.append(-2.0 * float(cavity_energy(xt, stt, bnd, s_bnd)) - float(lqy))
        lu = torch.tensor(lus[1:], device=dev); sfwd = torch.logsumexp(lu, 0)
        jstar = int(torch.multinomial(torch.softmax(lu, 0), 1, generator=gen))
        lu_rev = lu.clone(); lu_rev[jstar] = lus[0]
        mtm_acc.append(float(torch.rand((), device=dev, generator=gen).log() < (sfwd - torch.logsumexp(lu_rev, 0))))
    model.train()
    return {"block_nll": st.mean(bnll), "block_clash": st.mean(bclash), "mtm_accept": st.mean(mtm_acc)}


def main():
    a = parse(); dev = torch.device(a.device)
    torch.manual_seed(0); gen = torch.Generator(device=dev).manual_seed(1)
    d = torch.load(a.dataset, map_location=dev, weights_only=False)
    X, S, L = d["x"].float(), d["s"].long(), float(d["L"]); split = min(a.train_frames, X.shape[0] - 1)
    train = build_pool(X, S, L, range(split), a.radii, a.r_ctx, gen)
    held = build_pool(X, S, L, range(split, X.shape[0]), a.radii, a.r_ctx, gen, per=1)[:20]
    print(f"cavity-block train={len(train)} held={len(held)} (WITH boundary) dev={dev}", flush=True)
    model = KA3DScaffoldAR(flow_bins=a.flow_bins, flow_tail=a.flow_tail).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr); start, series, best = 0, [], math.inf
    if a.resume and a.artifact.exists():
        stt = torch.load(a.artifact, map_location=dev, weights_only=False)
        model.load_state_dict(stt["state_dict"]); opt.load_state_dict(stt["optimizer"])
        start = int(stt["step"]) + 1; series = stt.get("series", []); print(f"resumed {start}", flush=True)
    a.artifact.parent.mkdir(parents=True, exist_ok=True); t0 = time.time()
    for step in range(start, a.steps + 1):
        ids = torch.randint(len(train), (a.batch,), generator=gen, device=dev); loss = X.new_zeros(())
        for i in ids:
            pr = train[int(i)]; blk = rand_blob(pr["n"], pr["R"], gen, dev)
            loss = loss - block_log_prob(model, pr["x"], pr["s"], blk, pr["bnd"], pr["s_bnd"], pr["R"]) / int(blk.sum())
        loss = loss / a.batch
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()
        if step % a.eval_every == 0 or step == a.steps:
            m = evaluate(model, held, gen, dev); m["step"] = step; series.append(m)
            print(f"step {step:6d} block_nll={m['block_nll']:+.3f} block_clash={100*m['block_clash']:.1f}% "
                  f"mtm_accept={100*m['mtm_accept']:.1f}% ({time.time()-t0:.0f}s)", flush=True)
            torch.save({"state_dict": model.state_dict(), "optimizer": opt.state_dict(), "step": step,
                        "series": series, "config": vars(a)}, a.artifact)


if __name__ == "__main__":
    main()
