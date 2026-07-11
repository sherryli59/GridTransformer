"""Does the K=4 block-MTM chain decorrelate with ENOUGH compute (more proposals + more sweeps)?
Definitive two-chain test on ONE small cavity (R=1.6): chain A starts from the reference C0, chain B from
an INDEPENDENT full-regen. Track qA=ov(A,C0), qB=ov(B,C0), qAB=ov(A,B) per sweep. If qA (from 1.0) and qB
(from low) MEET at a common value, the chain is equilibrated and that value = the pinned self-overlap (PTS).
If qA stays ~1.0, local K=4 moves cannot rearrange the pinned interior in feasible time."""
import argparse, time
import torch
from liquid_coupling_flow.ka3d_scaffold_ebm import KA3DScaffoldEBM
from liquid_coupling_flow.ka3d_scaffold_ar import label_to_scaffold, fixed_ball_scaffold
from liquid_coupling_flow.ka3d_cavity_ar import _mic
from liquid_coupling_flow.ka3d_cavity_carve import carve
from liquid_coupling_flow.ka_energy import ka_energy

ART = "liquid_coupling_flow/artifacts"; dev = "cuda"


def energy(xo, so, bnd, s_bnd):
    x = torch.cat([xo, bnd], 0); s = torch.cat([so, s_bnd], 0)
    return float(ka_energy(x[None], s.long()[None], 100.0)[0])


def overlap(C, C0, a=0.3):
    return float((torch.cdist(C0, C).min(1).values < a).float().mean())


def imtm(m, xo, so, blk, bnd, s_bnd, R, beta, Nt, gen):
    u0 = energy(xo, so, bnd, s_bnd); lqx = float(m.block_log_prob(xo, so, blk, bnd, s_bnd, R))
    lus = [-beta * u0 - lqx]; props = []
    for _ in range(Nt):
        xn, sn, lq = m.sample_block(xo, so, blk, bnd, s_bnd, R, gen=gen)
        props.append((xn, sn)); lus.append(-beta * energy(xn, sn, bnd, s_bnd) - float(lq))
    lu = torch.tensor(lus[1:], device=dev); sfwd = torch.logsumexp(lu, 0)
    J = int(torch.multinomial(torch.softmax(lu, 0), 1, generator=gen)); lr = lu.clone(); lr[J] = lus[0]
    if torch.rand((), device=dev, generator=gen).log() < (sfwd - torch.logsumexp(lr, 0)):
        return props[J][0], props[J][1], 1
    return xo, so, 0


def blob(n, R, K, gen):
    a = fixed_ball_scaffold(n, R, dev); seed = int(torch.randint(n, (), generator=gen, device=dev))
    mm = torch.zeros(n, dtype=torch.bool, device=dev)
    mm[(a - a[seed]).norm(dim=-1).topk(min(K, n), largest=False).indices] = True
    return mm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--R", type=float, default=1.6); ap.add_argument("--sweeps", type=int, default=150)
    ap.add_argument("--ntrials", type=int, default=32); ap.add_argument("--K", type=int, default=4)
    ap.add_argument("--beta", type=float, default=2.0); ap.add_argument("--r-ctx", type=float, default=2.5)
    ap.add_argument("--out", default="reports/logs-2026-07-11/focused_converge.pt")
    a = ap.parse_args(); R = a.R
    ck = torch.load(f"{ART}/ka3d_cavity_ebm3ax.pt", map_location=dev, weights_only=False)
    m = KA3DScaffoldEBM(cat_bins=128, cat_range=2.5).to(dev)
    m.load_state_dict(ck["state_dict"], strict=False); m.eval(); m.use_frame = False
    d = torch.load(f"{ART}/ka3d_dataset_N512_T0.5.pt", map_location=dev, weights_only=False)
    X, S, L = d["x"].to(dev).float(), d["s"].to(dev).long(), float(d["L"])
    gen = torch.Generator(device=dev).manual_seed(0)
    # first valid cavity
    for ci in range(900, 950):
        c = torch.rand(3, generator=gen, device=dev) * L; p = carve(X[ci], S[ci], c, R, L)
        if p["n_in"] >= a.K + 6:
            break
    xo, so, _ = label_to_scaffold(_mic(p["x_in"], c, L), p["s_in"], R)
    xout = _mic(p["x_out"], c, L); bm = xout.norm(dim=-1) < (R + a.r_ctx)
    bnd, s_bnd, n = xout[bm], p["s_out"][bm], xo.shape[0]
    C0 = xo.clone()
    allmask = torch.ones(n, dtype=torch.bool, device=dev)
    Ax, As = xo.clone(), so.clone()                                   # chain A: reference start
    Bx, Bs, _ = m.sample_block(xo, so, allmask, bnd, s_bnd, R, gen=gen)  # chain B: independent regen start
    Bs = Bs.clone()
    mv = 2 * n                                                        # moves/sweep (over-cover the interior)
    print(f"cavity R={R} n_in={n} n_bnd={bnd.shape[0]}  chain-B start ov(B,C0)={overlap(Bx,C0):.2f}  "
          f"K={a.K} ntrials={a.ntrials} mv/sweep={mv}", flush=True)
    traj = {"qA": [1.0], "qB": [overlap(Bx, C0)], "qAB": [overlap(Ax, Bx)]}
    accA = accB = tot = 0; t0 = time.time()
    for t in range(a.sweeps):
        for _ in range(mv):
            blk = blob(n, R, a.K, gen)
            Ax, As, aA = imtm(m, Ax, As, blk, bnd, s_bnd, R, a.beta, a.ntrials, gen)
            blk = blob(n, R, a.K, gen)
            Bx, Bs, aB = imtm(m, Bx, Bs, blk, bnd, s_bnd, R, a.beta, a.ntrials, gen)
            accA += aA; accB += aB; tot += 1
        traj["qA"].append(overlap(Ax, C0)); traj["qB"].append(overlap(Bx, C0)); traj["qAB"].append(overlap(Ax, Bx))
        if t % 10 == 0 or t == a.sweeps - 1:
            print(f"  sweep {t:3d}: qA={traj['qA'][-1]:.2f} qB={traj['qB'][-1]:.2f} qAB={traj['qAB'][-1]:.2f} "
                  f"accept {100*accA/tot:.0f}%/{100*accB/tot:.0f}%  ({time.time()-t0:.0f}s)", flush=True)
    torch.save({"traj": traj, "R": R, "n": n, "config": vars(a)}, a.out)
    print(f"\nVERDICT: qA {traj['qA'][0]:.2f}->{traj['qA'][-1]:.2f}, qB {traj['qB'][0]:.2f}->{traj['qB'][-1]:.2f}; "
          f"meet={'YES' if abs(traj['qA'][-1]-traj['qB'][-1])<0.1 else 'NO'} (|dq|={abs(traj['qA'][-1]-traj['qB'][-1]):.2f})", flush=True)
    print(f"saved {a.out}", flush=True)


if __name__ == "__main__":
    main()
