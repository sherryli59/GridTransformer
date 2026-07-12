"""Can a FIXED-STEP flow (exact-logdet-able, no L-BFGS) reach the good basin by soft->hard HOMOTOPY?

Anneal the soft-core radius a from fat (untangle: particles slide apart freely) to thin (harden toward the
true LJ core), with p=6,s~sigma,w~4eps so the soft-core -> true LJ repulsion as a->0. Fixed T steps, fixed
eta schedule => per-step Jacobian logdet is well-defined. We report TRUE KA block energy after the FLOW
ALONE (the deployable, exact-able number) and the min per-step eigenvalue of I-eta*H (injectivity => exact).
Compare vs the +6.9 soft->LBFGS floor and +0.9 ideal. If flow-alone reaches < ~6 with min-eig>0, the exact
deterministic corrector is within reach with forward steps; if injectivity fails at the hard end, we need
backward-Euler there."""
import torch, statistics as st
from liquid_coupling_flow.ka3d_ebm_batched import KA3DScaffoldEBMBatched
from liquid_coupling_flow.ka3d_scaffold_ar import label_to_scaffold, fixed_ball_scaffold
from liquid_coupling_flow.ka3d_cavity_ar import _mic
from liquid_coupling_flow.ka3d_cavity_carve import carve
from liquid_coupling_flow.ka_energy import ka_energy

dev = "cuda"; R = 2.0; RCTX = 2.5; K = 12; M = 8; BIGL = 100.0


def load_base():
    ck = torch.load("liquid_coupling_flow/artifacts/ka3d_cavity_ebm3ax.pt", map_location=dev, weights_only=False)
    m = KA3DScaffoldEBMBatched(cat_bins=128, cat_range=2.5).to(dev)
    m.load_state_dict(ck["state_dict"], strict=False); m.eval(); m.use_frame = False
    return m


def block_E(xblk, sblk, cage_x, cage_s):
    allx = torch.cat([xblk, cage_x], 1); alls = torch.cat([sblk, cage_s], 1)
    return ka_energy(allx, alls.long(), BIGL)


def sc_U(x, cage_x, a, w=4.0, s=0.9, p=6):
    allx = torch.cat([x, cage_x], 1); Kk, P = x.shape[1], allx.shape[1]
    r2 = (x[:, :, None, :] - allx[:, None, :, :]).square().sum(-1)
    r2 = r2.masked_fill(torch.eye(Kk, P, device=dev, dtype=torch.bool)[None], 1e18)
    return (0.5 * w * (s * s / (r2 + a * a)).pow(p).sum((1, 2))).sum()


def anneal_flow(xb, cage_x, T=120, a0=0.7, a1=0.15, eta=1.5e-4, check_inj=False):
    """soft->hard homotopy: a linearly annealed fat->thin; fixed steps; accumulate logdet + min-eig."""
    Kk = xb.shape[1]; M = xb.shape[0]; logdet = xb.new_zeros(M); mineig = xb.new_full((M,), 1e9)
    for t in range(T):
        a = a0 + (a1 - a0) * t / (T - 1)
        x = xb.detach().requires_grad_(True)
        g = torch.autograd.grad(sc_U(x, cage_x, a), x, create_graph=check_inj)[0]
        if check_inj:
            flat_g = g.reshape(M, 3 * Kk)
            H = torch.stack([torch.autograd.grad(flat_g[:, i].sum(), x, retain_graph=True)[0].reshape(M, 3 * Kk)
                             for i in range(3 * Kk)], 1)
            with torch.no_grad():
                Jm = torch.eye(3 * Kk, device=dev)[None] - eta * H
                logdet = logdet + torch.slogdet(Jm)[1]
                mineig = torch.minimum(mineig, torch.linalg.eigvalsh(0.5 * (Jm + Jm.transpose(1, 2)))[:, 0])
        with torch.no_grad():
            xb = xb - eta * g.detach()
    return xb.detach(), logdet.detach(), mineig.detach()


def floor(xb0, sblk, cage_x, cage_s, iters=150):
    x = xb0.clone().detach().requires_grad_(True)
    opt = torch.optim.LBFGS([x], lr=0.3, max_iter=iters, line_search_fn="strong_wolfe", tolerance_grad=1e-6)

    def closure():
        opt.zero_grad(); e = block_E(x, sblk, cage_x, cage_s).sum(); e.backward(); return e
    opt.step(closure)
    with torch.no_grad():
        return block_E(x, sblk, cage_x, cage_s).mean().item()


m = load_base()
d = torch.load("liquid_coupling_flow/artifacts/ka3d_dataset_N512_T0.5.pt", map_location=dev, weights_only=False)
X, S, L = d["x"].to(dev).float(), d["s"].to(dev).long(), float(d["L"]); gen = torch.Generator(device=dev).manual_seed(0)
flowE, flowLB, minei, ncav = [], [], [], 0
for ci in range(900, 970):
    c = torch.rand(3, generator=gen, device=dev) * L; p = carve(X[ci], S[ci], c, R, L)
    if p["n_in"] < K + 6:
        continue
    xo, so, _ = label_to_scaffold(_mic(p["x_in"], c, L), p["s_in"], R)
    xout = _mic(p["x_out"], c, L); bm = xout.norm(dim=-1) < (R + RCTX); bnd, sb = xout[bm], p["s_out"][bm]
    n = xo.shape[0]; a = fixed_ball_scaffold(n, R, dev); seed = int(torch.randint(n, (), generator=gen, device=dev))
    blk = torch.zeros(n, dtype=torch.bool, device=dev); blk[(a - a[seed]).norm(dim=-1).topk(K, largest=False).indices] = True
    order = torch.argsort(blk.to(torch.uint8), stable=True); n_ret = int((~blk).sum()); blk_re = blk[order]
    xo_re, so_re = xo[order], so[order]
    cage_x = torch.cat([bnd[None].expand(M, bnd.shape[0], 3), xo_re[None, :n_ret].expand(M, n_ret, 3)], 1)
    cage_s = torch.cat([sb[None].expand(M, bnd.shape[0]), so_re[None, :n_ret].expand(M, n_ret)], 1)
    data_blk = xo_re[None, n_ret:].expand(M, K, 3).contiguous(); data_bs = so_re[None, n_ret:].expand(M, K)
    E_data = block_E(data_blk, data_bs, cage_x, cage_s).mean().item()
    x0, s0, _ = m.sample_block_b(xo_re[None].expand(M, n, 3), so_re[None].expand(M, n), blk_re, bnd, sb, R, gen=gen)
    ar_blk = x0[:, n_ret:].contiguous(); ar_bs = s0[:, n_ret:]
    xf, ld, me = anneal_flow(ar_blk, cage_x, check_inj=True)
    flowE.append((block_E(xf, ar_bs, cage_x, cage_s).mean().item() - E_data) / K)
    flowLB.append((floor(xf, ar_bs, cage_x, cage_s) - E_data) / K)
    minei.append(me.min().item())
    ncav += 1
    print(f"  cav {ncav}: flow-alone {flowE[-1]:+8.1f}  flow->LBFGS {flowLB[-1]:+7.2f}  min-eig {minei[-1]:+.3f}", flush=True)
    if ncav >= 10:
        break


def med(v):
    return st.median(v)


print(f"\nK={K}, {ncav} cavities  (soft->hard annealed FIXED-STEP flow; good basin ~ +1)", flush=True)
print(f"  flow-alone (deployable, exact-able)  {med(flowE):+8.2f}   (vs +6.9 soft->LBFGS, +0.9 ideal)", flush=True)
print(f"  flow->LBFGS polish (diagnostic)      {med(flowLB):+8.2f}", flush=True)
print(f"  INJECTIVITY min per-step eig I-eta*H  {min(minei):+.4f}   ({'OK diffeo/exact' if min(minei) > 0 else 'FOLD at hard end -> backward-Euler'})", flush=True)
print("VERDICT:", "flow-alone reaches basin + injective -> EXACT corrector ready to build" if med(flowE) < 8 and min(minei) > 0
      else ("flow reaches basin but FOLDS -> need backward-Euler hard phase" if med(flowE) < 8 else "flow-alone too hot -> tune schedule/steps"), flush=True)
