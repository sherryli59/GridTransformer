"""SWEEP the declash potential for the physics-relaxation corrector. Goal: ONE smooth potential that
BOTH (a) declashes a clashy K=12 AR block below the 29% full-cage floor, AND (b) keeps an EXACT logdet
(analytic per-step slogdet(I - eta*H) matches finite-difference on a toy).

Potential = purely-repulsive SOFT-CORE POWER LAW between the block and {block + cage}:
    U = w * sum_{i<j-ish} (s^2 / (r_ij^2 + a^2))^p
Force ~ 2 p w s^{2p} r / (r^2+a^2)^{p+1}: MONOTONE-increasing as r shrinks (unlike a Gaussian, whose
force vanishes at r->0 and can't separate the worst overlaps), bounded by the soft core a (=> no clamp
=> exact Hessian logdet). We sweep (p, a, w, eta, T) and report declash + logdet-match for each."""
import torch, statistics as st, itertools
from liquid_coupling_flow.ka3d_ebm_batched import KA3DScaffoldEBMBatched
from liquid_coupling_flow.ka3d_scaffold_ar import label_to_scaffold, fixed_ball_scaffold
from liquid_coupling_flow.ka3d_cavity_ar import _mic
from liquid_coupling_flow.ka3d_cavity_carve import carve

dev = "cuda"; R = 2.0; RCTX = 2.5; K = 12; M = 16; CUT = 0.8


def load_base():
    ck = torch.load("liquid_coupling_flow/artifacts/ka3d_cavity_ebm3ax.pt", map_location=dev, weights_only=False)
    m = KA3DScaffoldEBMBatched(cat_bins=128, cat_range=2.5).to(dev)
    m.load_state_dict(ck["state_dict"], strict=False); m.eval(); m.use_frame = False
    return m


def softcore_U(xb, cage_x, s, a, w, p):
    """Purely-repulsive soft-core power law between block xb [.,K,3] and {block+cage}. Species-independent."""
    allx = torch.cat([xb, cage_x], 1); Kk, P = xb.shape[1], allx.shape[1]
    r2 = (xb[:, :, None, :] - allx[:, None, :, :]).square().sum(-1)                     # [.,K,P]
    eye = torch.eye(Kk, P, device=dev, dtype=torch.bool)[None]
    r2 = r2.masked_fill(eye, 1e18)
    return 0.5 * w * (s * s / (r2 + a * a)).pow(p).sum((1, 2))


def relax(xb, cage_x, s, a, w, p, T, eta, want_logdet=False):
    """T steps of x <- x - eta*grad(U); NO clamp. If want_logdet, accumulate exact sum(log|det(I-eta*H)|)."""
    M = xb.shape[0]; Kk = xb.shape[1]; logdet = xb.new_zeros(M)
    for _ in range(T):
        x = xb.detach().requires_grad_(True)
        g = torch.autograd.grad(softcore_U(x, cage_x, s, a, w, p).sum(), x, create_graph=want_logdet)[0]
        if want_logdet:
            flat_g = g.reshape(M, 3 * Kk)
            H = torch.stack([torch.autograd.grad(flat_g[:, i].sum(), x, retain_graph=True)[0].reshape(M, 3 * Kk)
                             for i in range(3 * Kk)], 1)
            with torch.no_grad():
                logdet = logdet + torch.slogdet(torch.eye(3 * Kk, device=dev)[None] - eta * H)[1]
        with torch.no_grad():
            xb = xb - eta * g.detach()
    return xb.detach(), logdet.detach()


def clash_div(xb_full_re, blk_re, bnd):
    Mn = xb_full_re.shape[0]; bi_idx = torch.nonzero(blk_re).squeeze(1)
    allx = torch.cat([xb_full_re, bnd[None].expand(Mn, bnd.shape[0], 3)], 1)
    cl, blks = [], xb_full_re[:, blk_re]
    for mi in range(Mn):
        bi = blks[mi]; dm = torch.cdist(bi, allx[mi])
        for r, gi in enumerate(bi_idx):
            dm[r, gi] = 9.0
        cl.append(float((dm.min(1).values < CUT).float().mean()))
    div = [float((torch.cdist(blks[i], blks[j]).min(1).values < 0.3).float().mean())
           for i in range(min(Mn, 8)) for j in range(i + 1, min(Mn, 8))]
    return st.mean(cl), (st.mean(div) if div else 0.0)


def logdet_check(s, a, w, p, T, eta):
    """finite-diff volume of the relax map on a K=4 toy vs the analytic accumulated logdet."""
    gg = torch.Generator(device=dev).manual_seed(7)
    xt = torch.randn(1, 4, 3, generator=gg, device=dev) * 0.4
    cxt = torch.randn(1, 20, 3, generator=gg, device=dev)
    eps = 1e-3; base_out, ld1 = relax(xt, cxt, s, a, w, p, T, eta, want_logdet=True)
    J = torch.zeros(12, 12, device=dev)
    for i in range(12):
        xp = xt.clone().reshape(1, 12); xp[0, i] += eps
        op, _ = relax(xp.reshape(1, 4, 3), cxt, s, a, w, p, T, eta); J[:, i] = (op.reshape(12) - base_out.reshape(12)) / eps
    return ld1.item(), torch.slogdet(J)[1].item()


# ---- build a small fixed pool of clashy AR blocks ----
m = load_base()
d = torch.load("liquid_coupling_flow/artifacts/ka3d_dataset_N512_T0.5.pt", map_location=dev, weights_only=False)
X, S, L = d["x"].to(dev).float(), d["s"].to(dev).long(), float(d["L"]); gen = torch.Generator(device=dev).manual_seed(0)
pool = []
for ci in range(900, 960):
    c = torch.rand(3, generator=gen, device=dev) * L; p_ = carve(X[ci], S[ci], c, R, L)
    if p_["n_in"] < K + 6:
        continue
    xo, so, _ = label_to_scaffold(_mic(p_["x_in"], c, L), p_["s_in"], R)
    xout = _mic(p_["x_out"], c, L); bm = xout.norm(dim=-1) < (R + RCTX); bnd, sb = xout[bm], p_["s_out"][bm]
    n = xo.shape[0]; a_sc = fixed_ball_scaffold(n, R, dev); seed = int(torch.randint(n, (), generator=gen, device=dev))
    blk = torch.zeros(n, dtype=torch.bool, device=dev); blk[(a_sc - a_sc[seed]).norm(dim=-1).topk(K, largest=False).indices] = True
    order = torch.argsort(blk.to(torch.uint8), stable=True); n_ret = int((~blk).sum()); blk_re = blk[order]
    x0, s0, _ = m.sample_block_b(xo[None].expand(M, n, 3), so[None].expand(M, n), blk, bnd, sb, R, gen=gen)
    x0_re = x0[:, order]
    cage_x = torch.cat([bnd[None].expand(M, bnd.shape[0], 3), x0_re[:, :n_ret]], 1)
    pool.append((x0_re, n_ret, blk_re, bnd, cage_x))
    if len(pool) >= 8:
        break
raw = st.mean([clash_div(p_[0], p_[2], p_[3])[0] for p_ in pool])
print(f"pool={len(pool)}  raw clash={100*raw:.0f}%  (full-cage floor 29%, half-cage 57%)", flush=True)

# ---- sweep ----
configs = []
for p_pow, a, w, eta, T in itertools.product([3, 4], [0.35, 0.45], [2.0, 6.0], [8e-4, 2e-3], [30, 60]):
    configs.append((0.95, a, w, p_pow, T, eta))
print(f"sweeping {len(configs)} configs...\n", flush=True)
best = None
for (s, a, w, p_pow, T, eta) in configs:
    cs, ds = [], []
    for (x0_re, n_ret, blk_re, bnd, cage_x) in pool:
        xb1, _ = relax(x0_re[:, n_ret:], cage_x, s, a, w, p_pow, T, eta)
        xc = x0_re.clone(); xc[:, n_ret:] = xb1
        c1, d1 = clash_div(xc, blk_re, bnd); cs.append(c1); ds.append(d1)
    clash = 100 * st.mean(cs); div = 100 * st.mean(ds)
    la, fd = logdet_check(s, a, w, p_pow, T, eta)
    ld_ok = abs(la - fd) < 0.05 * max(1.0, abs(fd))
    tag = "  <== declash+exact" if (clash < 29 and ld_ok and div < 85) else ""
    print(f"p={p_pow} a={a} w={w:.0f} eta={eta:.0e} T={T:3d} | clash {clash:4.0f}% div {div:3.0f}% | "
          f"logdet {la:+8.2f} vs fd {fd:+8.2f} {'OK' if ld_ok else 'XX'}{tag}", flush=True)
    if clash < 29 and ld_ok and div < 85 and (best is None or clash < best[0]):
        best = (clash, s, a, w, p_pow, T, eta)
print("\nBEST:", best if best else "none passed both gates", flush=True)
