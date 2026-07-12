"""Are the AR block positions TANGLED (overlapping -> need barrier crossing, no deterministic map fixes it)
or merely TOO CLOSE (slidable apart -> soft-then-hard relaxation reaches the good basin)?

Per cavity, TRUE-KA inherent-structure floor of the AR block (AR species) under three relaxers:
  (1) LBFGS-direct              : L-BFGS straight from clashy AR pos (gets stuck ~+105)
  (2) softcore-declash -> LBFGS : first slide particles apart with a smooth soft-core GD, then L-BFGS polish
  (3) langevin-anneal -> LBFGS  : short STOCHASTIC soft-core Langevin (can cross barriers -> untangle), then polish
Compare to C-level ~+2 (DATA pos + AR species = the good basin exists). If (2) reaches ~+2 the tangles are
slidable => a deterministic soft-then-hard corrector works (exact-able). If only (3) reaches it => genuine
overlaps needing stochastic untangling => the fix must be a better BASE or an SMC guard, not a det. map."""
import torch, statistics as st
from liquid_coupling_flow.ka3d_ebm_batched import KA3DScaffoldEBMBatched
from liquid_coupling_flow.ka3d_scaffold_ar import label_to_scaffold, fixed_ball_scaffold
from liquid_coupling_flow.ka3d_cavity_ar import _mic
from liquid_coupling_flow.ka3d_cavity_carve import carve
from liquid_coupling_flow.ka_energy import ka_energy

dev = "cuda"; R = 2.0; RCTX = 2.5; K = 12; M = 8; BIGL = 100.0
S_SC, A_SC, W_SC, P_SC = 0.95, 0.35, 6.0, 4                                            # strong declash soft-core


def load_base():
    ck = torch.load("liquid_coupling_flow/artifacts/ka3d_cavity_ebm3ax.pt", map_location=dev, weights_only=False)
    m = KA3DScaffoldEBMBatched(cat_bins=128, cat_range=2.5).to(dev)
    m.load_state_dict(ck["state_dict"], strict=False); m.eval(); m.use_frame = False
    return m


def block_E(xblk, sblk, cage_x, cage_s):
    allx = torch.cat([xblk, cage_x], 1); alls = torch.cat([sblk, cage_s], 1)
    return ka_energy(allx, alls.long(), BIGL)


def softcore_grad(xb, cage_x):
    x = xb.detach().requires_grad_(True)
    allx = torch.cat([x, cage_x], 1); Kk, P = x.shape[1], allx.shape[1]
    r2 = (x[:, :, None, :] - allx[:, None, :, :]).square().sum(-1)
    r2 = r2.masked_fill(torch.eye(Kk, P, device=dev, dtype=torch.bool)[None], 1e18)
    U = (0.5 * W_SC * (S_SC * S_SC / (r2 + A_SC * A_SC)).pow(P_SC).sum((1, 2))).sum()
    return torch.autograd.grad(U, x)[0].detach()


def softcore_declash(xb, cage_x, T=40, eta=2e-3):
    for _ in range(T):
        xb = xb - eta * softcore_grad(xb, cage_x)
    return xb.detach()


def langevin_declash(xb, cage_x, gen, T=200, eta=2e-3, temp=0.05):
    """soft-core Langevin: descent + noise => CAN climb barriers to untangle genuine overlaps."""
    for _ in range(T):
        noise = torch.randn(xb.shape, generator=gen, device=dev) * ((2 * eta * temp) ** 0.5)
        xb = xb - eta * softcore_grad(xb, cage_x) + noise
    return xb.detach()


def floor(xb0, sblk, cage_x, cage_s, iters=150):
    x = xb0.clone().detach().requires_grad_(True)
    opt = torch.optim.LBFGS([x], lr=0.3, max_iter=iters, line_search_fn="strong_wolfe",
                            tolerance_grad=1e-6, tolerance_change=1e-9)

    def closure():
        opt.zero_grad(); e = block_E(x, sblk, cage_x, cage_s).sum(); e.backward(); return e
    opt.step(closure)
    with torch.no_grad():
        return block_E(x, sblk, cage_x, cage_s).mean().item()


m = load_base()
d = torch.load("liquid_coupling_flow/artifacts/ka3d_dataset_N512_T0.5.pt", map_location=dev, weights_only=False)
X, S, L = d["x"].to(dev).float(), d["s"].to(dev).long(), float(d["L"]); gen = torch.Generator(device=dev).manual_seed(0)
f1, f2, f3, cC, ncav = [], [], [], [], 0
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
    f1.append((floor(ar_blk, ar_bs, cage_x, cage_s) - E_data) / K)
    f2.append((floor(softcore_declash(ar_blk, cage_x), ar_bs, cage_x, cage_s) - E_data) / K)
    f3.append((floor(langevin_declash(ar_blk, cage_x, gen), ar_bs, cage_x, cage_s) - E_data) / K)
    cC.append((floor(data_blk, ar_bs, cage_x, cage_s) - E_data) / K)
    ncav += 1
    print(f"  cav {ncav}: (1)direct {f1[-1]:+8.1f}  (2)soft->LBFGS {f2[-1]:+8.1f}  (3)langevin->LBFGS {f3[-1]:+8.1f}  | C(DATApos) {cC[-1]:+5.2f}", flush=True)
    if ncav >= 10:
        break


def med(v):
    return st.median(v)


print(f"\nK={K}, {ncav} cavities  (median dE/block-particle above equilibrium; good basin ~ +2)", flush=True)
print(f"  (1) LBFGS direct              {med(f1):+8.2f}", flush=True)
print(f"  (2) softcore-declash -> LBFGS {med(f2):+8.2f}   (LOW => tangles are SLIDABLE, det. corrector works)", flush=True)
print(f"  (3) langevin-anneal  -> LBFGS {med(f3):+8.2f}   (only-this LOW => genuine overlaps, need stochastic)", flush=True)
print(f"  C   DATApos+ARspecies         {med(cC):+8.2f}   (the good basin exists)", flush=True)
if med(f2) < 8.0:
    print("VERDICT: SLIDABLE -> soft-then-hard deterministic corrector reaches the good basin (make it exact next)", flush=True)
elif med(f3) < 8.0:
    print("VERDICT: GENUINE OVERLAPS -> only stochastic untangling works; deterministic map cannot. Base/SMC fix", flush=True)
else:
    print("VERDICT: neither reaches the basin from AR positions -> AR proposal too far; rethink the block move", flush=True)
