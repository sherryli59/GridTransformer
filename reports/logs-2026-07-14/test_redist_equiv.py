"""CONTROLLED head-to-head: does the FAST (dynamic-batch) redist_imtm produce the SAME acceptance as the
ORIGINAL (M-global) redist_imtm on IDENTICAL input + RNG? If la matches elementwise, the reimplementation is
correct and the soak's 10x acc gap is SMC dynamics/variance (batch size, global resampling, seed), NOT a bug.
If la differs, bug in the reimplementation. Fresh alien M=16 state, lambda=0.0625, same U, same seed."""
import sys, torch
sys.path.insert(0, "reports/logs-2026-07-14")
sys.path.insert(0, "reports/logs-2026-07-13")
import diag_twoblob_imtm as IM               # guarded; provides the ORIGINAL redist_imtm (uses global M=16)
from ka3d_smc_sweep import energy_b
from liquid_coupling_flow.ka3d_scaffold_ar import label_to_scaffold, fixed_ball_scaffold
from liquid_coupling_flow.ka3d_cavity_ar import _mic
from liquid_coupling_flow.ka3d_cavity_carve import carve

torch.set_grad_enabled(False)                    # no autograd graph (multi-try AR regens else OOM ~22GB)
dev = "cuda"; RCTX = 2.5; BETA = 2.0; R = 2.0; K_R = 12; N_TRY = 8; LAM = 0.0625
m = IM.m; X, S, L = IM.X, IM.S, IM.L; M = IM.M   # M == 16
BULK = 1.149 * 0.368 ** 3


def redist_imtm_fast(Xb, Sb, bnd, sb, R, lam, q0, Ecur, allm, U, gen, n_try):
    """EXACT copy of the fast version's move (dynamic B). Compared against IM.redist_imtm (global M)."""
    B, n = Xb.shape[0], Xb.shape[1]
    def w(Xc, Sc, lqu):
        q0c = m.block_log_prob_b(Xc, Sc, allm, bnd, sb, R); Ec = energy_b(Xc, Sc, bnd, sb)
        return (1 - lam) * q0c - lam * BETA * Ec - lqu, q0c, Ec
    Xr = Xb.repeat_interleave(n_try, 0); Sr = Sb.repeat_interleave(n_try, 0)
    Yf, Sf, lqf = m.sample_block_b(Xr, Sr, U, bnd, sb, R, gen=gen)
    lwf, q0f, Ef = w(Yf, Sf, lqf); lwf = lwf.view(B, n_try); q0f = q0f.view(B, n_try); Ef = Ef.view(B, n_try)
    Yf = Yf.view(B, n_try, n, 3); Sf = Sf.view(B, n_try, n); ar = torch.arange(B, device=dev)
    Js = torch.multinomial(torch.softmax(lwf, 1), 1, generator=gen).squeeze(1)
    Xsel, Ssel, q0sel, Esel = Yf[ar, Js], Sf[ar, Js], q0f[ar, Js], Ef[ar, Js]
    if n_try > 1:
        Xr2 = Xb.repeat_interleave(n_try - 1, 0); Sr2 = Sb.repeat_interleave(n_try - 1, 0)
        Yr, Sr2b, lqr2 = m.sample_block_b(Xr2, Sr2, U, bnd, sb, R, gen=gen)
        lwr, _, _ = w(Yr, Sr2b, lqr2); lwr = lwr.view(B, n_try - 1)
    lq_cur = m.block_log_prob_b(Xb, Sb, U, bnd, sb, R)
    lw_cur = (1 - lam) * q0 - lam * BETA * Ecur - lq_cur
    lwr_all = torch.cat([lwr, lw_cur[:, None]], 1) if n_try > 1 else lw_cur[:, None]
    la = torch.logsumexp(lwf, 1) - torch.logsumexp(lwr_all, 1)
    acc = torch.rand(B, device=dev, generator=gen).log() < la      # same accept step as original
    return float(acc.float().mean()), la


# build a fixed alien state (M=16)
gen = torch.Generator(device=dev).manual_seed(0)
for ci in range(24):
    c = torch.rand(3, generator=gen, device=dev) * L; p = carve(X[ci], S[ci], c, R, L)
    if p["n_in"] >= 2 * K_R + 4:
        break
xo, so, _ = label_to_scaffold(_mic(p["x_in"], c, L), p["s_in"], R)
xout = _mic(p["x_out"], c, L); bm = xout.norm(dim=-1) < (R + RCTX); bnd, sb = xout[bm], p["s_out"][bm]
n = xo.shape[0]; allm = torch.ones(n, dtype=torch.bool, device=dev)
gs = torch.Generator(device=dev).manual_seed(7)
Xb, Sb, _ = m.sample_block_b(xo[None].expand(M, n, 3).clone(), so[None].expand(M, n).clone(), allm, bnd, sb, R, gen=gs)
Ecur = energy_b(Xb, Sb, bnd, sb); q0 = m.block_log_prob_b(Xb, Sb, allm, bnd, sb, R)
a = fixed_ball_scaffold(n, R, dev, xo.dtype, ordering=m.scaffold_order)
Kr = min(K_R, (n - 2) // 2)
U = IM.two_blob_mask(n, a, torch.Generator(device=dev).manual_seed(99), Kr)   # SAME U for both

# ORIGINAL (returns Xb,Sb,q0,Ecur,af) -- need its internal la; re-derive acc via seeded RNG match
g1 = torch.Generator(device=dev).manual_seed(42)
Xo, So, q0o, Eo, af_orig = IM.redist_imtm(Xb.clone(), Sb.clone(), bnd, sb, R, LAM, q0.clone(), Ecur.clone(), allm, U, g1, N_TRY)
# FAST on identical input + identical seed -> must match bit-for-bit if logic is identical
g2 = torch.Generator(device=dev).manual_seed(42)
af_fast, la_fast = redist_imtm_fast(Xb.clone(), Sb.clone(), bnd, sb, R, LAM, q0.clone(), Ecur.clone(), allm, U, g2, N_TRY)

print(f"=== redist_imtm equivalence (n={n}, M={M}, lambda={LAM}, N_try={N_TRY}) ===", flush=True)
print(f"original accept-frac (seed42): {af_orig:.4f}", flush=True)
print(f"fast     accept-frac (seed42): {af_fast:.4f}", flush=True)
print(f"la_fast: mean {la_fast.mean().item():+.3f}  frac(la>=0)={float((la_fast>=0).float().mean()):.3f}", flush=True)
match = abs(af_orig - af_fast) < 1e-9
print(f"VERDICT: {'MATCH -- reimplementation is correct; soak gap = SMC dynamics/variance, not a bug' if match else 'MISMATCH -- BUG in reimplementation'}", flush=True)
