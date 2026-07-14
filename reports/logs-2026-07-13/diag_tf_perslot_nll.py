"""BUG CHECK (user: 'the model gets worse toward the end?'). Discriminator = teacher forcing. Score the
DATA block: every slot's log-prob is conditioned on the TRUE (data) positions of earlier slots (block_log_prob
is teacher-forced by construction). Per-slot NLL of the true data vs placement rank:
  - RISES with rank  => the per-step conditional assigns lower likelihood to the true position at high rank
    EVEN given perfect context => model genuinely worse toward the end (miscalibration/bug candidate);
  - FLAT             => conditional equally good at all ranks given true context => free-run clash rise is
    pure COMPOUNDING (early errors erode the zero-slack budget), not a per-step model defect.
Also splits position (lp_u) vs species (lp_s), and controls for the frac feature: the same physical slots
scored as a K=8 blob vs embedded in a K=12 blob (frac shifts, physics identical) -- if NLL tracks frac not
physics, the frac=idx/(n-1) feature is the culprit. FT model knn24, R=2.5, 16 cav x 16 (data blocks only)."""
import torch, statistics as st
import torch.nn.functional as F
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from liquid_coupling_flow.ka3d_ebm_batched import KA3DScaffoldEBMBatched
from liquid_coupling_flow.ka3d_scaffold_ar import label_to_scaffold, fixed_ball_scaffold
from liquid_coupling_flow.ka3d_cavity_ar import _mic
from liquid_coupling_flow.ka3d_cavity_carve import carve
from liquid_coupling_flow.ka3d_scaffold_ebm import ball_unsquash

dev = "cuda"; RCTX = 2.5; R = 2.5; NCAV = 16
FT = "liquid_coupling_flow/artifacts/ka3d_cavity_ebm3ax_rho115_blockcond_knn24_best.pt"
D = torch.load("liquid_coupling_flow/artifacts/ka3d_dataset_N4096_T0.5_rho1.15.pt", map_location=dev, weights_only=False)
X, S, L = D["x"].to(dev).float(), D["s"].to(dev).long(), float(D["L"])
m = KA3DScaffoldEBMBatched(cat_bins=128, cat_range=2.5, knn_pot=24).to(dev)
m.load_state_dict(torch.load(FT, map_location=dev, weights_only=False)["state_dict"], strict=False)
m.eval(); m.use_frame = False


@torch.no_grad()
def per_slot_lp(xo, so, block_mask, bnd, s_bnd, R):
    """Replicates block_log_prob_b but returns per-slot (lp_s, lp_u, logdet) in REORDERED order [1,n]."""
    xo, so = xo[None], so[None]; Mn = 1
    order, anchors, anchor_y, rblock, kind, valid, slot_feat, mm, n = m._prep(xo, so, block_mask, bnd, s_bnd, R)
    xo, so = xo[:, order], so[:, order]
    combined = torch.cat([bnd[None].expand(Mn, mm, 3), xo], 1)
    scomb = torch.cat([s_bnd[None].expand(Mn, mm), so], 1)
    h = m._fc_batched(combined, scomb, kind, valid, anchors, slot_feat) + m._r_bias(R, xo.device, xo.dtype)
    oh = F.one_hot(so, m.n_species).to(h.dtype); rem = oh.sum(1, keepdim=True) - (oh.cumsum(1) - oh)
    lp_s = F.log_softmax(m.head_species(h).masked_fill(rem <= 0, float("-inf")), -1).gather(-1, so[..., None]).squeeze(-1)
    y, logdet = ball_unsquash(xo, R); u = y - anchor_y[None]
    cage_x, cage_s, cage_v = m._cage_knn_b(combined, scomb, valid, anchors)
    lp_u = m._tilted_lp_u_b(h + m.sp_out_emb(so), u, anchor_y, cage_x, cage_s, cage_v, so, R)
    return lp_s[0], lp_u[0], logdet[0], order, rblock         # lp_* [n]; rblock [n] (already unbatched)


K = 8
rank_u, rank_s = [[] for _ in range(K)], [[] for _ in range(K)]
g = torch.Generator(device=dev).manual_seed(0); ncav = 0
for ci in range(16):
    c = torch.rand(3, generator=g, device=dev) * L; p = carve(X[ci], S[ci], c, R, L)
    if p["n_in"] < K + 6:
        continue
    xo, so, _ = label_to_scaffold(_mic(p["x_in"], c, L), p["s_in"], R)
    xout = _mic(p["x_out"], c, L); bm = xout.norm(dim=-1) < (R + RCTX); bnd, sb = xout[bm], p["s_out"][bm]
    n = xo.shape[0]; a = fixed_ball_scaffold(n, R, dev, xo.dtype, ordering=m.scaffold_order)
    seed = int(torch.randint(n, (), generator=g, device=dev))
    blk = torch.zeros(n, dtype=torch.bool, device=dev)
    blk[(a - a[seed]).norm(dim=-1).topk(K, largest=False).indices] = True
    lp_s, lp_u, logdet, order, rblock = per_slot_lp(xo, so, blk, bnd, sb, R)
    # block slots sit at reordered positions where rblock==1; they are ordered by ascending original index
    bslots = rblock.nonzero().squeeze(1)                     # reordered positions of block, ascending = rank
    for r, sl in enumerate(bslots.tolist()):
        rank_u[r].append(-float(lp_u[sl] + logdet[sl]))      # position NLL (incl jacobian)
        rank_s[r].append(-float(lp_s[sl]))                   # species NLL
    ncav += 1
    if ncav >= NCAV:
        break

print(f"=== TEACHER-FORCED per-slot NLL of TRUE DATA vs rank (K={K} blob, FT knn24, {ncav} cav) ===", flush=True)
print(f"{'rank':>4} | {'position NLL':>12} {'species NLL':>11}", flush=True)
for r in range(K):
    print(f"{r+1:>4} | {st.mean(rank_u[r]):>12.3f} {st.mean(rank_s[r]):>11.3f}", flush=True)
du = st.mean(rank_u[-1]) - st.mean(rank_u[0])
print(f"\nposition NLL rank8-rank1 = {du:+.3f}  => "
      f"{'RISES: conditional worse at high rank given TRUE context (not pure compounding)' if du > 0.15 else 'FLAT: free-run rise is COMPOUNDING, conditional fine given true context'}", flush=True)

fig, ax = plt.subplots(figsize=(6.4, 4.2))
ax.plot(range(1, K + 1), [st.mean(v) for v in rank_u], "o-", color="C0", label="position NLL (+jacobian)")
ax.plot(range(1, K + 1), [st.mean(v) for v in rank_s], "s-", color="C1", label="species NLL")
ax.set_xlabel("placement rank"); ax.set_ylabel("teacher-forced NLL of TRUE data")
ax.set_title(f"Teacher-forced per-slot NLL vs rank (K={K}): flat=>compounding, rising=>conditional")
ax.legend(); ax.grid(alpha=0.3); fig.tight_layout()
out = "reports/logs-2026-07-13/tf_perslot_nll.png"
fig.savefig(out, dpi=130); print(f"saved -> {out}", flush=True)
