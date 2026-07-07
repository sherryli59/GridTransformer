"""GB1: the collective MOVE MODEL — transition conditional q(x'_block | x_block, env, beta).

A1/A2 learn STATE conditionals (where a block sits given surroundings); this learns a TRANSITION: given the
CURRENT positions of a k=8 block (the measured cold move class is 1-8 particle compact string hops
[[stage-b spec]]), propose where they hop TOGETHER. Trained on harvested relaxation events
(artifacts/ka_event_bank_N100.pt); MH-guarded with the exact two-way density (one forward pass per direction).

Construction (maximal reuse): MoveModel = HeatBathModel (beta-FiLM) + a third token role carrying the CURRENT
block positions into the context (and into the pair-feature machinery). The env frame comes from
frame_ctx_slots (config-independent given the block slots) => identical for forward and reverse scoring.

EXACTNESS SUBTLETY (occupancy symmetry, the swap-breathe pattern): block selection is slot-based; after a move
the block slots must be occupied by exactly the moved particles, else the REVERSE move would select a different
set and detailed balance breaks. move_mh re-derives slot occupancy in the proposed state and ABORTS on mismatch
(symmetric abort preserves DB; aborts are an efficiency cost, counted in info).
"""
from __future__ import annotations
import os, math, torch, torch.nn as nn
from liquid_coupling_flow import ka_cluster as KC
from liquid_coupling_flow.ka_cluster_flow import _scaffold, slot_order, ART
from liquid_coupling_flow.ka_heatbath import HeatBathModel
from liquid_coupling_flow.ka_cluster_csmc import cluster_energy_total

D_MOBILE = 0.35


class MoveModel(HeatBathModel):
    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        d = self.sp_emb.embedding_dim
        self.cur_emb = nn.Embedding(1, d)                        # role: CURRENT position of a block particle

    def _ctx_tokens(self, pos, s, cluster_idx, sc, L):
        """Env context (inherited, env-anchored frame) + the current block positions as extra tokens — this is
        what makes it a TRANSITION conditional. Current-block tokens also join ctx_u/ctx_sp so the per-step
        pair features see them."""
        origin, R, ctx_tok, q_scaf, ctx_u, ctx_sp = super()._ctx_tokens(pos, s, cluster_idx, sc, L)
        B = pos.shape[0]
        sp_cl = s[:, cluster_idx] if s.dim() == 2 else s[cluster_idx][None].expand(B, -1)
        cur_u = KC.to_frame(pos[:, cluster_idx], origin, R, L)   # [B,k,2] current block, in-frame
        cur_tok = self._featpos(cur_u) + self.sp_emb(sp_cl) + self.cur_emb.weight[0]
        return (origin, R, torch.cat([ctx_tok, cur_tok], 1), q_scaf,
                torch.cat([ctx_u, cur_u], 1), torch.cat([ctx_sp, sp_cl], 1))

    @torch.no_grad()
    def sample_move(self, pos, s, cl, sc, L, beta):
        self._set_beta(beta, pos.shape[0], pos.dtype, pos.device)
        try:
            xC, logq = self.sample(pos, s, cl, sc, L)
        finally:
            self._beta_ctx = None
        return xC, logq                                          # [B,k,2], [B]

    def log_q_move(self, pos, s, cl, xC_query, sc, L, beta):
        """Exact density of xC_query given (env, CURRENT block = pos[:, cl], beta). Grad-capable."""
        self._set_beta(beta, pos.shape[0], pos.dtype, pos.device)
        try:
            lq = self.log_q(pos, s, cl, xC_query, sc, L)
        finally:
            self._beta_ctx = None
        return lq


@torch.no_grad()
def move_mh(m, pos, s, cl, sc, L, beta=2.0, geo=None, gen=None):
    """One collective block move at slots cl (pos/s slot-ordered). alpha = min(1, e^{-beta dU} q_rev/q_fwd).
    With geo, enforces OCCUPANCY SYMMETRY (abort if the proposed state's cl-slot occupants != the block
    particles). Returns (pos_new, s, info)."""
    B, N, _ = pos.shape; dev = pos.device
    beta_t = beta if torch.is_tensor(beta) else torch.full((B,), float(beta), device=dev)
    xC_old = pos[:, cl]
    xC_new, logq_f = m.sample_move(pos, s, cl, sc, L, beta_t)
    pos_new = pos.clone(); pos_new[:, cl] = xC_new
    logq_r = m.log_q_move(pos_new, s, cl, xC_old, sc, L, beta_t)
    dU = cluster_energy_total(xC_new, pos, s, cl, L) - cluster_energy_total(xC_old, pos, s, cl, L)
    abort = torch.zeros(B, dtype=torch.bool, device=dev)
    if geo is not None:
        occ = geo._curve_order(pos_new, N)[:, cl]                # proposed-state occupants of the block slots
        cl_sorted = cl.sort().values
        abort = ~(occ.sort(1).values == cl_sorted[None]).all(1)
    log_alpha = -beta_t * dU + logq_r - logq_f
    u = torch.rand(B, device=dev, generator=gen)
    accept = (torch.log(u) < log_alpha) & ~abort
    pos_out = torch.where(accept[:, None, None], pos_new, pos)
    return pos_out, s, {"accept": accept.float().mean().item(), "abort_frac": abort.float().mean().item(),
                        "dU_mean": float(dU.mean())}


@torch.no_grad()
def move_mh_sweep(m, pos, s, sc, L, geo, beta=2.0, k_block=8, n_moves=None, gen=None):
    """pi_beta-invariant collective-move sweep (lab-frame state; re-slot-order per move, scatter back)."""
    B, N, _ = pos.shape; dev = pos.device
    n_moves = N if n_moves is None else n_moves
    accs, abrs = [], []
    for _ in range(n_moves):
        seed = int(torch.randint(0, N, (1,), device=dev, generator=gen).item())
        order = geo._curve_order(pos, N)
        pos_o = torch.gather(pos, 1, order[..., None].expand(-1, -1, 2))
        s_o = torch.gather(s, 1, order)
        cl = KC.cluster_slots(seed, sc, k_block, L)
        pos_n, _, info = move_mh(m, pos_o, s_o, cl, sc, L, beta=beta, geo=geo, gen=gen)
        idx = order[:, cl]
        pos = pos.clone()
        pos[torch.arange(B, device=dev)[:, None], idx] = pos_n[:, cl]
        accs.append(info["accept"]); abrs.append(info["abort_frac"])
    return pos, s, {"accept": sum(accs) / len(accs), "abort_frac": sum(abrs) / len(abrs)}


# ---------- event bank -> training tuples ----------

def event_to_tuple(ev, s, sc, L, geo, k_block=8, d_mobile=D_MOBILE):
    """Bank event -> (pos_o [N,2] slot-ordered xa, s_o [N], cl [k], target [k,2] = block particles' OWN xb
    positions, beta). Returns None if the mobile set is not fully covered by the seed's k-NN block (partial
    events give noisy supervision). Particle correspondence: xa/xb rows are the same particle; one shared
    slot-order permutation keeps it."""
    xa, xb = ev["xa"], ev["xb"]
    N = xa.shape[0]
    order = geo._curve_order(xa[None], N)[0]                     # slot j <- particle order[j]
    pos_o, xb_o = xa[order], xb[order]
    s_o = s[order]
    d = xb_o - pos_o
    d = (d - L * torch.round(d / L)).norm(dim=-1)                # [N] per-slot displacement
    mobile_slots = (d > d_mobile).nonzero().squeeze(-1)
    mob_set = set(mobile_slots.tolist())
    # try each mobile slot as the block seed (displacement-ordered) until one block covers the whole set
    for seed in sorted(mob_set, key=lambda j: -float(d[j])):
        cl = KC.cluster_slots(seed, sc, k_block, L)
        if mob_set <= set(cl.tolist()):
            return pos_o, s_o, cl, xb_o[cl], float(ev["beta"])
    return None


def _augment_pair(xa, xb, L, gen=None):
    """Shared random shift + D4 (rot90/reflect) applied to BOTH frames (events are jointly equivariant)."""
    sh = (torch.rand(2, generator=gen) * L).to(xa.device)
    k = int(torch.randint(0, 4, (1,), generator=gen).item())
    fl = bool(torch.randint(0, 2, (1,), generator=gen).item())
    out = []
    for x in (xa, xb):
        y = torch.remainder(x + sh, L)
        for _ in range(k):                                       # rotate 90deg about the box center
            y = torch.stack([y[:, 1], L - y[:, 0]], -1)
        if fl:
            y = torch.stack([y[:, 0], L - y[:, 1]], -1)
        out.append(torch.remainder(y, L))
    return out[0], out[1]


def build_tuples(bank="ka_event_bank_N100.pt", n_aug=8, k_block=8, device="cpu", seed=0):
    """All bank events (+ time-reversal: (xb,xa) is an equilibrium event too) x n_aug augmentations ->
    training tuples grouped by block pattern (shared cl => batchable). Returns (groups, n_skipped, n_total)."""
    lad = torch.load(os.path.join(ART, bank), map_location=device, weights_only=False)
    sc, L, geo = _scaffold(100, device)
    s = None
    ref = torch.load(os.path.join(ART, "ka_reference_N100.pt"), map_location=device, weights_only=False)
    s = ref["s"].long().to(device)
    gen = torch.Generator().manual_seed(seed)
    groups, n_skip, n_tot = {}, 0, 0
    for b, grp in lad["harvest"].items():
        for ev in grp["events"]:
            if not (ev["single_cluster"] and ev["k"] <= 10):
                continue
            for xa, xb in ((ev["xa"], ev["xb"]), (ev["xb"], ev["xa"])):     # time-flip
                for _ in range(n_aug):
                    aa, bb = _augment_pair(xa.to(device), xb.to(device), L, gen=gen)
                    t = event_to_tuple({"xa": aa, "xb": bb, "beta": b}, s, sc, L, geo, k_block=k_block)
                    n_tot += 1
                    if t is None:
                        n_skip += 1; continue
                    key = tuple(t[2].tolist())
                    groups.setdefault(key, []).append(t)
    return groups, n_skip, n_tot


def train(steps=15000, n_ctx=24, d_model=128, n_head=4, n_layer=3, lr=3e-4, n_aug=8, k_block=8,
          val_frac=0.1, tag="", device="cuda" if torch.cuda.is_available() else "cpu"):
    """MLE on the event bank: -log q(target | current block, env, beta), batched within block-pattern groups.
    Val split at the GROUP level (whole block patterns held out — stricter than per-tuple)."""
    import time
    sc, L, geo = _scaffold(100, device)
    groups, n_skip, n_tot = build_tuples(n_aug=n_aug, k_block=k_block, device=device)
    keys = sorted(groups.keys())
    g = torch.Generator().manual_seed(1)
    perm = torch.randperm(len(keys), generator=g).tolist()
    n_val = max(1, int(val_frac * len(keys)))
    val_keys = {keys[i] for i in perm[:n_val]}
    tr_keys = [k for k in keys if k not in val_keys]
    arch = {"n_bins": 24, "box": 3.0, "n_ctx": n_ctx, "d_model": d_model, "n_head": n_head, "n_layer": n_layer,
            "head": "spline", "num_flow_bins": 16, "tail_bound": 3.5, "pair_feats": True}
    m = MoveModel(rho=1.2, **arch).to(device).train()
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=1e-4)
    warm = 300
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda st: min((st + 1) / warm,
        0.5 + 0.5 * math.cos(math.pi * max(0, st - warm) / max(1, steps - warm))))
    nparam = sum(p.numel() for p in m.parameters())
    n_tr = sum(len(groups[k]) for k in tr_keys)
    print(f"MOVE MODEL train: {n_tr} train tuples / {len(tr_keys)} patterns (+{n_val} val patterns), "
          f"skipped {n_skip}/{n_tot} (coverage), params {nparam/1e6:.2f}M, {arch}", flush=True)

    def batch_loss(key, tuples, train_mode=True):
        pos = torch.stack([t[0] for t in tuples]).to(device)
        s_b = torch.stack([t[1] for t in tuples]).to(device)
        cl = torch.tensor(key, dtype=torch.long, device=device)
        tgt = torch.stack([t[3] for t in tuples]).to(device)
        beta = torch.tensor([t[4] for t in tuples], device=device)
        return -(m.log_q_move(pos, s_b, cl, tgt, sc, L, beta) / k_block).mean()

    t0 = time.time(); best_val = float("inf")
    for step in range(steps):
        key = tr_keys[int(torch.randint(0, len(tr_keys), (1,)).item())]
        tup = groups[key]
        if len(tup) > 64:
            idx = torch.randperm(len(tup))[:64].tolist(); tup = [tup[i] for i in idx]
        loss = batch_loss(key, tup)
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 5.0)
        opt.step(); sched.step()
        if step % 500 == 0 or step == steps - 1:
            m.eval()
            with torch.no_grad():
                vl = [batch_loss(k, groups[k]) .item() for k in val_keys]
            m.train()
            v = sum(vl) / len(vl)
            print(f"  step {step:5d} train {loss.item():.3f}  VAL {v:.3f}  lr {sched.get_last_lr()[0]:.1e} "
                  f"{time.time()-t0:.0f}s", flush=True)
            if v < best_val:
                best_val = v
                torch.save({"state_dict": m.state_dict(), "step": step, "val": v, "k_block": k_block, **arch},
                           os.path.join(ART, f"ka_move_model{tag}_N100.pt"))
    print(f"saved best (val {best_val:.3f}) -> ka_move_model{tag}_N100.pt", flush=True)


def _load(ck, device):
    m = MoveModel(rho=1.2, n_bins=ck["n_bins"], box=ck["box"], n_ctx=ck["n_ctx"], d_model=ck["d_model"],
                  n_head=ck["n_head"], n_layer=ck["n_layer"], head=ck["head"],
                  num_flow_bins=ck["num_flow_bins"], tail_bound=ck["tail_bound"],
                  pair_feats=ck["pair_feats"]).to(device).eval()
    m.load_state_dict(ck["state_dict"]); return m


if __name__ == "__main__":
    import sys
    train(steps=int(sys.argv[1]) if len(sys.argv) > 1 else 15000)
