"""Proposal-A: the in-frame conditional cluster generator. Generates the k cluster particles autoregressively
in the x_R-only equivariant frame, conditioned on the fixed scaffold-context neighbours, with factorized binned
heads over the in-frame position -> EXACT joint density (frame Jacobian = 1; box < L/2 -> no torus aliasing,
so no fold). Equivariant by construction (everything is in-frame). See the cluster-move design spec.

Interface (batched over B chains; a shared cluster_idx [k]):
  sample(pos[B,N,2], s[N], cluster_idx[k], sc[N,2], L) -> (xC_new[B,k,2], logq[B])
  log_q(pos[B,N,2], s[N], cluster_idx[k], xC_query[B,k,2], sc[N,2], L) -> logq[B]
"""
from __future__ import annotations
import os, math, torch, torch.nn as nn, torch.nn.functional as F
from liquid_coupling_flow import ka_cluster as KC

ART = os.path.join(os.path.dirname(__file__), "artifacts")


class ClusterProposal(nn.Module):
    def __init__(self, rho=1.2, n_bins=24, d_model=128, n_head=4, n_layer=3, n_ctx=16, box=3.0,
                 n_species=2, **_):
        super().__init__()
        self.rho, self.n_bins, self.n_ctx, self.box = rho, n_bins, n_ctx, box
        self.bin_w = 2 * box / n_bins
        self.enc = nn.Sequential(nn.Linear(6, d_model), nn.GELU(), nn.Linear(d_model, d_model))  # in-frame pos -> feat
        self.sp_emb = nn.Embedding(n_species, d_model)
        self.role_emb = nn.Embedding(2, d_model)                     # 0 = x_R neighbour, 1 = placed cluster particle
        self.query = nn.Linear(d_model, d_model)                     # query token from the slot's in-frame scaffold pos
        self.slot_emb = nn.Embedding(64, d_model)                    # canonical cluster-position index (k <= 64)
        layer = nn.TransformerEncoderLayer(d_model, n_head, 4 * d_model, batch_first=True, dropout=0.0)
        self.tr = nn.TransformerEncoder(layer, n_layer)
        self.head_a = nn.Linear(d_model, n_bins)
        self.head_b = nn.Linear(d_model, n_bins)
        self.bin_a_emb = nn.Embedding(n_bins, d_model)

    # ---- binning (in-frame coord in [-box, box); genuine zero outside) ----
    def _bin(self, u):
        b = ((u + self.box) / self.bin_w).floor().long()
        return torch.where((u >= -self.box) & (u < self.box), b.clamp(0, self.n_bins - 1),
                           torch.full_like(b, -1))

    def _bin_center(self, b):
        return (b.float() + 0.5) * self.bin_w - self.box

    def _featpos(self, p):                                          # [...,2] in-frame -> [...,d]
        a = math.pi * p / self.box
        return self.enc(torch.cat([p, torch.sin(a), torch.cos(a)], -1))

    # ---- per-step context for cluster particle i, given x_R tokens + placed cluster tokens ----
    def _step_ctx(self, qpos_i, slot_i, ctx_tok, placed_tok):
        """qpos_i [B,2] in-frame scaffold pos of cluster slot i; ctx_tok [B,n_ctx,d]; placed_tok [B,i,d]."""
        B = qpos_i.shape[0]
        q = (self.query(self._featpos(qpos_i)) + self.slot_emb(torch.full((B,), slot_i, device=qpos_i.device)))[:, None]
        seq = torch.cat([q, ctx_tok] + ([placed_tok] if placed_tok is not None and placed_tok.shape[1] else []), 1)
        return self.tr(seq)[:, 0]                                   # [B,d] query-token output

    def _ctx_tokens(self, pos, s, cluster_idx, sc, L):
        """Frame + the fixed x_R context tokens (in-frame) + the in-frame scaffold positions of the cluster.
        pos [B,N,2] and s [B,N] are CURVE/SLOT-ORDERED (slot j -> particle near sc[j])."""
        slots = KC.frame_ctx_slots(cluster_idx, sc, L, self.n_ctx)
        sc_c = KC.cluster_scaffold_center(cluster_idx, sc, L).to(pos.dtype)
        origin, R = KC.frame_from_positions(pos[:, slots, :], sc_c, L)               # [B,2],[B,2,2]
        ctx_u = KC.to_frame(pos[:, slots, :], origin, R, L)                          # [B,n_ctx,2] x_R in-frame
        ctx_tok = self._featpos(ctx_u) + self.sp_emb(s[:, slots]) + self.role_emb.weight[0]   # s [B,N] -> [B,n_ctx]
        q_scaf = KC.to_frame(sc[cluster_idx].to(pos.dtype)[None].expand(pos.shape[0], -1, -1), origin, R, L)  # [B,k,2]
        return origin, R, ctx_tok, q_scaf

    def _placed_tok(self, placed_u, placed_sp):
        if placed_u.shape[1] == 0:
            return None
        return self._featpos(placed_u) + self.sp_emb(placed_sp) + self.role_emb.weight[1]

    @torch.no_grad()
    def sample(self, pos, s, cluster_idx, sc, L):
        B = pos.shape[0]; k = cluster_idx.shape[0]; dev = pos.device
        origin, R, ctx_tok, q_scaf = self._ctx_tokens(pos, s, cluster_idx, sc, L)
        sp = s[:, cluster_idx]                                                       # [B,k] cluster species (fixed)
        placed_u = torch.zeros(B, 0, 2, device=dev); placed_sp = torch.zeros(B, 0, dtype=torch.long, device=dev)
        logq = torch.zeros(B, device=dev)
        for i in range(k):
            ctx = self._step_ctx(q_scaf[:, i], i, ctx_tok, self._placed_tok(placed_u, placed_sp))
            la = F.log_softmax(self.head_a(ctx), -1); ba = torch.multinomial(la.exp(), 1).squeeze(1)
            lb = F.log_softmax(self.head_b(ctx + self.bin_a_emb(ba)), -1); bb = torch.multinomial(lb.exp(), 1).squeeze(1)
            a = self._bin_center(ba) + (torch.rand(B, device=dev) - 0.5) * self.bin_w
            b = self._bin_center(bb) + (torch.rand(B, device=dev) - 0.5) * self.bin_w
            u_i = torch.stack([a, b], -1)
            logq = logq + la.gather(1, ba[:, None]).squeeze(1) + lb.gather(1, bb[:, None]).squeeze(1) - 2 * math.log(self.bin_w)
            placed_u = torch.cat([placed_u, u_i[:, None]], 1)
            placed_sp = torch.cat([placed_sp, sp[:, i:i + 1]], 1)
        xC_lab = KC.from_frame(placed_u, origin, R, L)                               # [B,k,2]
        return xC_lab, logq

    def log_q(self, pos, s, cluster_idx, xC_query, sc, L):          # grad-capable (training); MH wraps in no_grad
        B = pos.shape[0]; k = cluster_idx.shape[0]; dev = pos.device
        origin, R, ctx_tok, q_scaf = self._ctx_tokens(pos, s, cluster_idx, sc, L)
        sp = s[:, cluster_idx]
        u = KC.to_frame(xC_query, origin, R, L)                                      # [B,k,2] query in-frame
        ba_all = self._bin(u[..., 0]); bb_all = self._bin(u[..., 1])                 # [B,k] (-1 if out of box)
        placed_u = torch.zeros(B, 0, 2, device=dev); placed_sp = torch.zeros(B, 0, dtype=torch.long, device=dev)
        logq = torch.zeros(B, device=dev)
        for i in range(k):
            ctx = self._step_ctx(q_scaf[:, i], i, ctx_tok, self._placed_tok(placed_u, placed_sp))
            la = F.log_softmax(self.head_a(ctx), -1); ba = ba_all[:, i]
            lb = F.log_softmax(self.head_b(ctx + self.bin_a_emb(ba.clamp(0))), -1); bb = bb_all[:, i]
            ok = (ba >= 0) & (bb >= 0)
            step = (la.gather(1, ba.clamp(0)[:, None]).squeeze(1) + lb.gather(1, bb.clamp(0)[:, None]).squeeze(1)
                    - 2 * math.log(self.bin_w))
            logq = logq + torch.where(ok, step, torch.full_like(step, -69.0))        # genuine zero outside box
            placed_u = torch.cat([placed_u, u[:, i][:, None]], 1)
            placed_sp = torch.cat([placed_sp, sp[:, i:i + 1]], 1)
        return logq


def _scaffold(N, device):
    """Returns (sc, L, geo). geo provides _curve_order to put a config into slot order (slot j -> particle
    near sc[j]) -- REQUIRED before any cluster op (the cluster is only spatially local in slot order)."""
    from liquid_coupling_flow.ka_noncausal import NonCausalLF
    ck = torch.load(os.path.join(ART, "ka_noncausal_N100.pt"), map_location=device, weights_only=False)
    m = NonCausalLF(rho=1.2, n_bins=192, knn=ck["knn"], canonical=False).to(device)
    return m.geo._scaffold(N, device), m.geo._Lof(N), m.geo


def slot_order(pos, s, geo, N):
    """Curve-order a config -> (pos_ord [B,N,2], s_ord [B,N]) so slot j is the particle near scaffold j."""
    order = geo._curve_order(pos, N)
    pos_ord = torch.gather(pos, 1, order[..., None].expand(-1, -1, 2))
    s_ord = torch.gather(s.expand(pos.shape[0], N) if s.dim() == 1 else s, 1, order)
    return pos_ord, s_ord


def train(steps=15000, k=7, n_bins=24, box=3.0, n_ctx=16, lr=3e-4, train_N=100,
          device="cuda" if torch.cuda.is_available() else "cpu"):
    """Conditional MLE: maximize log q(true cluster | true surroundings) over random slot-clusters on the
    reference. A random seed slot per step (shared across the B configs); covers all clusters over training."""
    import time
    from liquid_coupling_flow.ka_gridformer_train import augment
    ref = torch.load(os.path.join(ART, f"ka_reference_N{train_N}.pt"), map_location=device, weights_only=False)
    data, s = ref["x"].to(device), ref["s"].to(device).long(); N = data.shape[1]
    sc, L, geo = _scaffold(N, device)
    P = ClusterProposal(rho=1.2, n_bins=n_bins, box=box, n_ctx=n_ctx).to(device).train()
    opt = torch.optim.AdamW(P.parameters(), lr=lr, weight_decay=1e-4); B, t0 = 128, time.time()
    warm = 400
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda st: min((st + 1) / warm,
        0.5 + 0.5 * math.cos(math.pi * max(0, st - warm) / max(1, steps - warm))))
    print(f"CLUSTER GENERATOR train @N={train_N}, {steps} steps, k={k}, n_bins={n_bins}, box={box}", flush=True)
    for step in range(steps):
        idx = torch.randint(0, data.shape[0], (B,), device=device)
        pos, s_ord = slot_order(augment(data[idx], L), s, geo, N)                    # curve-order -> slot order
        seed = int(torch.randint(0, N, (1,)).item()); cl = KC.cluster_slots(seed, sc, k, L)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(device == "cuda")):
            loss = -(P.log_q(pos, s_ord, cl, pos[:, cl], sc, L) / k).mean()
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(P.parameters(), 5.0); opt.step(); sched.step()
        if step % 1000 == 0:
            print(f"  step {step:5d} -logq/k {loss.item():.3f} lr {sched.get_last_lr()[0]:.1e} {time.time()-t0:.0f}s", flush=True)
        if (step + 1) % max(1, steps // 3) == 0:
            torch.save({"state_dict": P.state_dict(), "k": k, "n_bins": n_bins, "box": box, "n_ctx": n_ctx,
                        "step": step + 1}, os.path.join(ART, f"ka_cluster_flow_N{train_N}.pt"))
    torch.save({"state_dict": P.state_dict(), "k": k, "n_bins": n_bins, "box": box, "n_ctx": n_ctx, "step": steps},
               os.path.join(ART, f"ka_cluster_flow_N{train_N}.pt"))
    print(f"saved ka_cluster_flow_N{train_N}.pt", flush=True)


if __name__ == "__main__":
    import sys
    train(steps=int(sys.argv[1]) if len(sys.argv) > 1 else 15000,
          k=int(sys.argv[2]) if len(sys.argv) > 2 else 7)
