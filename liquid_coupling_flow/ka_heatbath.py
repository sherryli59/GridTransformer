"""Stage A2: beta-conditioned FULL-CAGE single-site heat-bath — the depth candidate.

Stage A1 proved a better LOCAL CLUSTER proposal does not close the SMC depth gap (the barrier is the glassy
tail, not cluster acceptance). The remaining lever is a conditional that sees MORE of the cage: a SINGLE-SITE
move conditioned on the FULL cage (all k neighbours at once, non-causal) — the sharp regime that reached g_BB
2.00 in the non-causal experiment ([[full-cage-lever-needs-energy]]) — made EXACT by an MH energy guard (the
guard whose absence made pure Gibbs collapse) and a FROZEN cage.

Construction (maximal reuse, inherited exactness): a `ClusterProposal` used at k=1 + a beta FiLM. The site is
placed in a frame anchored on `frame_ctx_slots([site])` — the scaffold-neighbour slots, a config-independent
function of (site, sc) — so the cage + frame depend on the OTHER particles only, NEVER on x_site. That is the
named frozen-cage invariant, and it is the SAME config-independence that makes the cluster move exact. Single
particle => no half-cage AR problem => full cage. Exact per-site spline density scores the reverse move.

Kernel: single-site MH, alpha = min(1, exp(-beta*dU) * q(x_i|cage,beta)/q(x_i'|cage,beta)), dU via ka_pair_row.
Train: beta-conditioned MLE on ALL Stage-0 ladder rungs (pt_ladder_N100.pt).
"""
from __future__ import annotations
import os, math, time, torch, torch.nn as nn
from liquid_coupling_flow import ka_cluster as KC
from liquid_coupling_flow.ka_cluster_flow import ClusterProposal, _scaffold, slot_order, ART
from liquid_coupling_flow.ka_energy import ka_pair_row


class HeatBathModel(ClusterProposal):
    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        d = self.sp_emb.embedding_dim
        self.beta_film = nn.Sequential(nn.Linear(1, d), nn.GELU(), nn.Linear(d, d))
        nn.init.zeros_(self.beta_film[-1].weight); nn.init.zeros_(self.beta_film[-1].bias)   # inert at init
        self._beta_ctx = None                                        # stashed [B,d] beta embedding

    def _ctx_at(self, *a, **k):
        ctx = super()._ctx_at(*a, **k)
        return ctx if self._beta_ctx is None else ctx + self._beta_ctx

    def _cl(self, site, device):
        return site if torch.is_tensor(site) else torch.tensor([site], device=device, dtype=torch.long)

    def _set_beta(self, beta, B, dtype, device):
        if not torch.is_tensor(beta):
            beta = torch.full((B,), float(beta), device=device)
        self._beta_ctx = self.beta_film(beta.reshape(-1, 1).to(dtype))

    @torch.no_grad()
    def sample_site(self, pos, s, site, sc, L, beta):
        self._set_beta(beta, pos.shape[0], pos.dtype, pos.device)
        try:
            xC, logq = self.sample(pos, s, self._cl(site, pos.device), sc, L)
        finally:
            self._beta_ctx = None
        return xC[:, 0], logq                                        # [B,2], [B]

    def log_q_site(self, pos, s, site, xi, sc, L, beta):
        """Grad-capable (training). Exact per-site density of xi under the frozen cage at temperature beta."""
        self._set_beta(beta, pos.shape[0], pos.dtype, pos.device)
        try:
            lq = self.log_q(pos, s, self._cl(site, pos.device), xi[:, None], sc, L)
        finally:
            self._beta_ctx = None
        return lq                                                    # [B]


@torch.no_grad()
def single_site_mh(model, pos, s, site, sc, L, beta, gen=None):
    """One single-site heat-bath MH move at slot `site` (pos/s slot-ordered). EXACT (frozen cage). Returns
    (pos_new, accept[B] bool). Only the site moves; species untouched."""
    B, dev = pos.shape[0], pos.device
    if not torch.is_tensor(beta):
        beta = torch.full((B,), float(beta), device=dev)
    xi_old = pos[:, site]                                            # [B,2]
    xi_new, logq_new = model.sample_site(pos, s, site, sc, L, beta)  # proposal + its density
    logq_old = model.log_q_site(pos, s, site, xi_old, sc, L, beta)   # reverse density, SAME frozen cage
    dU = ka_pair_row(pos, s, site, xi_new, L) - ka_pair_row(pos, s, site, xi_old, L)   # exact single-site dU
    log_alpha = -beta * dU + logq_old - logq_new
    u = torch.rand(B, device=dev, generator=gen)
    accept = torch.log(u) < log_alpha
    pos_new = pos.clone()
    pos_new[:, site] = torch.where(accept[:, None], xi_new, xi_old)
    return pos_new, accept


@torch.no_grad()
def single_site_mh_sweep(model, pos, s, sc, L, geo, beta=2.0, n_moves=None, gen=None):
    """A pi_beta-invariant single-site heat-bath sweep (valid SMC mutation kernel). Re-slot-order per move so
    slot `site` is the particle near scaffold slot `site` (its cage = scaffold neighbours). Species untouched.

    QUASI-ERGODICITY CAVEAT (measured): each move is pi-invariant (single-site DB exact; verified vs exact
    Gaussian-displacement on a frozen cage to 3e-3), BUT the beta-conditioned proposal is SHARPER than the true
    pi(x_i|cage) (negative training -logq), so as an INDEPENDENCE proposal it rarely suggests uphill moves ->
    STANDALONE it over-cools (~-3.30 vs true -3.276 at N=100 beta=2). MIX with an ergodic kernel (displacement)
    and it converges to the correct -3.276 AND fast. Never use standalone; always co-mutate (the SMC stack does)."""
    B, N, _ = pos.shape; dev = pos.device
    n_moves = N if n_moves is None else n_moves
    acc = []
    for _ in range(n_moves):
        site = int(torch.randint(0, N, (1,), device=dev, generator=gen).item())
        order = geo._curve_order(pos, N)
        pos_o = torch.gather(pos, 1, order[..., None].expand(-1, -1, 2))
        s_o = torch.gather(s, 1, order)
        pos_n, a = single_site_mh(model, pos_o, s_o, site, sc, L, beta, gen=gen)
        idx = order[:, site]
        pos = pos.clone()
        pos[torch.arange(B, device=dev)[:, None], idx[:, None]] = pos_n[:, site:site + 1]
        acc.append(a.float().mean().item())
    return pos, {"accept": sum(acc) / len(acc)}


def _load(ck, device):
    m = HeatBathModel(rho=1.2, n_bins=ck["n_bins"], box=ck["box"], n_ctx=ck["n_ctx"],
                      d_model=ck.get("d_model", 192), n_head=ck.get("n_head", 6),
                      n_layer=ck.get("n_layer", 4), head=ck.get("head", "spline"),
                      num_flow_bins=ck.get("num_flow_bins", 16), tail_bound=ck.get("tail_bound", 3.5),
                      pair_feats=ck.get("pair_feats", True)).to(device).eval()
    m.load_state_dict(ck["state_dict"]); return m


def train(steps=20000, n_ctx=24, d_model=192, n_head=6, n_layer=4, lr=3e-4, train_N=100,
          num_flow_bins=16, tail_bound=3.5, pair_feats=True, tag="", B=128,
          device="cuda" if torch.cuda.is_available() else "cpu"):
    """beta-conditioned single-site MLE over ALL Stage-0 ladder rungs: maximize log q(true x_site | cage, beta).
    A random rung (=> beta) and a random site per step; covers all sites and temperatures over training."""
    from liquid_coupling_flow.ka_gridformer_train import augment
    lad = torch.load(os.path.join(ART, f"pt_ladder_N{train_N}.pt"), map_location=device, weights_only=False)
    betas = torch.tensor(lad["betas"], device=device)                # [M] cold->hot
    rungs = [c.to(device) for c in lad["configs_per_rung"]]          # list[M] of [n,N,2]
    s_all = lad["s"].to(device).long()                              # [N] shared species labelling
    M = len(rungs); N = rungs[0].shape[1]
    sc, L, geo = _scaffold(N, device)
    arch = {"n_bins": 24, "box": 3.0, "n_ctx": n_ctx, "d_model": d_model, "n_head": n_head, "n_layer": n_layer,
            "head": "spline", "num_flow_bins": num_flow_bins, "tail_bound": tail_bound, "pair_feats": pair_feats}
    m = HeatBathModel(rho=1.2, **arch).to(device).train()
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=1e-4); t0 = time.time()
    warm = 400
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda st: min((st + 1) / warm,
        0.5 + 0.5 * math.cos(math.pi * max(0, st - warm) / max(1, steps - warm))))
    nparam = sum(p.numel() for p in m.parameters())
    print(f"HEATBATH train @N={train_N}, {steps} steps, {M} rungs, {arch}, params {nparam/1e6:.2f}M", flush=True)
    for step in range(steps):
        r = int(torch.randint(0, M, (1,)).item()); data = rungs[r]; beta_r = betas[r]
        idx = torch.randint(0, data.shape[0], (B,), device=device)
        pos, s_ord = slot_order(augment(data[idx], L), s_all, geo, N)
        site = int(torch.randint(0, N, (1,)).item())
        loss = -m.log_q_site(pos, s_ord, site, pos[:, site], sc, L, beta_r.expand(B)).mean()
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 5.0)
        opt.step(); sched.step()
        if step % 1000 == 0:
            print(f"  step {step:5d} -logq {loss.item():.3f} (rung {r} beta {beta_r:.2f}) "
                  f"lr {sched.get_last_lr()[0]:.1e} {time.time()-t0:.0f}s", flush=True)
        if (step + 1) % max(1, steps // 3) == 0:
            torch.save({"state_dict": m.state_dict(), "step": step + 1, "betas": lad["betas"], **arch},
                       os.path.join(ART, f"ka_heatbath{tag}_N{train_N}.pt"))
    torch.save({"state_dict": m.state_dict(), "step": steps, "betas": lad["betas"], **arch},
               os.path.join(ART, f"ka_heatbath{tag}_N{train_N}.pt"))
    print(f"saved ka_heatbath{tag}_N{train_N}.pt", flush=True)


if __name__ == "__main__":
    import sys
    train(steps=int(sys.argv[1]) if len(sys.argv) > 1 else 20000)
