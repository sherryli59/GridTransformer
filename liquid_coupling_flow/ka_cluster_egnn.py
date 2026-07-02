"""EGNN cluster flow: conditional continuous normalizing flow placing the k cluster particles under the periodic
traceable-EGNN central-force velocity (all-see-all, no coupling blind spot), cage fixed context, frame-free.
See docs/superpowers/specs/2026-06-30-egnn-cluster-flow-design.md. Interface mirrors ClusterProposal."""
from __future__ import annotations
import os, math, torch, torch.nn as nn
from liquid_coupling_flow import ka_cluster as KC
from liquid_coupling_flow.ka_cluster_flow import slot_order, _scaffold
from liquid_coupling_flow.ka_gridformer import _wrap_pm

ART = os.path.join(os.path.dirname(__file__), "artifacts")


def _t(t, dev, ref):
    return torch.tensor(float(t), device=dev, dtype=ref.dtype)


def cage_centroid(cage, L):                                  # cage [B,m,2] -> [B,2] min-image mean on cage[:,0]
    a = cage[:, :1]
    return torch.remainder(a[:, 0] + _wrap_pm(cage - a, L).mean(1), L)


def build_cloud(pos, s, cluster_idx, sc, L, n_cage):
    """cloud = [cluster (k); nearest n_cage non-cluster particles]. Returns (cloud[B,P,2], sp[B,P], c[B,2])."""
    B, N, _ = pos.shape; dev = pos.device; k = cluster_idx.shape[0]
    clu = pos[:, cluster_idx]                                                     # [B,k,2]
    # cluster centroid (min-image on the seed)
    seed = clu[:, :1]                                                             # [B,1,2]
    ccen = torch.remainder(seed[:, 0] + _wrap_pm(clu - seed, L).mean(1), L)       # [B,2]
    d = _wrap_pm(pos - ccen[:, None], L).norm(dim=-1)                             # [B,N] dist to cluster centroid
    d[:, cluster_idx] = float("inf")                                             # exclude the cluster
    cage_idx = d.topk(n_cage, largest=False).indices                             # [B,n_cage]
    cage = torch.gather(pos, 1, cage_idx[..., None].expand(-1, -1, 2))            # [B,n_cage,2]
    cage_sp = torch.gather(s, 1, cage_idx)                                        # [B,n_cage]
    cloud = torch.cat([clu, cage], 1)                                            # [B,P,2]
    sp = torch.cat([s[:, cluster_idx], cage_sp], 1)                              # [B,P]
    # n_cage=0 (no-context ablation): base anchor falls back to the CLUSTER centroid (cage_centroid of an
    # empty cage is NaN). The anchor leaks the true position -> nocage results speak ONLY to intra exclusion.
    return cloud, sp, (ccen if n_cage == 0 else cage_centroid(cage, L))


def base_logp(x0, c, sigma_b, L):                          # x0 [B,k,2], c [B,2] -> [B]
    d2 = (_wrap_pm(x0 - c[:, None], L) ** 2).sum(-1)        # MIN-IMAGE: base is a torus Gaussian at c
    return (-d2 / (2 * sigma_b ** 2) - math.log(2 * math.pi * sigma_b ** 2)).sum(-1)


def sample_base(c, k, sigma_b, gen=None):                  # -> [B,k,2] full 2k-dim Gaussian at c (no zero-COM)
    return c[:, None] + sigma_b * torch.randn(c.shape[0], k, 2, generator=gen, device=c.device, dtype=c.dtype)


@torch.no_grad()
def compute_sigma_b(data, s, geo, sc, L, k, n_cage):
    """Isotropic std of the true cluster particles' min-image displacement from the cluster centroid."""
    N = data.shape[1]; pos, s_ord = slot_order(data, s, geo, N); disp = []
    for seed in range(0, N, 3):
        cl = KC.cluster_slots(seed, sc, k, L); clu = pos[:, cl]
        sd = clu[:, :1]
        ccen = torch.remainder(sd[:, 0] + _wrap_pm(clu - sd, L).mean(1), L)
        disp.append(_wrap_pm(clu - ccen[:, None], L).reshape(-1, 2))
    return float(torch.cat(disp, 0).std().item())


from liquid_coupling_flow.ipl44.learndiffeq.learndiffeq.particles.velocities.egnn_traceable import EGNN_dynamics


class ConditionalEGNN(nn.Module):
    """Wraps a periodic EGNN_dynamics over the fixed cloud (k cluster + n_cage cage). vel_div returns ONLY the
    cluster velocities and the cluster-restricted divergence (cage is fixed context -> not in the log-det)."""
    def __init__(self, n_cage=48, k=7, r_c=3.0, L=None, hidden_nf=64, n_layers=4, n_species=2, max_neighbors=None,
                 rep_prior=False):
        super().__init__()
        P = k + n_cage
        mn = P - 1 if max_neighbors is None else min(int(max_neighbors), P - 1)   # None -> full cloud; else k-NN
        self.egnn = EGNN_dynamics(n_particles=P, n_dimension=2, cutoff=r_c, max_neighbors=mn,
                                  L=L, n_species=n_species, hidden_nf=hidden_nf, n_layers=n_layers,
                                  rep_prior=rep_prior)

    def vel_div(self, cloud, t, sp, k):
        vel, divpp = self.egnn.forward_and_perparticle_divergence(cloud, t, sp)   # [B,P,2],[B,P]
        return vel[:, :k], divpp[:, :k].sum(-1)                                    # cluster velocities + div


class EGNNClusterFlow(nn.Module):
    """Conditional CNF: the k cluster particles move under the periodic traceable-EGNN central-force velocity;
    the cage (fixed context) does not. sample: t 0->1 (base->data). log_q: t 1->0 (data->base), recovering the
    base point and its exact log-density via the cluster-restricted divergence. NOT translation/rotation invariant
    for the cluster alone (design keystone): the cage is a fixed reference that pins the cluster's absolute
    position, so there is no zero-COM subspace and no frame."""

    def __init__(self, sigma_b, n_cage=48, k=7, r_c=3.0, L=None, hidden_nf=64, n_layers=4, n_steps=16, n_species=2,
                 max_neighbors=None, rep_prior=False, integrator="rk4", n_picard=32, picard_tol=1e-9):
        super().__init__()
        self.sigma_b = float(sigma_b); self.n_cage = n_cage; self.k = k; self.n_steps = n_steps
        # integrator: "rk4" (default, 4th-order, NOT time-reversible -> sample==log_q holds only to integration
        # accuracy, ~4e-3 at n_steps=32) or "midpoint" (implicit midpoint, TIME-REVERSIBLE: div evaluated at the
        # shared min-image midpoint so forward/reverse log-dets cancel to the inner-solve accuracy). REQUIREMENTS
        # for the reversible win (measured: 4e-14 vs RK4 4e-3): (1) DOUBLE precision — float32 floors the Picard
        # solve at ~1e-7 and roundoff accumulates (~6e-3, no better than RK4); (2) enough steps that Picard
        # CONTRACTS (h*Lip(v) < 1) — n_steps>=32 here converges, n_steps=16 does NOT (residual ~3e-2). NOT suited
        # to a stiff field (rep_prior on): Picard diverges -> use rk4 or an implicit/Newton solve there. This is
        # the exact-log_q path for the prior-OFF model (run it in double). Not a trained parameter (integration-
        # time knob); set post-load like n_steps if desired.
        self.integrator = integrator; self.n_picard = n_picard; self.picard_tol = picard_tol
        self.ce = ConditionalEGNN(n_cage=n_cage, k=k, r_c=r_c, L=L, hidden_nf=hidden_nf, n_layers=n_layers,
                                  n_species=n_species, max_neighbors=max_neighbors, rep_prior=rep_prior)
        with torch.no_grad():                              # flow-matching init: untrained velocity == 0 (identity, tame)
            self.ce.egnn.pot_model[-1].weight.zero_(); self.ce.egnn.pot_model[-1].bias.zero_()

    def _midpoint_step(self, cl, cage, sp, k, L, t, h, dev, cloud):
        """One implicit-midpoint step, time-reversible. Solves x_next = cl + h*v(mid) with mid = the MIN-IMAGE
        midpoint of (cl, x_next) by Picard iteration (velocity-only) to picard_tol, then evaluates the divergence
        ONCE at the converged midpoint. The min-image midpoint is symmetric in its two endpoints, so the reverse
        step (-h from x_next) reconstructs the identical mid -> logdet_fwd and logdet_rev cancel to Picard
        tolerance. Returns (x_next[B,k,2], h*div_mid[B])."""
        tm = _t(t + 0.5 * h, dev, cloud)
        x_next = cl
        for _ in range(self.n_picard):
            mid = torch.remainder(cl + 0.5 * _wrap_pm(x_next - cl, L), L)
            v = self.ce.egnn.forward(tm, torch.cat([mid, cage], 1), sp)[:, :k]
            x_new = torch.remainder(cl + h * v, L)
            if _wrap_pm(x_new - x_next, L).abs().max() < self.picard_tol:
                x_next = x_new; break
            x_next = x_new
        mid = torch.remainder(cl + 0.5 * _wrap_pm(x_next - cl, L), L)     # converged shared midpoint
        _, div_mid = self.ce.vel_div(torch.cat([mid, cage], 1), tm, sp, k)
        return x_next, h * div_mid

    def _integrate(self, cloud, sp, k, L, reverse):
        """Integrate the cluster (first k) over t; cage (rest) fixed. reverse=False: t 0->1 (base->data),
        accumulate -div; reverse=True: t 1->0 (data->base), accumulate +div. Returns (cluster[B,k,2], logdet).
        Integrator per self.integrator: 'rk4' (4th-order) or 'midpoint' (reversible)."""
        dev = cloud.device; B = cloud.shape[0]; dt = 1.0 / self.n_steps
        logdet = torch.zeros(B, device=dev, dtype=cloud.dtype)
        cl = cloud[:, :k]; cage = cloud[:, k:]
        for i in range(self.n_steps):
            t = 1.0 - i * dt if reverse else i * dt
            h = -dt if reverse else dt
            if self.integrator == "midpoint":
                cl, dstep = self._midpoint_step(cl, cage, sp, k, L, t, h, dev, cloud)
                logdet = logdet + dstep
                continue
            # RK4 the AUGMENTED ODE [x, logdet]: div is computed at all 4 stages (O(dt^4), not Euler) so the
            # forward/reverse log-det cancels to integration precision -> sampler==scorer holds.
            v1, d1 = self.ce.vel_div(torch.cat([cl, cage], 1), _t(t, dev, cloud), sp, k)
            v2, d2 = self.ce.vel_div(torch.cat([torch.remainder(cl + 0.5 * h * v1, L), cage], 1), _t(t + 0.5 * h, dev, cloud), sp, k)
            v3, d3 = self.ce.vel_div(torch.cat([torch.remainder(cl + 0.5 * h * v2, L), cage], 1), _t(t + 0.5 * h, dev, cloud), sp, k)
            v4, d4 = self.ce.vel_div(torch.cat([torch.remainder(cl + h * v3, L), cage], 1), _t(t + h, dev, cloud), sp, k)
            cl = torch.remainder(cl + (h / 6.0) * (v1 + 2 * v2 + 2 * v3 + v4), L)
            logdet = logdet + (h / 6.0) * (d1 + 2 * d2 + 2 * d3 + d4)     # ∫ div dt (sign folded into h)
        return cl, logdet

    @torch.no_grad()
    def sample(self, pos, s, cluster_idx, sc, L):
        cloud, sp, c = build_cloud(pos, s, cluster_idx, sc, L, self.n_cage)
        z = sample_base(c, self.k, self.sigma_b)
        cloud = torch.cat([z, cloud[:, self.k:]], 1)
        cl, logdet = self._integrate(cloud, sp, self.k, L, reverse=False)
        logq = base_logp(z, c, self.sigma_b, L) - logdet  # d log p = -div dt over the forward pass
        return cl, logq

    @torch.no_grad()
    def sample_fast(self, pos, s, cluster_idx, sc, L):
        """Velocity-only sampling (NO divergence/logdet) -> just positions. For the g(r) gate (structural
        clash/g_BB), which never uses logq -> ~10x faster than sample() (skips the per-stage autograd divergence)."""
        cloud, sp, c = build_cloud(pos, s, cluster_idx, sc, L, self.n_cage)
        cl = sample_base(c, self.k, self.sigma_b); cage = cloud[:, self.k:]; dt = 1.0 / self.n_steps
        def vo(x, t):
            return self.ce.egnn.forward(_t(t, x.device, x), torch.cat([x, cage], 1), sp)[:, :self.k]
        for i in range(self.n_steps):
            t = i * dt
            v1 = vo(cl, t)
            v2 = vo(torch.remainder(cl + 0.5 * dt * v1, L), t + 0.5 * dt)
            v3 = vo(torch.remainder(cl + 0.5 * dt * v2, L), t + 0.5 * dt)
            v4 = vo(torch.remainder(cl + dt * v3, L), t + dt)
            cl = torch.remainder(cl + (dt / 6.0) * (v1 + 2 * v2 + 2 * v3 + v4), L)
        return cl

    @torch.no_grad()                                      # log_q is a value (gate/MH/IS); training uses flow matching
    def log_q(self, pos, s, cluster_idx, xC_query, sc, L):
        cloud, sp, c = build_cloud(pos, s, cluster_idx, sc, L, self.n_cage)
        cloud = torch.cat([xC_query, cloud[:, self.k:]], 1)
        z, logdet = self._integrate(cloud, sp, self.k, L, reverse=True)
        return base_logp(z, c, self.sigma_b, L) + logdet


from scipy.optimize import linear_sum_assignment


def ot_assign(z, x, sp, L):
    """Species-aware per-particle OT: Hungarian on the min-image squared cost, cross-species forbidden. -> perm[B,k]
    such that base particle i flows to data particle perm[i] (same species), minimizing total transport."""
    B, k, _ = z.shape; dev = z.device
    d = _wrap_pm(z[:, :, None, :] - x[:, None, :, :], L)                          # [B,k,k,2]  z_i vs x_j
    cost = (d ** 2).sum(-1)                                                       # [B,k,k]
    cost = cost.masked_fill(sp[:, :, None] != sp[:, None, :], 1e6)                # forbid cross-species matches
    cc = cost.detach().cpu().numpy(); perms = []
    for b in range(B):
        _, col = linear_sum_assignment(cc[b]); perms.append(torch.as_tensor(col, device=dev))
    return torch.stack(perms, 0)                                                  # [B,k]


_ARCH = ("n_cage", "k", "r_c", "hidden_nf", "n_layers", "n_steps", "n_species", "max_neighbors", "rep_prior")


def load_flow(ck, device):
    arch = {kk: ck[kk] for kk in _ARCH if kk in ck}
    P = EGNNClusterFlow(sigma_b=ck["sigma_b"], L=ck["L"], **arch).to(device).eval()
    P.load_state_dict(ck["state_dict"]); return P


def train(steps=15000, k=7, n_cage=48, r_c=3.0, hidden_nf=64, n_layers=4, n_steps=16, lr=3e-4, train_N=100,
          save=True, batch=128, ckpt_every=2000, max_neighbors=24, tag="", n_configs=None, fix_seed=None,
          use_augment=True, resume=False, rep_prior=False,
          device="cuda" if torch.cuda.is_available() else "cpu"):
    """OT conditional flow matching: regress the EGNN velocity onto the OT-straightened base->data field. No ODE
    integration in the loop (velocity-only forward) -> cheap + stable; the exact log_q is inference-only.
    max_neighbors=24 (k-NN) keeps the all-see-all intra-cluster fix (co-cluster particles are the nearest) but
    cuts the O(P^2) memory -> big batch. None -> full cloud."""
    import time
    from liquid_coupling_flow.ka_gridformer_train import augment
    ref = torch.load(os.path.join(ART, f"ka_reference_N{train_N}.pt"), map_location=device, weights_only=False)
    data, s = ref["x"].to(device), ref["s"].to(device).long()
    if n_configs is not None:
        data = data[:n_configs]                      # overfit probe: memorize a fixed tiny subset
    N = data.shape[1]
    sc, L, geo = _scaffold(N, device); sigma_b = compute_sigma_b(data, s, geo, sc, L, k, n_cage)
    flow = EGNNClusterFlow(sigma_b=sigma_b, n_cage=n_cage, k=k, r_c=r_c, L=L, hidden_nf=hidden_nf,
                           n_layers=n_layers, n_steps=n_steps, max_neighbors=max_neighbors,
                           rep_prior=rep_prior).to(device).train()
    opt = torch.optim.AdamW(flow.parameters(), lr=lr, weight_decay=1e-4); Bsz = batch
    arch = dict(n_cage=n_cage, k=k, r_c=r_c, hidden_nf=hidden_nf, n_layers=n_layers, n_steps=n_steps, n_species=2,
                max_neighbors=max_neighbors, rep_prior=rep_prior)
    ckpt = os.path.join(ART, f"ka_cluster_egnn{tag}_N{train_N}.pt")
    if resume and os.path.exists(ckpt):
        old = torch.load(ckpt, map_location=device, weights_only=False)
        flow.load_state_dict(old["state_dict"]); print(f"resumed {ckpt} @ step {old['step']}", flush=True)
    print(f"EGNN-FLOW train N={train_N} steps={steps} k={k} n_cage={n_cage} batch={Bsz} sigma_b={sigma_b:.3f} "
          f"params {sum(p.numel() for p in flow.parameters())/1e6:.2f}M", flush=True)
    loss_first = None; t0 = time.time()
    for step in range(steps):
        idx = torch.randint(0, data.shape[0], (Bsz,), device=device)
        pos, s_ord = slot_order(augment(data[idx], L) if use_augment else data[idx], s, geo, N)
        seed = fix_seed if fix_seed is not None else int(torch.randint(0, N, (1,)).item())
        cl = KC.cluster_slots(seed, sc, k, L)
        cloud, sp, c = build_cloud(pos, s_ord, cl, sc, L, n_cage); x1 = cloud[:, :k]
        z = sample_base(c, k, sigma_b)                                           # base
        perm = ot_assign(z, x1, sp[:, :k], L)                                    # species-aware per-particle OT
        x1p = torch.gather(x1, 1, perm[..., None].expand(-1, -1, 2))             # OT-matched data
        target = _wrap_pm(x1p - z, L)                                            # straight displacement (min-image)
        t = torch.rand(Bsz, device=device)
        xt = torch.remainder(z + t[:, None, None] * target, L)                   # interpolant
        v_pred = flow.ce.egnn.forward(t, torch.cat([xt, cloud[:, k:]], 1), sp)[:, :k]   # velocity-only (fast)
        loss = ((v_pred - target) ** 2).sum(-1).mean()
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(flow.parameters(), 5.0); opt.step()
        if loss_first is None: loss_first = loss.item()
        if step % 1000 == 0:
            print(f"  step {step:5d} fm-loss {loss.item():.4f} {time.time()-t0:.0f}s", flush=True)
        if save and (step + 1) % ckpt_every == 0:
            torch.save({"state_dict": flow.state_dict(), "sigma_b": sigma_b, "L": L, "step": step + 1,
                        "loss_first": loss_first, "loss_last": loss.item(), **arch}, ckpt)
    ck = {"state_dict": flow.state_dict(), "sigma_b": sigma_b, "L": L, "step": steps,
          "loss_first": loss_first, "loss_last": loss.item(), **arch}
    if save:
        torch.save(ck, ckpt); print(f"saved {ckpt}", flush=True)
    return ck


@torch.no_grad()
def gate(train_N=100, k=7, device="cuda" if torch.cuda.is_available() else "cpu", Bsz=128):
    from liquid_coupling_flow.ka_cluster_flow import gate_measure
    ref = torch.load(os.path.join(ART, f"ka_reference_N{train_N}.pt"), map_location=device, weights_only=False)
    s = ref["s"].to(device).long(); N = ref["x"].shape[1]; sc, L, geo = _scaffold(N, device)
    ck = torch.load(os.path.join(ART, f"ka_cluster_egnn_N{train_N}.pt"), map_location=device, weights_only=False)
    P = load_flow(ck, device); P.n_steps = 6              # coarse RK4 for the gate (structural clash/g_BB only)
    pos0, sso = slot_order(ref["x"][:Bsz].to(device), s, geo, N)
    out = os.path.join(ART, f"ka_cluster_egnn_gate_N{train_N}.png")

    class _Fast:                                          # gate_measure only uses .sample(...)[0] (positions)
        def __init__(s, f): s.f = f
        def sample(s, pos, sp, cl, sc, L): return s.f.sample_fast(pos, sp, cl, sc, L), None
    res = gate_measure(_Fast(P), pos0, sso, sc, L, N, k, out_png=out)
    print("GATE(egnn-flow):", res, flush=True); print("saved", out, flush=True)


if __name__ == "__main__":
    import sys
    mode = sys.argv[1] if len(sys.argv) > 1 else ""
    if mode == "gate":
        gate()
    elif mode == "nocage":       # no-context ablation: k=7 alone, base at the true cluster centroid
        train(steps=15000, n_cage=0, max_neighbors=None, batch=256, tag="_nocage")
    elif mode == "overfit":      # capacity probe: memorize 4 configs x 1 cluster, no augmentation
        train(steps=6000, batch=64, n_configs=4, fix_seed=5, use_augment=False, tag="_overfit")
    else:
        train(steps=int(sys.argv[1]) if len(sys.argv) > 1 else 15000,
              batch=int(sys.argv[2]) if len(sys.argv) > 2 else 128)
