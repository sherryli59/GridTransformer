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
    return cloud, sp, cage_centroid(cage, L)


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
    def __init__(self, n_cage=48, k=7, r_c=3.0, L=None, hidden_nf=64, n_layers=4, n_species=2):
        super().__init__()
        P = k + n_cage
        self.egnn = EGNN_dynamics(n_particles=P, n_dimension=2, cutoff=r_c, max_neighbors=P - 1,
                                  L=L, n_species=n_species, hidden_nf=hidden_nf, n_layers=n_layers)

    def vel_div(self, cloud, t, sp, k):
        vel, divpp = self.egnn.forward_and_perparticle_divergence(cloud, t, sp)   # [B,P,2],[B,P]
        return vel[:, :k], divpp[:, :k].sum(-1)                                    # cluster velocities + div


class EGNNClusterFlow(nn.Module):
    """Conditional CNF: the k cluster particles move under the periodic traceable-EGNN central-force velocity;
    the cage (fixed context) does not. sample: t 0->1 (base->data). log_q: t 1->0 (data->base), recovering the
    base point and its exact log-density via the cluster-restricted divergence. NOT translation/rotation invariant
    for the cluster alone (design keystone): the cage is a fixed reference that pins the cluster's absolute
    position, so there is no zero-COM subspace and no frame."""

    def __init__(self, sigma_b, n_cage=48, k=7, r_c=3.0, L=None, hidden_nf=64, n_layers=4, n_steps=16, n_species=2):
        super().__init__()
        self.sigma_b = float(sigma_b); self.n_cage = n_cage; self.k = k; self.n_steps = n_steps
        self.ce = ConditionalEGNN(n_cage=n_cage, k=k, r_c=r_c, L=L, hidden_nf=hidden_nf, n_layers=n_layers,
                                  n_species=n_species)
        with torch.no_grad():                              # flow-matching init: untrained velocity == 0 (identity, tame)
            self.ce.egnn.pot_model[-1].weight.zero_(); self.ce.egnn.pot_model[-1].bias.zero_()

    def _integrate(self, cloud, sp, k, L, reverse):
        """RK4 integrate the cluster (first k) over t; cage (rest) fixed. reverse=False: t 0->1 (base->data),
        accumulate -div; reverse=True: t 1->0 (data->base), accumulate +div. Returns (cluster[B,k,2], logdet)."""
        dev = cloud.device; B = cloud.shape[0]; dt = 1.0 / self.n_steps
        logdet = torch.zeros(B, device=dev, dtype=cloud.dtype)
        cl = cloud[:, :k]; cage = cloud[:, k:]
        for i in range(self.n_steps):
            t = 1.0 - i * dt if reverse else i * dt
            h = -dt if reverse else dt
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

    @torch.no_grad()                                      # log_q is a value (gate/MH/IS); training uses flow matching
    def log_q(self, pos, s, cluster_idx, xC_query, sc, L):
        cloud, sp, c = build_cloud(pos, s, cluster_idx, sc, L, self.n_cage)
        cloud = torch.cat([xC_query, cloud[:, self.k:]], 1)
        z, logdet = self._integrate(cloud, sp, self.k, L, reverse=True)
        return base_logp(z, c, self.sigma_b, L) + logdet
