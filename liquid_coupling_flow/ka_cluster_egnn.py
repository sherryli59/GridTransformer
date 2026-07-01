"""EGNN cluster flow: conditional continuous normalizing flow placing the k cluster particles under the periodic
traceable-EGNN central-force velocity (all-see-all, no coupling blind spot), cage fixed context, frame-free.
See docs/superpowers/specs/2026-06-30-egnn-cluster-flow-design.md. Interface mirrors ClusterProposal."""
from __future__ import annotations
import os, math, torch, torch.nn as nn
from liquid_coupling_flow import ka_cluster as KC
from liquid_coupling_flow.ka_cluster_flow import slot_order, _scaffold
from liquid_coupling_flow.ka_gridformer import _wrap_pm

ART = os.path.join(os.path.dirname(__file__), "artifacts")


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


def base_logp(x0, c, sigma_b):                              # x0 [B,k,2], c [B,2] -> [B]
    d2 = ((x0 - c[:, None]) ** 2).sum(-1)
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
