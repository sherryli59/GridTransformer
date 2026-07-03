"""Joint species-position flow (v1, minimal standalone): torus-ODE positions + count-preserving swap-CTMC
species, one shared EGNN trunk with two heads. See docs/superpowers/specs/2026-06-30-joint-species-position-
flow-design.md. Reuses the vendored learndiffeq EGNN message passing WITHOUT modifying it: we run the GCL
layers manually to tap the hidden node features (which EGNN_dynamics discards) for a species denoiser head,
while the coordinate output gives the position velocity. Species stay exactly 22:22 at every t (Kawasaki)."""
from __future__ import annotations
import math, torch, torch.nn as nn, torch.nn.functional as F
from learndiffeq.particles.velocities.egnn import EGNN_dynamics, log_map, exp_map, dist_sq_torus
from learndiffeq.particles.utils import linear_assignment_permutation


class JointSpeciesFlow(nn.Module):
    """Shared EGNN trunk -> (position velocity v, per-particle species logits)."""

    def __init__(self, n_particles, L, hidden_nf=32, n_layers=3, n_species=2, tanh=True, attention=True,
                 two_time=False):
        super().__init__()
        self.N = n_particles; self.L = float(L); self.n_species = n_species; self.two_time = two_time
        self.dyn = EGNN_dynamics(n_particles=n_particles, n_dimension=2, hidden_nf=hidden_nf, n_layers=n_layers,
                                 recurrent=True, attention=attention, condition_time=True, tanh=tanh, agg="sum",
                                 L=float(L), n_species=n_species)
        self.species_head = nn.Sequential(nn.Linear(hidden_nf, hidden_nf), nn.SiLU(), nn.Linear(hidden_nf, n_species))
        if two_time:
            # decoupled species-noise clock: injected additively after the EGNN input projection so the
            # vendored EGNN stays untouched. Enables querying (t_pos=1, t_spec=0) = "clean geometry, garbage
            # labels" -> an s-independent geometry table (the coupled-t model is OOD there: t=0 also noises x).
            self.tspec_proj = nn.Linear(1, hidden_nf)

    def forward(self, t, x, s, t_spec=None):
        """t (=t_pos) [B,1,1], x [B,N,2], s [B,N] long, t_spec [B,1,1] (two_time only; default = t)
        -> v [B,N,2], logits [B,N,n_species]."""
        dyn = self.dyn; B = x.shape[0]; N = self.N; dev = x.device
        edges = dyn._cast_edges2batch(dyn.edges, B, N); edges = [edges[0].to(dev), edges[1].to(dev)]
        xf = x.reshape(B * N, 2)
        h_t = t.expand(-1, N, -1).reshape(B * N, 1)                       # time scalar per node
        h = torch.cat([h_t, dyn.species_embedding(s).view(B * N, -1)], dim=-1)
        edge_attr = dist_sq_torus(xf[edges[0]], xf[edges[1]], self.L).unsqueeze(-1)
        egnn = dyn.egnn
        hh = egnn.embedding(h); xx = xf.clone()
        if self.two_time:
            ts = t if t_spec is None else t_spec
            hh = hh + self.tspec_proj(ts.expand(-1, N, -1).reshape(B * N, 1))
        for i in range(egnn.n_layers):
            hh, xx, _ = egnn._modules["gcl_%d" % i](hh, edges, xx, edge_attr=edge_attr)
        vel = log_map(xf, xx, self.L).view(B, N, 2)                       # torus velocity = coordinate update
        logits = self.species_head(hh).view(B, N, self.n_species)
        return vel, logits

    # ---- generation: Euler ODE for x + count-preserving swap tau-leap for s ----
    @torch.no_grad()
    def sample(self, x0, s0, n_steps=250, n_inner=4, swap_scale=1.0, t_eps=1e-3):
        """x0 [B,N,2] (torus), s0 [B,N] long (exactly 22:22). Returns x1 [B,N,2], s1 [B,N] long (still 22:22)."""
        B, N = x0.shape[0], self.N; dev = x0.device
        x = x0.clone(); s = s0.clone(); dt = 1.0 / n_steps
        ar = torch.arange(B, device=dev)
        for k in range(n_steps):
            tk = torch.full((B, 1, 1), k * dt, device=dev)
            v, logits = self.forward(tk, x, s)
            x = exp_map(x, dt * v, self.L)
            pB = F.softmax(logits, dim=-1)[..., 1]                        # P(final species = B) per particle
            pA = 1.0 - pB
            inv = 1.0 / max(1.0 - k * dt, t_eps)
            for _ in range(n_inner):                                      # a few count-preserving swap attempts
                isA = (s == 0); isB = (s == 1)
                wA = (pB * isA) + 1e-12                                   # an A that wants to be B
                wB = (pA * isB) + 1e-12                                   # a B that wants to be A
                iA = torch.multinomial(wA, 1).squeeze(1)                  # [B]
                iB = torch.multinomial(wB, 1).squeeze(1)
                rate = pB[ar, iA] * pA[ar, iB] * inv
                do = (torch.rand(B, device=dev) < (rate * dt * swap_scale).clamp(0, 1))
                do = do & (s[ar, iA] == 0) & (s[ar, iB] == 1)             # guard: must be a real A-B pair
                s[ar[do], iA[do]] = 1; s[ar[do], iB[do]] = 0             # swap -> count preserved
        return torch.remainder(x, self.L), s


# ---------- conditional paths + training targets (C2/C3) ----------
def global_position_ot(x0, x1, L):
    """Species-agnostic batched OT pairing of base->target positions. Returns x0 permuted to align to x1."""
    return linear_assignment_permutation(x0, x1, L=L)[0]


def random_22_labeling(B, N, nB, device):
    """Uniform-random labeling with exactly nB ones per config."""
    base = torch.zeros(B, N, dtype=torch.long, device=device); base[:, :nB] = 1
    perm = torch.argsort(torch.rand(B, N, device=device), dim=1)
    return torch.gather(base, 1, perm)


def kawasaki_interpolate(s0, s1, t, u=None):
    """Count-preserving species path: each pending A<->B mismatch (matched in equal-size sets) is resolved with
    prob t. Returns s_t [B,N] long, exactly 22:22 at every t. t is [B] in [0,1]. Vectorized (no per-config
    loop); `u` [B,N] optional for deterministic testing against the reference implementation."""
    B, N = s0.shape; dev = s0.device
    st = s0.clone()
    AtoB = (s0 == 0) & (s1 == 1)                                          # pending A->B per config
    BtoA = (s0 == 1) & (s1 == 0)                                          # pending B->A (equal count to AtoB)
    # order the pending sets within each config; pair the k-th A->B with the k-th B->A; resolve pair iff u<t
    oa = torch.argsort(AtoB.float(), dim=1, descending=True, stable=True)  # pending-A indices first
    ob = torch.argsort(BtoA.float(), dim=1, descending=True, stable=True)
    na = AtoB.sum(1)                                                      # = nb per config
    if u is None:
        u = torch.rand(B, N, device=dev)                                 # one uniform per pending pair slot
    valid = torch.arange(N, device=dev)[None, :] < na[:, None]           # slot k active iff k < na[b]
    sel = (u < t[:, None]) & valid                                        # [B,N] resolved slots
    rows = torch.arange(B, device=dev)[:, None].expand(B, N)[sel]
    ai = oa[sel]; bi = ob[sel]
    st[rows, ai] = s1[rows, ai]                                           # resolved A->B
    st[rows, bi] = s1[rows, bi]                                           # resolved B->A
    return st


def _kawasaki_interpolate_ref(s0, s1, t, u):
    """Reference (loop) implementation kept for the equivalence test."""
    B, N = s0.shape
    st = s0.clone()
    AtoB = (s0 == 0) & (s1 == 1)
    BtoA = (s0 == 1) & (s1 == 0)
    oa = torch.argsort(AtoB.float(), dim=1, descending=True, stable=True)
    ob = torch.argsort(BtoA.float(), dim=1, descending=True, stable=True)
    na = AtoB.sum(1)
    resolve = u < t[:, None]
    for b in range(B):
        m = int(na[b])
        if m == 0:
            continue
        ai = oa[b, :m]; bi = ob[b, :m]; r = resolve[b, :m]
        st[b, ai[r]] = s1[b, ai[r]]
        st[b, bi[r]] = s1[b, bi[r]]
    return st


def per_species_ot(x0, x1, nA, L):
    """Per-species OT for species-SORTED configs (first nA particles are A): Hungarian within each block.
    NOTE: an ABLATION option vs v1's global OT — the 6.5x OT win on eRSI was OT-vs-nothing, not
    per-species-vs-global; do not presume it transfers to the joint flow."""
    a = linear_assignment_permutation(x0[:, :nA], x1[:, :nA], L=L)[0]
    b = linear_assignment_permutation(x0[:, nA:], x1[:, nA:], L=L)[0]
    return torch.cat([a, b], dim=1)


@torch.no_grad()
def denoiser_eval(model, x1, s1, t_eval=0.9, n_rep=4, t_pos=None):
    """G1 metric: denoiser accuracy + ECE at the swap-proposer operating point. Builds (x_t, s_t) at t=t_eval
    from random s0 + Kawasaki, x_t on the interpolant toward x1 (x0 uniform, globally OT-aligned).
    Two-time models: t_pos controls the POSITION interpolant separately (t_pos=1.0, t_eval=0.0 = the
    geometry-table query: clean positions, uninformative labels)."""
    B, N = s1.shape; dev = x1.device; L = model.L
    tp_val = t_eval if t_pos is None else t_pos
    accs, confs, cors, mm_accs = [], [], [], []
    for _ in range(n_rep):
        x0 = torch.rand(B, N, 2, device=dev) * L
        x0 = global_position_ot(x0, x1, L)
        tp = torch.full((B,), tp_val, device=dev)
        t = torch.full((B,), t_eval, device=dev)
        x_t = exp_map(x1, (1.0 - tp)[:, None, None] * log_map(x1, x0, L), L)
        s0 = random_22_labeling(B, N, int(s1[0].sum()), dev)
        s_t = kawasaki_interpolate(s0, s1, t)
        if getattr(model, "two_time", False):
            _, logits = model(tp.view(B, 1, 1), x_t, s_t, t_spec=t.view(B, 1, 1))
        else:
            _, logits = model(t.view(B, 1, 1), x_t, s_t)
        p = torch.softmax(logits, -1)
        pred = p.argmax(-1)
        accs.append((pred == s1).float().mean())
        mm = s_t != s1                                                   # the still-misassigned sites
        if mm.any():
            mm_accs.append((pred == s1)[mm].float().mean())              # the honest discriminator
        confs.append(p.max(-1).values.flatten()); cors.append((pred == s1).float().flatten())
    conf = torch.cat(confs); cor = torch.cat(cors)
    bins = torch.linspace(0.5, 1.0, 11, device=conf.device); ece = torch.tensor(0.0, device=conf.device)
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (conf >= lo) & (conf < hi)
        if m.any():
            ece = ece + m.float().mean() * (conf[m].mean() - cor[m].mean()).abs()
    return {"acc": float(torch.stack(accs).mean()), "ece": float(ece),
            "acc_mismatch": float(torch.stack(mm_accs).mean()) if mm_accs else float("nan")}


def joint_loss(model, x0, x1, s0, s1, lam=1.0):
    """L_pos (division-free FM) + lam * L_spec (denoiser CE), on shared interpolated (x_t, s_t). x0 already
    OT-aligned to x1. s0 random 22:22, s1 data labeling (aligned to x1's index order).
    If model.two_time: t_pos and t_spec are sampled INDEPENDENTLY, so the model learns all four corners —
    including (t_pos=1, t_spec=0): clean geometry + uninformative labels = the geometry table."""
    B, N = x1.shape[0], x1.shape[1]; dev = x1.device; L = model.L
    t = torch.rand(B, device=dev)
    x_t = exp_map(x1, (1.0 - t)[:, None, None] * log_map(x1, x0, L), L)
    if getattr(model, "two_time", False):
        t_spec = torch.rand(B, device=dev)
        s_t = kawasaki_interpolate(s0, s1, t_spec)
        v, logits = model(t.view(B, 1, 1), x_t, s_t, t_spec=t_spec.view(B, 1, 1))
    else:
        s_t = kawasaki_interpolate(s0, s1, t)
        v, logits = model(t.view(B, 1, 1), x_t, s_t)
    target = -log_map(x1, x0, L)
    L_pos = (torch.sum((v - target) ** 2, dim=(1, 2)) / (N * 2)).mean()
    L_spec = F.cross_entropy(logits.reshape(B * N, -1), s1.reshape(B * N))
    return L_pos + lam * L_spec, L_pos.detach(), L_spec.detach()
