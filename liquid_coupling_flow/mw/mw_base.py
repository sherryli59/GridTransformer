"""Uniform base distribution q0 for the mW annealed-SMC path (Task 5). log q0 is a CONSTANT
(uniform-in-box, i.i.d. per particle), so it must drop out of every mutation acceptance ratio —
LOGQ_CONST=True lets mw_smc.mutation_sweeps certify that reduction BEFORE any neural base exists
(Phase-2 fault isolation)."""
from __future__ import annotations
import math, os, torch


class UniformBase:
    LOGQ_CONST = True

    def __init__(self, N, L):
        self.N, self.L = N, L
        self._lq = -N * 3 * math.log(L)

    def sample(self, B, gen=None):
        return torch.rand(B, self.N, 3, generator=gen,
                          device=gen.device if gen is not None and gen.device.type == "cuda" else "cpu") * self.L

    def log_q(self, x):
        return torch.full((x.shape[0],), self._lq, device=x.device)


class GeneratorBase:
    """mw_smc base backed by a trained autoregressive generator (v10 & kin).

    Wraps any model exposing ``sample(B, N, L, gen, return_logq=True)`` and
    ``log_prob(x, L, preordered=True)`` so mw_smc's smc_run / mutation_sweeps can use it exactly
    like ``UniformBase``:
      * ``sample(B, gen) -> x``           : draw B configs in the model's canonical AR order;
      * ``log_q(x) -> [B]``               : the AR base log-density, evaluated with preordered=True
                                            (the labeled-space exactness convention -- the current
                                            particle labeling IS the AR conditioning order; the SMC
                                            weights and pi_lambda-invariant mutation MH are exact in
                                            that labeled space), in eval mode, no_grad, chunked <=128;
      * ``LOGQ_CONST = False``            : log q0 is NOT a constant, so mutation_sweeps must (and
                                            does) evaluate base.log_q PER site-move -- one causal
                                            forward per move (the cost the base gate measures).
    """

    LOGQ_CONST = False

    def __init__(self, model, N, L):
        self.model = model.eval()
        self.N, self.L = N, L

    def sample(self, B, gen=None):
        self.model.eval()
        with torch.no_grad():
            x, _ = self.model.sample(B, self.N, self.L, gen=gen, return_logq=True)
        return x

    def log_q(self, x, chunk=128):
        self.model.eval()
        outs = []
        with torch.no_grad():
            for i in range(0, x.shape[0], chunk):
                outs.append(self.model.log_prob(x[i:i + chunk], self.L, preordered=True))
        return torch.cat(outs)


def _wrap_pm(v, L):
    """Minimum-image wrap of a displacement into (-L/2, L/2] componentwise (module-local so mw_base
    stays light; identical to mw_generator.wrap_pm / mw_energy's inline d - L*round(d/L))."""
    return v - L * torch.round(v / L)


class ExcludedVolumeBase:
    """Non-learned analytic soft-core one-body insertion base -- the Task-11 control that asks whether
    the learned transformer base beats a simple physics base.

    Per autoregressive insertion step j (STORAGE/slot order -- the same preordered convention as
    GeneratorBase), the conditional is a PURE analytic soft-core repulsion:

        p(x_j | x_{<j}) = exp(-phi_j(x_j)) / Z_j,
        phi_j(x)       = sum_{i<j} u_rep(|wrap_pm(x - x_i, L)|),
        u_rep(r)       = EPS_REP * (SIGMA_REP / max(r, r_floor))**12.

    NO attraction: the base carves excluded volume (a hard-ish core, no clashes) with ZERO learning
    but installs NO shell structure -- so a learned base's shells remain its measurable edge.

    EXACT tractable density via PIECEWISE-CONSTANT grid quadrature on a G^3 grid. The density is
    DEFINED to be constant over each grid cell:

        p(x_j) = exp(-phi_j(cell_center(x_j))) / Z_j_disc,
        Z_j_disc = sum_cells exp(-phi_j(center)) * V_cell     (V_cell = (L/G)**3),

    so it is exactly normalized (sum_cells p(cell)*V_cell = 1) and, crucially, sample() and log_q()
    read the density at the SAME point (the cell center):
      * sample(): pick a cell ~ exp(-phi(center))*V_cell (categorical over cells), then place x_j
                  UNIFORMLY inside it. The continuous density at x_j is exactly p(x_j) above, and the
                  accumulated log p(x_j) = -phi_j(center) - log Z_j_disc.
      * log_q():  find x_j's cell, evaluate the SAME piecewise-constant log-density.
    => sample() == log_q() by construction (the weight-path exactness invariant), up to float.

    Ordering. sample() emits configs in slot order (slot j == insertion step j). log_q(x) defaults to
    preordered=True and scores x in its GIVEN storage order (slot j == step j, causal prefix = slots
    < j) with NO re-canonicalization -- exactly the density of sample()'s own output, mirroring
    GeneratorBase.log_q so mw_smc treats this base and the learned base identically. preordered=False
    first canonicalizes via the shared scaffold canonical_order (generator-comparable scoring of an
    unordered config; requires N a perfect cube). LOGQ_CONST=False (log q0 is configuration-dependent,
    so mutation_sweeps evaluates base.log_q per site-move -- the cost the base gate measures).
    """

    LOGQ_CONST = False

    def __init__(self, N, L, sigma_rep=1.0, eps_rep=1.0, G=24, r_floor=None, phi_max=60.0):
        self.N, self.L = N, L
        self.sigma_rep = float(sigma_rep)
        self.eps_rep = float(eps_rep)
        self.G = int(G)
        self.r_floor = float(r_floor) if r_floor is not None else 0.1 * self.sigma_rep
        self.phi_max = float(phi_max)
        self.h = L / self.G                                   # grid cell width
        self.V_cell = self.h ** 3
        self.logVcell = 3.0 * math.log(self.h)
        self._centers = None                                  # lazy per-device G^3 cell centers
        self._centers_device = None

    # -- geometry / potential helpers -------------------------------------------------
    def _grid_centers(self, device):
        """[G^3, 3] midpoint cell centers, row-major over (i,j,k): linear = i*G^2 + j*G + k."""
        if self._centers is None or self._centers_device != device:
            g = (torch.arange(self.G, device=device, dtype=torch.float32) + 0.5) * self.h
            cx, cy, cz = torch.meshgrid(g, g, g, indexing="ij")
            self._centers = torch.stack([cx, cy, cz], dim=-1).reshape(-1, 3)
            self._centers_device = device
        return self._centers

    def _u_rep(self, r):
        return self.eps_rep * (self.sigma_rep / r.clamp_min(self.r_floor)) ** 12

    def _phi_at(self, query, prefix, L, mchunk=2048):
        """phi at query points [B,Q,3] given a batched prefix [B,p,3] -> [B,Q], clamped to phi_max.
        Chunked over Q to bound the transient [B,chunk,p,3] pair tensor. p==0 -> zeros (step j=0)."""
        B, Q, _ = query.shape
        p = prefix.shape[1]
        out = torch.zeros(B, Q, device=query.device)
        if p == 0:
            return out
        for q0 in range(0, Q, mchunk):
            qc = query[:, q0:q0 + mchunk]                                     # [B,c,3]
            rel = _wrap_pm(qc[:, :, None, :] - prefix[:, None, :, :], L)       # [B,c,p,3]
            r = rel.norm(dim=-1)                                              # [B,c,p]
            out[:, q0:q0 + mchunk] = self._u_rep(r).sum(-1)                   # [B,c]
        return out.clamp_max(self.phi_max)

    def _cell_index(self, x):
        """Linear grid-cell index [.,] of positions x [.,3] (floor toward the containing cell,
        clamped to [0,G-1] so a stray fp value at the box edge maps to the boundary cell)."""
        idx = torch.floor(x / self.h).long().clamp(0, self.G - 1)             # [...,3]
        return idx[..., 0] * self.G * self.G + idx[..., 1] * self.G + idx[..., 2]

    # -- interface: sample / log_q ---------------------------------------------------
    @torch.no_grad()
    def sample(self, B, gen=None, return_logq=False):
        """Sequential AR insertion. j=0: uniform (empty prefix -> phi=0). j>0: score exp(-phi_j) on
        the G^3 grid, sample a cell (multinomial), place uniformly inside it. Returns x [B,N,3] (slot
        j == step j), or (x, logq) when return_logq (the exactness invariant's reference value)."""
        device = gen.device if (gen is not None and gen.device.type == "cuda") else "cpu"
        centers = self._grid_centers(device)                                  # [M,3]
        M = centers.shape[0]
        cen_b = centers[None].expand(B, -1, -1)                               # [B,M,3] view
        x = torch.zeros(B, self.N, 3, device=device)
        logq = torch.zeros(B, device=device)
        for j in range(self.N):
            if j == 0:
                phi = torch.zeros(B, M, device=device)
            else:
                phi = self._phi_at(cen_b, x[:, :j, :], self.L)               # [B,M]
            logZ = torch.logsumexp(-phi, dim=1) + self.logVcell              # [B]
            cell = torch.multinomial(torch.softmax(-phi, dim=1), 1, generator=gen).squeeze(1)  # [B]
            phi_sel = phi.gather(1, cell[:, None]).squeeze(1)                 # [B] phi at chosen center
            corner = centers[cell] - 0.5 * self.h                            # [B,3] cell lower corner
            u = torch.rand(B, 3, generator=gen, device=device)
            x[:, j, :] = corner + self.h * u                                 # uniform within the cell
            logq = logq + (-phi_sel - logZ)
        return (x, logq) if return_logq else x

    @torch.no_grad()
    def log_q(self, x, preordered=True, chunk=64):
        """Piecewise-constant AR log-density [B]. preordered=True (default; mw_smc / importance
        weights): score x in its given storage order. preordered=False: canonicalize via the shared
        scaffold canonical_order first (generator-comparable; needs N a cube). Mirrors sample()'s
        per-step reductions EXACTLY (same _phi_at, same logsumexp, same cell gather), so log_q of a
        config sample() produced equals its returned logq up to float."""
        device = x.device
        if not preordered:
            from liquid_coupling_flow.mw.mw_generator import mw_scaffold, canonical_order
            xw = torch.remainder(x, self.L)
            _, rank, R = mw_scaffold(self.N, self.L, device)
            perm = canonical_order(xw, self.L, R, rank)
            x = torch.gather(xw, 1, perm[..., None].expand(-1, -1, 3))
        outs = []
        for b0 in range(0, x.shape[0], chunk):
            outs.append(self._log_q_chunk(x[b0:b0 + chunk]))
        return torch.cat(outs)

    def _log_q_chunk(self, x):
        B = x.shape[0]
        device = x.device
        centers = self._grid_centers(device)
        cen_b = centers[None].expand(B, -1, -1)
        lin = self._cell_index(x)                                            # [B,N] cell of each x_j
        logq = torch.zeros(B, device=device)
        for j in range(self.N):
            if j == 0:
                phi = torch.zeros(B, centers.shape[0], device=device)
            else:
                phi = self._phi_at(cen_b, x[:, :j, :], self.L)
            logZ = torch.logsumexp(-phi, dim=1) + self.logVcell
            phi_sel = phi.gather(1, lin[:, j:j + 1]).squeeze(1)              # phi at x_j's cell center
            logq = logq + (-phi_sel - logZ)
        return logq


# v10-family architecture tags -> load_generator_v10 (identity-v4 pretrain, both fine-tunes).
_V10_TAGS = {
    "v4_exact_plus_toroidal_residual_v10",
    "v10_struct_finetune",
    "v10_struct_finetune2_reinforce",
}


def load_any_generator(path, device):
    """Load a generator checkpoint by dispatching on its ``architecture`` tag; raise on an unknown
    tag rather than guess.  Lazy per-branch imports keep the model stack out of mw_base's import
    graph (mw_smc imports UniformBase, which must stay light)."""
    ck = torch.load(path, map_location=device, weights_only=False)
    arch = ck.get("architecture")
    if arch in _V10_TAGS:
        from liquid_coupling_flow.mw.mw_generator_v10 import load_generator_v10
        return load_generator_v10(path, device=device)
    raise ValueError(
        f"load_any_generator: unknown architecture tag {arch!r} in {os.path.basename(path)}; "
        f"known tags = {sorted(_V10_TAGS)}")
