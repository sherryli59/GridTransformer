"""GEOMETRY-INVARIANT LOCAL-FRAME conditional (the direction the transfer diagnosis points to).
Place particle j NOT relative to the curve scaffold (N-specific) but in a LOCAL PHYSICAL FRAME built
from its already-placed neighbours: origin = nearest placed particle n1 (to scaffold_j, used only as a
LOCATOR), x-axis = n1->n2 (2nd nearest). Target = j's position in that frame (a,b). The frame is a
deterministic function of placed particles => the (a,b)->position map is a pure rotation+translation,
|det|=1, exact likelihood. The curve is demoted to a sequencer + a locator for "which neighbours";
the learned conditional sees only frame-relative neighbour positions+species => translation-, rotation-,
and SIZE-invariant. If this transfers where the scaffold conditional didn't, curve-anchoring was the bug.

This file: the geometry core + an exact round-trip test. Network/training added after the geometry is
proven bijective."""
from __future__ import annotations
import os, math, torch
import torch.nn as nn
import torch.nn.functional as F
from liquid_coupling_flow.ka_gridformer import KAGridformer, _wrap_pm

ART = os.path.join(os.path.dirname(__file__), "artifacts")
KLOC = 2                                                                # neighbours defining the frame (n1,n2)
KNN = 16                                                                # neighbours the conditional sees
GEOM_PERIODS = torch.tensor([0.5, 1.0, 2.0, 4.0, 8.0])                  # fixed PHYSICAL periods (size-invariant)


def _geo(N, device, cell_size=0.285):                                  # reuse gilbert constant-cell curve geometry
    return KAGridformer(L=(N / 1.2) ** 0.5, rho=1.2, R=32, repr_mode="arcnorm",
                        constant_cell=True, cell_size=cell_size).to(device)


SIGMA = 1.3                                                            # locator kernel width (~first shell)


def origin_of(pos, scaffold_j, L, j):
    """Soft weighted CENTROID of placed prefix particles (k<j), weighted by proximity to scaffold_j (a
    LOCATOR only). Continuous in pos => robust; a function of PLACED particles => unit-Jacobian anchor.
    pos [B,N,2], scaffold_j [2] (or [B,2]). Returns origin [B,2] = sum_k w_k (o + wrap(pos_k-o)) with o
    a stable reference (scaffold_j) to make the PBC mean well-defined."""
    d = _wrap_pm(pos[:, :j] - scaffold_j[None, None], L)                # prefix rel to scaffold_j (min-image) [B,j,2]
    w = torch.exp(-(d ** 2).sum(-1) / (2 * SIGMA ** 2))                 # [B,j] locator weights
    w = w / w.sum(1, keepdim=True).clamp_min(1e-12)
    return torch.remainder(scaffold_j[None] + (w[..., None] * d).sum(1), L)   # weighted centroid (PBC-safe)


def to_frame(pos, scaffold, L):
    """positions -> local coords (a,b) = pos_j - origin_j (global orientation; rotation via augmentation).
    origin_j = soft centroid of placed prefix near scaffold_j. j<2 fall back to scaffold (few neighbours)."""
    B, N, _ = pos.shape
    ab = torch.zeros(B, N, 2, device=pos.device)
    for j in range(N):
        o = scaffold[j][None].expand(B, 2) if j < 2 else origin_of(pos, scaffold[j], L, j)
        ab[:, j] = _wrap_pm(pos[:, j] - o, L)
    return ab


def from_frame_seq(ab, scaffold, L):
    """(a,b) -> positions SEQUENTIALLY (origin_j depends on placed 0..j-1)."""
    B, N, _ = ab.shape
    pos = torch.zeros(B, N, 2, device=ab.device)
    for j in range(N):
        o = scaffold[j][None].expand(B, 2) if j < 2 else origin_of(pos, scaffold[j], L, j)
        pos[:, j] = torch.remainder(o + ab[:, j], L)
    return pos


class KALocalFrameModel(nn.Module):
    """Geometry-invariant AR: place particle j at origin_j(placed) + (a,b), origin_j = soft centroid of
    placed neighbours near scaffold_j (a locator). The conditional sees ONLY neighbour positions relative
    to the centroid + species -> translation/size-invariant, no curve geometry in the learned function.
    Exact: origin_j is a function of the PREFIX, so the (a,b)->x_j map has |det|=1 (per-particle Jacobian)."""
    def __init__(self, rho=1.2, n_bins=192, d_model=192, n_head=6, n_layer=4, cell_size=0.285,
                 n_species=2, arc_range=3.0, knn=KNN, canonical=True):
        super().__init__()
        self.geo = KAGridformer(L=math.sqrt(100 / rho), rho=rho, R=32, repr_mode="arcnorm",
                                constant_cell=True, cell_size=cell_size)
        self.rho, self.n_bins, self.n_species, self.arc_range, self.knn = rho, n_bins, n_species, arc_range, knn
        self.canonical = canonical; self.d = 2; self.bin_w = 2 * arc_range / n_bins
        self.register_buffer("periods", GEOM_PERIODS)
        enc = 2 * self.d * len(GEOM_PERIODS)
        self.nbr_proj = nn.Linear(enc, d_model)
        self.sp_emb = nn.Embedding(n_species, d_model)
        self.query = nn.Parameter(torch.zeros(1, 1, d_model))
        layer = nn.TransformerEncoderLayer(d_model, n_head, 4 * d_model, batch_first=True,
                                           activation="gelu", norm_first=True)
        self.tr = nn.TransformerEncoder(layer, n_layer)
        self.head_a = nn.Linear(d_model, n_bins); self.head_b = nn.Linear(d_model, n_bins)
        self.bin_a_emb = nn.Embedding(n_bins, d_model)
        self.head_species = nn.Linear(d_model, n_species)

    def _Lof(self, N):
        return self.geo._Lof(N)

    def _arc_scale(self, N):
        return float(N) ** (1.0 / 6.0)

    def _periodic(self, rel):
        a = 2 * math.pi * rel.unsqueeze(-1) / self.periods
        return torch.cat([torch.sin(a), torch.cos(a)], -1).flatten(-2)

    def _bin(self, v):
        return ((v + self.arc_range) / self.bin_w).long().clamp(0, self.n_bins - 1)

    def _bin_center(self, b):
        return (b.float() + 0.5) * self.bin_w - self.arc_range

    def _local(self, pos, sp, sc, L, N):
        """All-particle (teacher-forced) local context + origins + neighbour gather.
        pos [B,N,2] curve-ordered. Returns context [B,N,d], origin [B,N,2]."""
        B = pos.shape[0]
        d = _wrap_pm(pos[:, None, :, :] - sc[None, :, None, :], L)        # pos_k - sc_j  [B,N(j),N(k),2]
        dist2 = (d ** 2).sum(-1)
        jj = torch.arange(N, device=pos.device)
        causal = jj[None, None, :] < jj[None, :, None]                    # k<j
        w = torch.exp(-dist2.masked_fill(~causal, 1e9) / (2 * SIGMA ** 2)) * causal.float()   # [B,N,N]
        wsum = w.sum(-1, keepdim=True).clamp_min(1e-12)
        origin = torch.remainder(sc[None] + (w[..., None] * d).sum(2) / wsum, L)              # soft centroid
        origin = torch.where((jj < 2)[None, :, None], sc[None].expand(B, N, 2), origin)       # j<2: scaffold
        idx = dist2.masked_fill(~causal, 1e9).topk(self.knn, dim=2, largest=False).indices     # [B,N,KNN]
        valid = torch.gather(causal.expand(B, N, N), 2, idx)                                   # [B,N,KNN] real nbr?
        nbr_pos = torch.gather(pos[:, None].expand(B, N, N, 2), 2, idx[..., None].expand(-1, -1, -1, 2))
        nbr_rel = _wrap_pm(nbr_pos - origin[:, :, None, :], L)                                 # rel to CENTROID
        nbr_sp = torch.gather(sp[:, None].expand(B, N, N), 2, idx)
        feat = self.nbr_proj(self._periodic(nbr_rel)) + self.sp_emb(nbr_sp)                    # [B,N,KNN,d]
        feat = feat * valid[..., None]                                                        # zero padded nbrs
        q = self.query.expand(B * N, 1, -1)
        seq = torch.cat([q, feat.reshape(B * N, self.knn, -1)], 1)                             # [B*N, 1+KNN, d]
        pad = torch.cat([torch.zeros(B * N, 1, dtype=torch.bool, device=pos.device),
                         ~valid.reshape(B * N, self.knn)], 1)                                  # mask padded keys
        h = self.tr(seq, src_key_padding_mask=pad)[:, 0].reshape(B, N, -1)
        return h, origin

    def log_prob(self, x, s, canonical=None):
        B, N = x.shape[0], x.shape[1]; s = s.long(); s = s.expand(B, N).clone() if s.dim() == 1 else s
        L = self._Lof(N); order = self.geo._curve_order(x, N)
        xo = torch.gather(x, 1, order[..., None].expand(-1, -1, 2)); so = torch.gather(s, 1, order)
        sc = self.geo._scaffold(N, x.device)
        context, origin = self._local(xo, so, sc, L, N)
        ab = _wrap_pm(xo - origin, L) / self._arc_scale(N)                                     # normalized local coords
        ba, bb = self._bin(ab[..., 0]), self._bin(ab[..., 1])
        la = F.log_softmax(self.head_a(context), -1)
        lb = F.log_softmax(self.head_b(context + self.bin_a_emb(ba)), -1)
        lp_ab = la.gather(-1, ba[..., None]).squeeze(-1) + lb.gather(-1, bb[..., None]).squeeze(-1)
        s_logits = self.head_species(context)
        if self.canonical if canonical is None else canonical:
            oh = F.one_hot(so, self.n_species).to(s_logits.dtype)
            rem = oh.sum(1, keepdim=True) - (oh.cumsum(1) - oh)
            s_logits = s_logits.masked_fill(rem <= 0, float("-inf"))
        lp_s = F.log_softmax(s_logits, -1).gather(-1, so[..., None]).squeeze(-1)
        vol = self.d * N * math.log(self.bin_w); jac = self.d * N * math.log(self._arc_scale(N))
        return (lp_ab + lp_s).sum(1) - vol - jac

    @torch.no_grad()
    def sample(self, B, N, n_B=None, device=None):
        L = self._Lof(N); sc = self.geo._scaffold(N, device); arc = self._arc_scale(N)
        pos = torch.zeros(B, N, 2, device=device); sp = torch.zeros(B, N, dtype=torch.long, device=device)
        rem = None
        if n_B is not None:
            rem = torch.zeros(B, self.n_species, device=device); rem[:, 0] = N - n_B; rem[:, 1] = n_B
        for j in range(N):
            h, origin = self._step(pos, sp, sc[j], j, L)
            s_logits = self.head_species(h)
            if rem is not None:
                s_logits = s_logits.masked_fill(rem <= 0, float("-inf"))
            sj = torch.multinomial(F.softmax(s_logits, -1), 1).squeeze(-1)
            if rem is not None:
                rem[torch.arange(B, device=device), sj] -= 1
            ba = torch.multinomial(F.softmax(self.head_a(h), -1), 1).squeeze(-1)
            bb = torch.multinomial(F.softmax(self.head_b(h + self.bin_a_emb(ba)), -1), 1).squeeze(-1)
            a = self._bin_center(ba) + (torch.rand(B, device=device) - 0.5) * self.bin_w
            b = self._bin_center(bb) + (torch.rand(B, device=device) - 0.5) * self.bin_w
            pos[:, j] = torch.remainder(origin + torch.stack([a, b], -1) * arc, L)
            sp[:, j] = sj
        return pos, sp

    def _step(self, pos, sp, sc_j, j, L):
        """Per-particle context + origin at sampling step j (placed = 0..j-1). j<2: origin=scaffold
        (matches log_prob); j>=2: soft centroid. Consistent with _local so q_sample == exp(log_prob)."""
        B = pos.shape[0]
        if j == 0:                                                                            # no neighbours
            return self.tr(self.query.expand(B, 1, -1))[:, 0], sc_j[None].expand(B, 2)
        d = _wrap_pm(pos[:, :j] - sc_j[None, None], L); dist2 = (d ** 2).sum(-1)               # [B,j]
        if j < 2:
            origin = sc_j[None].expand(B, 2)
        else:
            w = torch.exp(-dist2 / (2 * SIGMA ** 2)); wsum = w.sum(1, keepdim=True).clamp_min(1e-12)
            origin = torch.remainder(sc_j[None] + (w[..., None] * d).sum(1) / wsum, L)
        k = min(self.knn, j)
        idx = dist2.topk(k, dim=1, largest=False).indices                                      # [B,k]
        nbr_pos = torch.gather(pos[:, :j], 1, idx[..., None].expand(-1, -1, 2))
        nbr_rel = _wrap_pm(nbr_pos - origin[:, None, :], L)
        nbr_sp = torch.gather(sp[:, :j], 1, idx)
        feat = self.nbr_proj(self._periodic(nbr_rel)) + self.sp_emb(nbr_sp)                    # [B,k,d]
        h = self.tr(torch.cat([self.query.expand(B, 1, -1), feat], 1))[:, 0]
        return h, origin


@torch.no_grad()
def tf_clash(m, N, device, B=256):
    """Decisive transfer metric for the local-frame model: teacher-forced placement clash vs a spread-
    matched random control (place at origin + random offset of the data spread)."""
    ref = torch.load(os.path.join(ART, f"ka_reference_N{N}.pt"), map_location=device, weights_only=False)
    s, L, data = ref["s"].to(device).long(), ref["L"], ref["x"].to(device)[:B]
    order = m.geo._curve_order(data, N); xo = torch.gather(data, 1, order[..., None].expand(-1, -1, 2))
    so = torch.gather(s.expand(data.shape[0], N).clone() if s.dim() == 1 else s[:B], 1, order)
    sc = m.geo._scaffold(N, device); arc = m._arc_scale(N)
    context, origin = m._local(xo, so, sc, L, N)
    ba = torch.multinomial(F.softmax(m.head_a(context).reshape(-1, m.n_bins), -1), 1).reshape(B, N)
    bb = torch.multinomial(F.softmax(m.head_b(context + m.bin_a_emb(ba)).reshape(-1, m.n_bins), -1), 1).reshape(B, N)
    a = m._bin_center(ba) + (torch.rand(B, N, device=device) - 0.5) * m.bin_w
    bcoord = m._bin_center(bb) + (torch.rand(B, N, device=device) - 0.5) * m.bin_w
    pos_tf = torch.remainder(origin + torch.stack([a, bcoord], -1) * arc, L)
    rstd = _wrap_pm(xo - origin, L).std()
    pos_rand = torch.remainder(origin + rstd * torch.randn_like(xo), L)

    def clash(pos):
        out = []
        for j in range(1, N):
            df = pos[:, j:j + 1] - xo[:, :j]; df = df - L * torch.round(df / L)
            out.append(((df ** 2).sum(-1).min(1).values.sqrt() < 0.7).float().mean())
        return float(torch.stack(out)[5:].mean())
    return clash(pos_tf), clash(pos_rand)


def train(train_N=100, steps=40000, device="cuda" if torch.cuda.is_available() else "cpu"):
    import time
    from liquid_coupling_flow.ka_gridformer_train import augment
    ref = torch.load(os.path.join(ART, f"ka_reference_N{train_N}.pt"), map_location=device, weights_only=False)
    data, s, L = ref["x"].to(device), ref["s"].to(device).long(), ref["L"]
    m = KALocalFrameModel(rho=1.2, n_bins=192, knn=KNN).to(device)
    print(f"LOCAL-FRAME (geometry-invariant) train @N={train_N}, {sum(p.numel() for p in m.parameters())/1e6:.2f}M "
          f"params, {steps} steps. Transfer verdict = TF clash << random at UNSEEN N.", flush=True)
    opt = torch.optim.AdamW(m.parameters(), lr=3e-4, weight_decay=1e-4); B, t0 = 96, time.time()
    for step in range(steps):
        idx = torch.randint(0, data.shape[0], (B,), device=device)
        loss = -(m.log_prob(augment(data[idx], L), s) / train_N).mean()
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 5.0); opt.step()
        if step % 1000 == 0:
            print(f"  step {step:5d} -logq/N {loss.item():.3f} {time.time()-t0:.0f}s", flush=True)
        if (step + 1) % (steps // 4) == 0:
            m.eval(); line = []
            for Nv in (36, 100, 256):
                try:
                    tf, rn = tf_clash(m, Nv, device)
                    line.append(f"N{Nv}{'(HELDOUT)' if Nv != train_N else '(train)'} TF {tf:.3f} vs rand {rn:.3f}")
                except RuntimeError as e:                          # eval-OOM (e.g. N=256) must NOT kill training
                    torch.cuda.empty_cache(); line.append(f"N{Nv} EVAL-SKIP({type(e).__name__})")
            torch.cuda.empty_cache(); m.train()
            print(f"  >> step {step+1}: " + " | ".join(line), flush=True)
            torch.save({"state_dict": m.state_dict(), "rho": 1.2, "n_bins": 192, "knn": KNN,
                        "train_N": train_N, "step": step + 1}, os.path.join(ART, f"ka_localframe_N{train_N}.pt"))
            print(f"  [ckpt saved @ step {step+1}]", flush=True)
    torch.save({"state_dict": m.state_dict(), "rho": 1.2, "n_bins": 192, "knn": KNN,
                "train_N": train_N, "step": steps}, os.path.join(ART, f"ka_localframe_N{train_N}.pt"))
    print(f"saved ka_localframe_N{train_N}.pt", flush=True)


@torch.no_grad()
def main(device="cuda" if torch.cuda.is_available() else "cpu"):
    print("local-frame geometry exact round-trip test (foundation for exact likelihood):", flush=True)
    for N in (36, 100, 256):
        m = _geo(N, device); L = m._Lof(N)
        ref = torch.load(os.path.join(ART, f"ka_reference_N{N}.pt"), map_location=device, weights_only=False)
        data = ref["x"].to(device)[:64]
        order = m._curve_order(data, N); xo = torch.gather(data, 1, order[..., None].expand(-1, -1, 2))
        sc = m._scaffold(N, device)
        ab = to_frame(xo, sc, L)
        rt = from_frame_seq(ab, sc, L)
        err = (torch.remainder(rt - xo + L / 2, L) - L / 2).abs().max().item()
        print(f"  N={N:3d}: max round-trip |err| {err:.2e}  local-coord(a,b) std {ab[:, 2:].std():.3f} "
              f"range[{ab[:, 2:].min():.2f},{ab[:, 2:].max():.2f}]  -> {'PASS' if err < 1e-4 else 'FAIL'}", flush=True)


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "train":
        train(steps=int(sys.argv[2]) if len(sys.argv) > 2 else 40000)
    else:
        main()
