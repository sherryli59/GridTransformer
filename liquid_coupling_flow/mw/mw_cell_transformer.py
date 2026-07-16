"""Periodic cell context built from the July-15 selected scaffold trunk.

The cell model changes support geometry, not the successful local neural
ingredients: physical Fourier KNN tokens, separate prefix/boundary streams,
the four-layer transformer, sharp Cat3 head, and three learned radial tilts.
All context is causal; callers pass only strict earlier-color particles and
the current cell prefix.
"""
from __future__ import annotations

import math
from pathlib import Path

import torch
import torch.nn as nn

from liquid_coupling_flow.mw.mw_cell_chart import anchored_cube_forward
from liquid_coupling_flow.mw.mw_energy import (
    capped_mw_candidate_increment,
    capped_mw_candidate_pair_increment,
)


DEFAULT_SELECTED_SCAFFOLD = Path(__file__).resolve().parent / (
    "artifacts/mw_scaffold_runs/mw_scaffold_N512_v2_selected.pt"
)
JULY14_MLE_SCAFFOLD = Path(__file__).resolve().parent / (
    "artifacts/mw_scaffold_runs/mw_scaffold_N512_varK_aug_mle_step12750.pt"
)


def _zero_last(module: nn.Sequential) -> None:
    nn.init.zeros_(module[-1].weight)
    nn.init.zeros_(module[-1].bias)


class MWCellTransformerContext(nn.Module):
    """Size-local, periodic monatomic KNN transformer for one cell slot."""

    def __init__(self, d_model=128, n_head=4, n_layer=4, knn=20,
                 knn_bnd=20, bnd_cutoff=1.8, knn_pot=24,
                 n_rbf=12, rbf_max=3.0, phi_hidden=32, pair_dim=8):
        super().__init__()
        self.d_model = int(d_model)
        self.knn = int(knn)
        self.knn_bnd = int(knn_bnd)
        self.bnd_cutoff = float(bnd_cutoff)
        self.knn_pot = int(knn_pot)
        self.register_buffer("periods", torch.tensor([0.5, 1.0, 2.0, 4.0, 8.0]))
        self.nbr_proj = nn.Linear(2 * 3 * self.periods.numel(), self.d_model)
        # Keep source shapes verbatim; row zero is the monatomic token.
        self.sp_emb = nn.Embedding(2, self.d_model)
        self.kind_emb = nn.Embedding(2, self.d_model)
        self.query = nn.Parameter(torch.zeros(1, 1, self.d_model))
        self.slot_proj = nn.Sequential(
            nn.Linear(3, self.d_model), nn.GELU(),
            nn.Linear(self.d_model, self.d_model),
        )
        layer = nn.TransformerEncoderLayer(
            self.d_model, int(n_head), 4 * self.d_model, batch_first=True,
            activation="gelu", norm_first=True,
        )
        self.tr = nn.TransformerEncoder(layer, int(n_layer), enable_nested_tensor=False)
        self.sp_out_emb = nn.Embedding(2, self.d_model)

        # New cell/count conditioning starts neutral under a scaffold warm start.
        self.count_embed = nn.Sequential(
            nn.Linear(27, self.d_model), nn.SiLU(),
            nn.Linear(self.d_model, self.d_model),
        )
        self.cell_embed = nn.Sequential(
            nn.Linear(6, self.d_model), nn.SiLU(),
            nn.Linear(self.d_model, self.d_model),
        )
        _zero_last(self.count_embed)
        _zero_last(self.cell_embed)

        self.n_rbf = int(n_rbf)
        mu = torch.linspace(0.0, float(rbf_max), self.n_rbf)
        self.register_buffer("rbf_mu", mu)
        self.rbf_w = float(mu[1] - mu[0])

        def potential():
            return nn.Sequential(
                nn.Linear(self.n_rbf + int(pair_dim), int(phi_hidden)), nn.SiLU(),
                nn.Linear(int(phi_hidden), int(phi_hidden)), nn.SiLU(),
                nn.Linear(int(phi_hidden), 1),
            )

        self.pair_emb = nn.Embedding(4, int(pair_dim))
        self.pair_emb_a = nn.Embedding(4, int(pair_dim))
        self.pair_emb_b = nn.Embedding(4, int(pair_dim))
        self.phi = potential()
        self.phi_a = potential()
        self.phi_b = potential()

    @staticmethod
    def mic(delta: torch.Tensor, L: float) -> torch.Tensor:
        return delta - float(L) * torch.round(delta / float(L))

    def _periodic(self, rel: torch.Tensor) -> torch.Tensor:
        phase = 2.0 * math.pi * rel.unsqueeze(-1) / self.periods
        return torch.cat((torch.sin(phase), torch.cos(phase)), -1).flatten(-2)

    def _tokens(self, points: torch.Tensor, origin: torch.Tensor, kind: int,
                k_max: int, L: float, cutoff=None):
        if points.numel() == 0:
            return origin.new_zeros((0, self.d_model))
        rel = self.mic(points - origin[None], L)
        d2 = rel.square().sum(-1)
        valid = torch.ones(points.shape[0], dtype=torch.bool, device=points.device)
        if cutoff is not None:
            valid &= d2 < float(cutoff) ** 2
        order = torch.argsort(d2.masked_fill(~valid, float("inf")), stable=True)
        order = order[:min(int(k_max), points.shape[0])]
        order = order[valid[order]]
        rel = rel[order]
        if rel.numel() == 0:
            return origin.new_zeros((0, self.d_model))
        species = torch.zeros(rel.shape[0], dtype=torch.long, device=rel.device)
        kinds = torch.full_like(species, int(kind))
        return self.nbr_proj(self._periodic(rel)) + self.sp_emb(species) + self.kind_emb(kinds)

    def forward(self, earlier: torch.Tensor, prefix: torch.Tensor,
                origin: torch.Tensor, count_stencil: torch.Tensor,
                slot_features: torch.Tensor, cell_features: torch.Tensor,
                L: float) -> torch.Tensor:
        """Return one positional context; no same/later-stage data are accepted."""
        prefix_tokens = self._tokens(prefix, origin, 0, self.knn, L)
        earlier_tokens = self._tokens(
            earlier, origin, 1, self.knn_bnd, L, cutoff=self.bnd_cutoff
        )
        query = (
            self.query.reshape(-1) + self.slot_proj(slot_features)
            + self.count_embed(count_stencil) + self.cell_embed(cell_features)
        )
        seq = torch.cat((query[None], prefix_tokens, earlier_tokens), 0)[None]
        hidden = self.tr(seq)[:, 0]
        # Monatomic analogue of conditioning the position head on sampled species.
        return hidden[0] + self.sp_out_emb.weight[0]

    def forward_many(self, rows):
        """Evaluate independent cell-slot contexts in one transformer call.

        Each row is the argument dictionary of :meth:`forward`.  Rows may
        have different KNN token counts; padding is internal and cannot expose
        a same-stage particle because every row's ``earlier`` tensor is already
        a strict causal mask.
        """
        if not rows:
            return []
        sequences = []
        for row in rows:
            earlier = row["earlier"]
            prefix = row["prefix"]
            origin = row["origin"]
            prefix_tokens = self._tokens(prefix, origin, 0, self.knn, row["L"])
            earlier_tokens = self._tokens(
                earlier, origin, 1, self.knn_bnd, row["L"], cutoff=self.bnd_cutoff
            )
            query = (
                self.query.reshape(-1) + self.slot_proj(row["slot_features"])
                + self.count_embed(row["count_stencil"]) + self.cell_embed(row["cell_features"])
            )
            sequences.append(torch.cat((query[None], prefix_tokens, earlier_tokens), 0))
        width = max(seq.shape[0] for seq in sequences)
        B = len(sequences)
        packed = sequences[0].new_zeros(B, width, self.d_model)
        pad = torch.ones(B, width, dtype=torch.bool, device=packed.device)
        for b, seq in enumerate(sequences):
            packed[b, :seq.shape[0]] = seq
            pad[b, :seq.shape[0]] = False
        return list(self.tr(packed, src_key_padding_mask=pad)[:, 0] + self.sp_out_emb.weight[0])

    def _cage(self, earlier: torch.Tensor, prefix: torch.Tensor,
              origin: torch.Tensor, L: float) -> torch.Tensor:
        cage = torch.cat((earlier, prefix), 0)
        if cage.shape[0] <= self.knn_pot:
            return cage
        d2 = self.mic(cage - origin[None], L).square().sum(-1)
        return cage[torch.argsort(d2, stable=True)[:self.knn_pot]]

    def _learned_pair_cost(self, candidates: torch.Tensor, cage: torch.Tensor,
                           net: nn.Module, emb: nn.Embedding, L: float) -> torch.Tensor:
        if cage.numel() == 0:
            return candidates.new_zeros(candidates.shape[0])
        delta = self.mic(candidates[:, None] - cage[None], L)
        dist = delta.norm(dim=-1)
        rbf = torch.exp(-((dist[..., None] - self.rbf_mu) ** 2) / (2 * self.rbf_w ** 2))
        # Source pair index 0 is monatomic candidate/cage.
        pair = emb.weight[0].expand(*dist.shape, emb.embedding_dim)
        return net(torch.cat((rbf, pair), -1)).squeeze(-1).sum(-1)

    def axis_cost(self, fixed_u: torch.Tensor, axis: int, anchor: torch.Tensor,
                  cell_index: torch.Tensor, h: float, earlier: torch.Tensor,
                  prefix: torch.Tensor, L: float, bin_centers: torch.Tensor,
                  mw_pair_tilt=0.0, mw_three_tilt=0.0,
                  pair_cap=5.0, three_cap=5.0) -> torch.Tensor:
        """Candidate cost for one Cat3 axis, identical in sample and score."""
        u = fixed_u[None].expand(bin_centers.numel(), 3).clone()
        u[:, int(axis)] = bin_centers
        local, _ = anchored_cube_forward(u, anchor[None], width=float(h))
        candidates = cell_index.to(local.dtype)[None] * float(h) + local
        origin = cell_index.to(local.dtype) * float(h) + anchor * float(h)
        cage = self._cage(earlier, prefix, origin, L)
        net, emb = (
            (self.phi_a, self.pair_emb_a) if int(axis) == 0 else
            (self.phi_b, self.pair_emb_b) if int(axis) == 1 else
            (self.phi, self.pair_emb)
        )
        learned = self._learned_pair_cost(candidates, cage, net, emb, L)
        if cage.numel() == 0 or (not mw_pair_tilt and not mw_three_tilt):
            return learned
        if not mw_three_tilt:
            pair = capped_mw_candidate_pair_increment(
                candidates[None], cage[None], pair_cap=pair_cap, L=L,
            )[0]
            return learned + float(mw_pair_tilt) * pair
        pair, three = capped_mw_candidate_increment(
            candidates[None], cage[None], pair_cap=pair_cap,
            three_cap=three_cap, L=L,
        )
        return learned + float(mw_pair_tilt) * pair[0] + float(mw_three_tilt) * three[0]


def warm_start_cell_from_scaffold(model: nn.Module, checkpoint=DEFAULT_SELECTED_SCAFFOLD):
    """Transplant every shape-compatible local trunk/head/tilt tensor.

    Returns metadata suitable for persisting in a cell-q0 checkpoint.  The
    selected July-15 checkpoint is the primary source; callers can explicitly
    pass :data:`JULY14_MLE_SCAFFOLD` for the calibration A/B.
    """
    checkpoint = Path(checkpoint)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    source = payload["state_dict"]
    target = model.state_dict()
    mapping = {}
    direct = (
        "query", "periods", "nbr_proj.", "sp_emb.", "kind_emb.",
        "slot_proj.", "tr.", "sp_out_emb.", "rbf_mu", "pair_emb.",
        "pair_emb_a.", "pair_emb_b.", "phi.", "phi_a.", "phi_b.",
    )
    for source_key, value in source.items():
        if source_key.startswith("flow."):
            target_key = "position_head.base." + source_key[len("flow."):]
        elif source_key == "query" or any(source_key.startswith(p) for p in direct[1:]):
            target_key = "context_encoder." + source_key
        else:
            continue
        if target_key in target and target[target_key].shape == value.shape:
            mapping[target_key] = value
    expected_head = {
        "position_head.base.head_a.weight", "position_head.base.head_b.weight",
        "position_head.base.head_c.weight", "position_head.base.bin_a_emb.weight",
        "position_head.base.bin_b_emb.weight",
    }
    if not expected_head.issubset(mapping):
        missing = sorted(expected_head - mapping.keys())
        raise ValueError(f"incompatible scaffold checkpoint; missing {missing}")
    target.update(mapping)
    model.load_state_dict(target)
    return {
        "checkpoint": str(checkpoint),
        "source_step": int(payload.get("step", -1)),
        "source_variant": payload.get("variant", "july14_mle"),
        "source_pos_temp": payload.get("pos_temp"),
        "loaded_tensors": len(mapping),
        "excluded_species_head": True,
    }
