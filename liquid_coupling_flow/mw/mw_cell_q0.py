"""Exact augmented hierarchical mW cell q0 with scaffold-transplanted trunk."""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from liquid_coupling_flow.mw.mw_cell_chart import (
    AnchoredCubeCat3Head,
    anchored_cube_forward,
    anchored_cube_inverse,
    clip_anchor,
)
from liquid_coupling_flow.mw.mw_cell_count import OctreeCountModel
from liquid_coupling_flow.mw.mw_cell_geom import (
    LOG_COLOR_ORDERS,
    causal_cell_order,
    cell_colors,
    cell_local_coordinates,
    color_stages,
    cubic_orientations,
    from_oriented,
    group_by_cell_priority,
    sample_color_order,
    sample_orientation,
    to_oriented,
    unflatten_cell_indices,
    validate_color_order,
)
from liquid_coupling_flow.mw.mw_cell_transformer import MWCellTransformerContext


@dataclass
class AugmentedCellState:
    x: torch.Tensor
    priorities: torch.Tensor
    shift: torch.Tensor
    orientation_index: torch.Tensor
    color_order: torch.Tensor
    debug_counts: torch.Tensor | None = None

    def to(self, device) -> "AugmentedCellState":
        return AugmentedCellState(
            self.x.to(device),
            self.priorities.to(device),
            self.shift.to(device),
            self.orientation_index.to(device),
            self.color_order.to(device),
            None if self.debug_counts is None else self.debug_counts.to(device),
        )


def _wrap_pm(x: torch.Tensor, L: float) -> torch.Tensor:
    return x - float(L) * torch.round(x / float(L))


class MWCellQ0(nn.Module):
    """Normalized augmented cell density with strict earlier-stage context."""

    def __init__(self, d_model: int = 128, count_hidden: int = 96,
                 cat_bins: int = 64, knn: int = 20, pseudo_weight: float = 1.0,
                 n_head: int = 4, n_layer: int = 4, knn_bnd: int = 20,
                 bnd_cutoff: float = 1.8, knn_pot: int = 24,
                 pos_temp: float = 1.0, mw_pair_tilt: float = 0.02,
                 mw_three_tilt: float = 0.0, pair_cap: float = 5.0,
                 three_cap: float = 5.0):
        super().__init__()
        self.d_model = int(d_model)
        self.cat_bins = int(cat_bins)
        self.knn = int(knn)
        self.pseudo_weight = float(pseudo_weight)
        self.pos_temp = float(pos_temp)
        self.mw_pair_tilt = float(mw_pair_tilt)
        self.mw_three_tilt = float(mw_three_tilt)
        self.pair_cap = float(pair_cap)
        self.three_cap = float(three_cap)
        if self.knn < 1 or self.pseudo_weight <= 0 or self.pos_temp <= 0:
            raise ValueError("knn, pseudo_weight, and pos_temp must be positive")
        self.count_model = OctreeCountModel(hidden=count_hidden)
        self.context_encoder = MWCellTransformerContext(
            d_model=self.d_model, n_head=n_head, n_layer=n_layer,
            knn=self.knn, knn_bnd=knn_bnd, bnd_cutoff=bnd_cutoff,
            knn_pot=knn_pot,
        )
        self.position_head = AnchoredCubeCat3Head(self.d_model, num_bins=self.cat_bins)

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def dtype(self):
        return next(self.parameters()).dtype

    @staticmethod
    def _factorial_term(counts: torch.Tensor, N: int, dtype) -> torch.Tensor:
        work = counts.to(torch.float64)
        value = torch.lgamma(work + 1.0).sum() - math.lgamma(int(N) + 1)
        return value.to(dtype=dtype)

    @staticmethod
    def _aux_log_prob(h: float) -> float:
        return -3.0 * math.log(float(h)) - math.log(48.0) - LOG_COLOR_ORDERS

    @staticmethod
    def _count_stencil(counts: torch.Tensor, cell_index: torch.Tensor, G: int,
                       N: int, dtype) -> torch.Tensor:
        grid = counts.reshape(G, G, G)
        values = []
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                for dk in (-1, 0, 1):
                    idx = (
                        int((int(cell_index[0]) + di) % G),
                        int((int(cell_index[1]) + dj) % G),
                        int((int(cell_index[2]) + dk) % G),
                    )
                    values.append(grid[idx].to(dtype) / max(int(N), 1))
        return torch.stack(values)

    def _anchor(self, earlier_y: torch.Tensor, prefix_y: torch.Tensor,
                center: torch.Tensor, L: float, h: float) -> torch.Tensor:
        if earlier_y.numel() and prefix_y.numel():
            points = torch.cat((earlier_y, prefix_y), 0)
        elif earlier_y.numel():
            points = earlier_y
        else:
            points = prefix_y
        if points.numel():
            rel = _wrap_pm(points - center[None], L)
            d2 = rel.square().sum(-1)
            order = torch.argsort(d2, stable=True)[:self.knn]
            selected = rel[order]
            selected_d2 = d2[order]
            weights = torch.exp(-0.5 * selected_d2 / (float(h) ** 2))
            offset = (weights[:, None] * selected).sum(0) / (
                weights.sum() + self.pseudo_weight
            )
        else:
            offset = center.new_zeros(3)
        return clip_anchor(0.5 + offset / float(h))

    def _cell_context(self, counts: torch.Tensor, cell_index: torch.Tensor, K: int, t: int,
                      stage: int, physical_color: int, earlier_y: torch.Tensor,
                      prefix_y: torch.Tensor, N: int, L: float, G: int) -> tuple[torch.Tensor, torch.Tensor]:
        h = float(L) / int(G)
        center = (cell_index.to(dtype=self.dtype, device=self.device) + 0.5) * h
        earlier_y = earlier_y.to(device=self.device, dtype=self.dtype).reshape(-1, 3)
        prefix_y = prefix_y.to(device=self.device, dtype=self.dtype).reshape(-1, 3)
        anchor = self._anchor(earlier_y, prefix_y, center, L, h)
        stencil = self._count_stencil(counts.to(self.device), cell_index, G, N, self.dtype)
        # First three entries retain the July scaffold slot-projection shape.
        slot = center.new_tensor(
            [
                t / max(K - 1, 1),
                (anchor - 0.5).norm() / (3.0 ** 0.5 * 0.5),
                h / 2.5,
            ]
        )
        cell = center.new_tensor(
            [
                K / 16.0,
                t / 16.0,
                (K - t) / 16.0,
                stage / 7.0,
                physical_color / 7.0,
                math.log1p(N) / math.log1p(512),
            ]
        )
        hidden = self.context_encoder(
            earlier_y, prefix_y, center + (anchor - 0.5) * h,
            stencil, slot, cell, L,
        )
        return hidden, anchor

    def _cell_context_many(self, tasks):
        """Batched counterpart of :meth:`_cell_context` for one causal rank.

        ``tasks`` contain only strict-earlier particles and independent current
        cell prefixes. The method therefore changes execution order only, not
        the q0 factorization.
        """
        if not tasks:
            return []
        rows, anchors = [], []
        for task in tasks:
            counts, cell_index, K, t, stage, physical_color, earlier_y, prefix_y, N, L, G = task
            h = float(L) / int(G)
            center = (cell_index.to(dtype=self.dtype, device=self.device) + 0.5) * h
            earlier_y = earlier_y.to(device=self.device, dtype=self.dtype).reshape(-1, 3)
            prefix_y = prefix_y.to(device=self.device, dtype=self.dtype).reshape(-1, 3)
            anchor = self._anchor(earlier_y, prefix_y, center, L, h)
            stencil = self._count_stencil(counts.to(self.device), cell_index, G, N, self.dtype)
            slot = torch.stack((
                center.new_tensor(t / max(K - 1, 1)),
                (anchor - 0.5).norm() / (3.0 ** 0.5 * 0.5),
                center.new_tensor(h / 2.5),
            ))
            cell = center.new_tensor((
                K / 16.0, t / 16.0, (K - t) / 16.0, stage / 7.0,
                physical_color / 7.0, math.log1p(N) / math.log1p(512),
            ))
            rows.append({
                "earlier": earlier_y, "prefix": prefix_y,
                "origin": center + (anchor - 0.5) * h,
                "count_stencil": stencil, "slot_features": slot,
                "cell_features": cell, "L": L,
            })
            anchors.append(anchor)
        hidden = self.context_encoder.forward_many(rows)
        return list(zip(hidden, anchors))

    def _axis_cost(self, fixed_u, axis, anchor, cell_index, h,
                   earlier, prefix, L):
        flow = self.position_head.base
        centers = flow._ctr(torch.arange(flow.n, device=self.device))
        return self.context_encoder.axis_cost(
            fixed_u, axis, anchor, cell_index, h, earlier, prefix, L, centers,
            mw_pair_tilt=self.mw_pair_tilt,
            mw_three_tilt=self.mw_three_tilt,
            pair_cap=self.pair_cap, three_cap=self.three_cap,
        )

    def _position_log_prob(self, hidden, local, anchor, cell_index, h,
                           earlier, prefix, L):
        flow = self.position_head.base
        width = torch.as_tensor(h, device=local.device, dtype=local.dtype)
        valid = ((local >= 0.0) & (local < width)).all()
        if not bool(valid):
            return local.new_full((), float("-inf"))
        upper = torch.nextafter(width, torch.zeros_like(width))
        u, inverse_logdet = anchored_cube_inverse(
            torch.minimum(local.clamp_min(0.0), upper), anchor, width,
            eps=self.position_head.anchor_eps,
        )
        ba, bb, bc = flow._bin(u[0]), flow._bin(u[1]), flow._bin(u[2])
        z = torch.zeros(3, device=u.device, dtype=u.dtype)
        ua = z.clone(); ua[0] = flow._ctr(ba)
        ub = ua.clone(); ub[1] = flow._ctr(bb)
        logits_a = (flow.head_a(hidden) - self._axis_cost(
            z, 0, anchor, cell_index, h, earlier, prefix, L
        )) / self.pos_temp
        logits_b = (flow.head_b(hidden + flow.bin_a_emb(ba)) - self._axis_cost(
            ua, 1, anchor, cell_index, h, earlier, prefix, L
        )) / self.pos_temp
        logits_c = (flow.head_c(hidden + flow.bin_a_emb(ba) + flow.bin_b_emb(bb))
                    - self._axis_cost(
                        ub, 2, anchor, cell_index, h, earlier, prefix, L
                    )) / self.pos_temp
        return (
            F.log_softmax(logits_a, -1)[ba]
            + F.log_softmax(logits_b, -1)[bb]
            + F.log_softmax(logits_c, -1)[bc]
            - flow._logbw3 + inverse_logdet
        )

    @torch.no_grad()
    def _sample_position(self, hidden, anchor, cell_index, h,
                         earlier, prefix, L, gen=None):
        flow = self.position_head.base
        z = torch.zeros(3, device=self.device, dtype=self.dtype)
        logits_a = (flow.head_a(hidden) - self._axis_cost(
            z, 0, anchor, cell_index, h, earlier, prefix, L
        )) / self.pos_temp
        la = F.log_softmax(logits_a, -1)
        ba = torch.multinomial(la.exp(), 1, generator=gen)[0]
        ua = z.clone(); ua[0] = flow._ctr(ba)
        logits_b = (flow.head_b(hidden + flow.bin_a_emb(ba)) - self._axis_cost(
            ua, 1, anchor, cell_index, h, earlier, prefix, L
        )) / self.pos_temp
        lb = F.log_softmax(logits_b, -1)
        bb = torch.multinomial(lb.exp(), 1, generator=gen)[0]
        ub = ua.clone(); ub[1] = flow._ctr(bb)
        logits_c = (flow.head_c(hidden + flow.bin_a_emb(ba) + flow.bin_b_emb(bb))
                    - self._axis_cost(
                        ub, 2, anchor, cell_index, h, earlier, prefix, L
                    )) / self.pos_temp
        lc = F.log_softmax(logits_c, -1)
        bc = torch.multinomial(lc.exp(), 1, generator=gen)[0]
        dither = (torch.rand(3, device=self.device, dtype=self.dtype, generator=gen) - 0.5) * flow.bw
        u = torch.stack((flow._ctr(ba), flow._ctr(bb), flow._ctr(bc))) + dither
        local, forward_logdet = anchored_cube_forward(
            u, anchor, h, eps=self.position_head.anchor_eps,
        )
        latent_lp = la[ba] + lb[bb] + lc[bc] - flow._logbw3
        return local, latent_lp - forward_logdet

    def _score_one_terms(self, x: torch.Tensor, priorities: torch.Tensor, shift: torch.Tensor,
                         orientation_index: int, color_order: torch.Tensor, L: float,
                         G: int):
        """Return auxiliary/count base and one exact position factor per occupied cell."""
        N, h = x.shape[0], float(L) / int(G)
        if priorities.shape != (N,) or bool(((priorities < 0) | (priorities > 1)).any()):
            return x.new_full((), float("-inf")), []
        if bool(((shift < 0) | (shift >= h)).any()) or not 0 <= int(orientation_index) < 48:
            return x.new_full((), float("-inf")), []
        try:
            validate_color_order(color_order)
        except ValueError:
            return x.new_full((), float("-inf")), []
        orientation = cubic_orientations(x.device, x.dtype)[int(orientation_index)]
        grouping = group_by_cell_priority(x, priorities, L, G, shift, orientation)
        counts = grouping.counts
        base_log_prob = self.count_model.count_log_prob(counts, N, G)
        if not torch.isfinite(base_log_prob):
            return base_log_prob, []
        base_log_prob = base_log_prob + x.new_tensor(self._aux_log_prob(h))
        base_log_prob = base_log_prob + self._factorial_term(counts, N, x.dtype)

        _, local = cell_local_coordinates(x, L, G, shift, orientation)
        oriented = to_oriented(x, L, shift, orientation)
        stages = color_stages(color_order)
        schedule = causal_cell_order(G, color_order)
        factors = []
        stage_width = int(G) ** 3 // 8
        for stage in range(8):
            earlier = oriented[stages[grouping.colors] < stage]
            active = []
            for cell_id_tensor in schedule[stage * stage_width:(stage + 1) * stage_width]:
                cell_id = int(cell_id_tensor)
                members = grouping.permutation[grouping.cell_ids[grouping.permutation] == cell_id]
                K = members.numel()
                if K:
                    cell_index = unflatten_cell_indices(cell_id_tensor, G)
                    physical_color = int(cell_colors(cell_index))
                    if int(stages[physical_color]) != stage:
                        raise AssertionError("causal schedule/stage mismatch")
                    active.append({
                        "cell_id": cell_id, "cell_index": cell_index, "members": members,
                        "K": int(K), "physical_color": physical_color,
                        "log_prob": x.new_zeros(()),
                    })
            # At rank t every context depends only on earlier colors and its
            # own prefix. Thus all active cells can share one transformer call.
            for t in range(max((row["K"] for row in active), default=0)):
                rank_rows = [row for row in active if row["K"] > t]
                tasks = [
                    (counts, row["cell_index"], row["K"], t, stage, row["physical_color"],
                     earlier, oriented[row["members"][:t]], N, L, G)
                    for row in rank_rows
                ]
                for row, (hidden, anchor) in zip(rank_rows, self._cell_context_many(tasks)):
                    member = row["members"][t]
                    prefix = oriented[row["members"][:t]]
                    row["log_prob"] = row["log_prob"] + self._position_log_prob(
                        hidden, local[member], anchor, row["cell_index"], h,
                        earlier, prefix, L,
                    )
            factors.extend((row["cell_id"], row["K"], row["log_prob"]) for row in active)
        return base_log_prob, factors

    def _score_one(self, x: torch.Tensor, priorities: torch.Tensor, shift: torch.Tensor,
                   orientation_index: int, color_order: torch.Tensor, L: float,
                   G: int) -> torch.Tensor:
        base, factors = self._score_one_terms(
            x, priorities, shift, orientation_index, color_order, L, G,
        )
        return base + sum((term for _, _, term in factors), base.new_zeros(()))

    def log_prob(self, state: AugmentedCellState, L: float, G: int) -> torch.Tensor:
        """Score a batch, recomputing every grouping from the augmented state."""
        if state.x.ndim != 3 or state.x.shape[-1] != 3:
            raise ValueError("state.x must be [B,N,3]")
        B, N = state.x.shape[:2]
        if state.priorities.shape != (B, N) or state.shift.shape != (B, 3) \
                or state.orientation_index.shape != (B,) or state.color_order.shape != (B, 8):
            raise ValueError("inconsistent augmented-state shapes")
        return torch.stack([
            self._score_one(
                state.x[b], state.priorities[b], state.shift[b],
                int(state.orientation_index[b]), state.color_order[b], L, G,
            )
            for b in range(B)
        ])

    def position_terms_by_occupancy(self, state: AugmentedCellState, L: float, G: int):
        """Exact teacher-forced factors grouped by cell occupancy.

        The return value retains autograd graphs, so :meth:`balanced_nll` can
        equalize occupancy regimes without altering the normalized density used
        for evaluation or SMC.
        """
        base_terms, by_k = [], {}
        for b in range(state.x.shape[0]):
            base, factors = self._score_one_terms(
                state.x[b], state.priorities[b], state.shift[b],
                int(state.orientation_index[b]), state.color_order[b], L, G,
            )
            base_terms.append(base)
            for _, K, value in factors:
                by_k.setdefault(int(K), []).append(value)
        return torch.stack(base_terms), {
            K: torch.stack(values) for K, values in by_k.items()
        }

    def balanced_nll(self, state: AugmentedCellState, L: float, G: int):
        """Count NLL plus equal-weight occupied-cell-K position NLL.

        This is a training reweighting only.  The normalized model remains
        :meth:`log_prob`; validation always reports both the physical mean and
        the reweighted objective separately.
        """
        N = int(state.x.shape[1])
        base, by_k = self.position_terms_by_occupancy(state, L, G)
        count_nll = (-base / N).mean()
        per_k = {K: (-values / K).mean() for K, values in by_k.items()}
        position_nll = torch.stack(list(per_k.values())).mean()
        return count_nll + position_nll, {
            "count_nll": count_nll.detach(), "position_nll": position_nll.detach(),
            "position_nll_by_K": {K: v.detach() for K, v in per_k.items()},
        }

    def cell_position_log_prob(self, state: AugmentedCellState, batch_index: int,
                               cell_id: int, L: float, G: int) -> torch.Tensor:
        """Score one cell factor using the production strict earlier-stage mask."""
        b, cell_id = int(batch_index), int(cell_id)
        x, priorities = state.x[b], state.priorities[b]
        if not 0 <= cell_id < int(G) ** 3:
            raise ValueError("cell_id outside grid")
        _, factors = self._score_one_terms(
            x, priorities, state.shift[b], int(state.orientation_index[b]),
            state.color_order[b], L, G,
        )
        for term_id, _, value in factors:
            if term_id == cell_id:
                return value
        return x.new_zeros(())

    @torch.no_grad()
    def _sample_one(self, N: int, L: float, G: int, gen=None):
        h = float(L) / int(G)
        shift = torch.rand(3, device=self.device, dtype=self.dtype, generator=gen) * h
        orientation, orientation_index = sample_orientation(
            device=self.device, dtype=self.dtype, gen=gen
        )
        color_order = sample_color_order(device=self.device, gen=gen)
        counts, count_log_prob = self.count_model.sample_counts(N, G, gen=gen)
        priorities_by_cell = {}
        for cell_id in range(int(G) ** 3):
            K = int(counts[cell_id])
            priorities_by_cell[cell_id] = torch.sort(
                torch.rand(K, device=self.device, dtype=self.dtype, generator=gen)
            ).values

        y_by_cell = {}
        position_log_prob = self.dtype_tensor(0.0)
        stages = color_stages(color_order)
        schedule = causal_cell_order(G, color_order)
        for stage in range(8):
            stage_cells = schedule[stage * (int(G) ** 3 // 8):(stage + 1) * (int(G) ** 3 // 8)]
            earlier_parts = [y_by_cell[c] for c in y_by_cell]
            earlier = (
                torch.cat(earlier_parts, 0) if earlier_parts
                else torch.empty(0, 3, device=self.device, dtype=self.dtype)
            )
            pending = {}
            for cell_id_tensor in stage_cells:
                cell_id = int(cell_id_tensor)
                K = int(counts[cell_id])
                cell_index = unflatten_cell_indices(cell_id_tensor, G)
                physical_color = int(cell_colors(cell_index))
                if int(stages[physical_color]) != stage:
                    raise AssertionError("causal schedule/stage mismatch")
                pending[cell_id] = {"cell_index": cell_index, "K": K,
                                    "physical_color": physical_color, "prefix": []}
            for t in range(max((row["K"] for row in pending.values()), default=0)):
                rank_rows = [(cell_id, row) for cell_id, row in pending.items() if row["K"] > t]
                tasks = []
                for _, row in rank_rows:
                    prefix = (torch.stack(row["prefix"]) if row["prefix"] else
                              torch.empty(0, 3, device=self.device, dtype=self.dtype))
                    tasks.append((
                        counts, row["cell_index"], row["K"], t, stage,
                        row["physical_color"], earlier, prefix, N, L, G,
                    ))
                for (_, row), (hidden, anchor) in zip(rank_rows, self._cell_context_many(tasks)):
                    prefix = (torch.stack(row["prefix"]) if row["prefix"] else
                              torch.empty(0, 3, device=self.device, dtype=self.dtype))
                    local, lp = self._sample_position(
                        hidden, anchor, row["cell_index"], h, earlier, prefix, L, gen=gen
                    )
                    row["prefix"].append(row["cell_index"].to(self.dtype) * h + local)
                    position_log_prob = position_log_prob + lp
            pending = {
                cell_id: (torch.stack(row["prefix"]) if row["prefix"] else
                          torch.empty(0, 3, device=self.device, dtype=self.dtype))
                for cell_id, row in pending.items()
            }
            y_by_cell.update(pending)

        y = torch.cat([y_by_cell[cell_id] for cell_id in range(int(G) ** 3)], 0)
        priorities = torch.cat(
            [priorities_by_cell[cell_id] for cell_id in range(int(G) ** 3)], 0
        )
        x = from_oriented(y, L, shift, orientation)
        row_permutation = torch.randperm(N, device=self.device, generator=gen)
        x, priorities = x[row_permutation], priorities[row_permutation]
        log_prob = (
            count_log_prob + position_log_prob + self.dtype_tensor(self._aux_log_prob(h))
            + self._factorial_term(counts, N, self.dtype)
        )
        return x, priorities, shift, orientation_index, color_order, counts, log_prob

    def dtype_tensor(self, value) -> torch.Tensor:
        return torch.as_tensor(value, device=self.device, dtype=self.dtype)

    @torch.no_grad()
    def sample(self, B: int, N: int, L: float, G: int, gen=None) \
            -> tuple[AugmentedCellState, torch.Tensor]:
        """Sample labeled augmented states and independently accumulated path logq."""
        if int(N) < 1:
            raise ValueError("MWCellQ0 requires N>=1")
        rows = [self._sample_one(int(N), L, G, gen=gen) for _ in range(int(B))]
        state = AugmentedCellState(
            x=torch.stack([row[0] for row in rows]),
            priorities=torch.stack([row[1] for row in rows]),
            shift=torch.stack([row[2] for row in rows]),
            orientation_index=torch.tensor(
                [row[3] for row in rows], device=self.device, dtype=torch.long
            ),
            color_order=torch.stack([row[4] for row in rows]),
            debug_counts=torch.stack([row[5] for row in rows]),
        )
        return state, torch.stack([row[6] for row in rows])
