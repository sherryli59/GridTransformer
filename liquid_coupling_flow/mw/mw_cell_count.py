"""Normalized full-support octree count distribution for the mW cell base."""
from __future__ import annotations

import itertools
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


CHILD_OFFSETS = tuple(itertools.product((0, 1), repeat=3))


def _tree_depth(G: int) -> int:
    G = int(G)
    if G < 2 or G & (G - 1):
        raise ValueError(f"G must be a power of two >=2, got {G}")
    return int(math.log2(G))


def count_factor_count(G: int) -> int:
    """Number of categorical choices in a complete ``G^3`` octree."""
    _tree_depth(G)  # validate before returning the closed form
    # Seven choices per internal node and (G**3 - 1) / 7 internal nodes.
    return int(G) ** 3 - 1


def integer_compositions(total: int, parts: int = 8):
    """Yield all ordered nonnegative ``parts``-compositions of ``total``."""
    total, parts = int(total), int(parts)
    if total < 0 or parts < 1:
        raise ValueError("invalid composition size")
    if parts == 1:
        yield (total,)
        return
    for first in range(total + 1):
        for suffix in integer_compositions(total - first, parts - 1):
            yield (first,) + suffix


class OctreeCountModel(nn.Module):
    """Shared seven-factor integer split model with unrestricted count support.

    Each internal octree node factors an eight-child composition into seven
    categorical draws on ``0..remaining``; the eighth child receives the exact
    remainder. Candidate logits are produced by one shared MLP, so the parameter
    count is independent of N and tree depth.
    """

    def __init__(self, hidden: int = 96):
        super().__init__()
        self.hidden = int(hidden)
        # parent, remaining, allocated, log parent, level, child, xyz,
        # candidate/remaining, candidate/N, remaining/parent = 12 scalars.
        self.score = nn.Sequential(
            nn.Linear(12, self.hidden),
            nn.SiLU(),
            nn.Linear(self.hidden, self.hidden),
            nn.SiLU(),
            nn.Linear(self.hidden, 1),
        )
        # The residual starts at zero, so the initial model is the symmetric
        # multinomial base below rather than a random, lopsided composition.
        nn.init.zeros_(self.score[-1].weight)
        nn.init.zeros_(self.score[-1].bias)

    def _candidate_logits(self, *, parent: int, remaining: int, allocated: int,
                          level: int, depth: int, coord: tuple[int, int, int],
                          child: int, N: int, device, dtype) -> torch.Tensor:
        if remaining < 0 or parent < 0 or remaining > parent:
            raise ValueError("invalid split state")
        values = torch.arange(remaining + 1, device=device, dtype=dtype)
        denom_N = max(int(N), 1)
        denom_parent = max(int(parent), 1)
        denom_remaining = max(int(remaining), 1)
        level_width = max(1, 1 << int(level))
        common = torch.tensor(
            [
                parent / denom_N,
                remaining / denom_N,
                allocated / denom_N,
                math.log1p(parent) / math.log1p(denom_N),
                level / max(int(depth), 1),
                child / 7.0,
                coord[0] / level_width,
                coord[1] / level_width,
                coord[2] / level_width,
            ],
            device=device,
            dtype=dtype,
        ).expand(remaining + 1, -1)
        candidate = torch.stack(
            (
                values / denom_remaining,
                values / denom_N,
                torch.full_like(values, remaining / denom_parent),
            ),
            -1,
        )
        residual = self.score(torch.cat((common, candidate), -1)).squeeze(-1)
        # Conditional of an exchangeable multinomial split.  After the first
        # ``child`` allocations, the next allocation is Binomial(remaining,
        # 1 / number_of_unassigned_children).  It has full support, so q0
        # remains exact while avoiding a uniform-composition prior that gives
        # pathological 20+ particle cells before MLE has trained.
        m = 8 - int(child)
        p = 1.0 / m
        log_binomial = (
            torch.lgamma(values.new_tensor(float(remaining + 1)))
            - torch.lgamma(values + 1.0)
            - torch.lgamma(values.new_tensor(float(remaining)) - values + 1.0)
            + values * math.log(p)
            + (float(remaining) - values) * math.log1p(-p)
        )
        return log_binomial + residual

    def split_log_prob(self, children: torch.Tensor, *, N: int | None = None,
                       level: int = 0, depth: int = 1,
                       coord: tuple[int, int, int] = (0, 0, 0)) -> torch.Tensor:
        """Score one ordered eight-child composition."""
        children = torch.as_tensor(children, device=next(self.parameters()).device).long()
        if children.shape != (8,) or bool((children < 0).any()):
            raise ValueError("children must be a nonnegative length-eight vector")
        parent = int(children.sum())
        N = parent if N is None else int(N)
        if N < parent:
            raise ValueError("N cannot be smaller than the node count")
        dtype = next(self.parameters()).dtype
        log_prob = next(self.parameters()).new_zeros(())
        remaining = parent
        for child in range(7):
            value = int(children[child])
            if value > remaining:
                return log_prob.new_full((), float("-inf"))
            logits = self._candidate_logits(
                parent=parent,
                remaining=remaining,
                allocated=parent - remaining,
                level=level,
                depth=depth,
                coord=coord,
                child=child,
                N=N,
                device=log_prob.device,
                dtype=dtype,
            )
            log_prob = log_prob + F.log_softmax(logits, 0)[value]
            remaining -= value
        if int(children[7]) != remaining:
            return log_prob.new_full((), float("-inf"))
        return log_prob

    def sample_split(self, parent: int, *, N: int | None = None, level: int = 0,
                     depth: int = 1, coord: tuple[int, int, int] = (0, 0, 0),
                     gen=None) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample one node composition and return its path log probability."""
        parent = int(parent)
        if parent < 0:
            raise ValueError("parent count must be nonnegative")
        N = parent if N is None else int(N)
        if N < parent:
            raise ValueError("N cannot be smaller than the node count")
        parameter = next(self.parameters())
        values, log_prob = [], parameter.new_zeros(())
        remaining = parent
        for child in range(7):
            logits = self._candidate_logits(
                parent=parent,
                remaining=remaining,
                allocated=parent - remaining,
                level=level,
                depth=depth,
                coord=coord,
                child=child,
                N=N,
                device=parameter.device,
                dtype=parameter.dtype,
            )
            log_probs = F.log_softmax(logits, 0)
            value = int(torch.multinomial(log_probs.exp(), 1, generator=gen))
            values.append(value)
            log_prob = log_prob + log_probs[value]
            remaining -= value
        values.append(remaining)
        return torch.tensor(values, device=parameter.device, dtype=torch.long), log_prob

    def sample_counts(self, N: int, G: int, shift=None, orientation=None,
                      gen=None) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample flattened leaf counts summing exactly to N."""
        del shift, orientation  # reserved conditioning inputs; q remains normalized without them.
        N, depth = int(N), _tree_depth(G)
        if N < 0:
            raise ValueError("N must be nonnegative")
        parameter = next(self.parameters())
        nodes = {(0, 0, 0): N}
        total_log_prob = parameter.new_zeros(())
        for level in range(depth):
            children_at_level = {}
            for coord in sorted(nodes):
                split, lp = self.sample_split(
                    nodes[coord], N=N, level=level, depth=depth, coord=coord, gen=gen
                )
                total_log_prob = total_log_prob + lp
                for offset, value in zip(CHILD_OFFSETS, split.tolist()):
                    child_coord = tuple(2 * coord[d] + offset[d] for d in range(3))
                    children_at_level[child_coord] = value
            nodes = children_at_level
        counts = torch.zeros(int(G), int(G), int(G), dtype=torch.long, device=parameter.device)
        for coord, value in nodes.items():
            counts[coord] = value
        return counts.reshape(-1), total_log_prob

    def count_log_prob(self, counts: torch.Tensor, N: int, G: int,
                       shift=None, orientation=None) -> torch.Tensor:
        """Score a flattened or cubic leaf count vector under the octree."""
        del shift, orientation
        N, depth, G = int(N), _tree_depth(G), int(G)
        counts = torch.as_tensor(counts, device=next(self.parameters()).device).long()
        if counts.numel() != G ** 3:
            raise ValueError(f"expected {G ** 3} leaf counts, got {counts.numel()}")
        leaves = counts.reshape(G, G, G)
        if bool((leaves < 0).any()) or int(leaves.sum()) != N:
            return next(self.parameters()).new_full((), float("-inf"))
        total_log_prob = next(self.parameters()).new_zeros(())
        for level in range(depth):
            n_nodes = 1 << level
            child_span = G // (2 * n_nodes)
            for coord in itertools.product(range(n_nodes), repeat=3):
                child_counts = []
                for offset in CHILD_OFFSETS:
                    child_coord = tuple(2 * coord[d] + offset[d] for d in range(3))
                    slices = tuple(
                        slice(child_coord[d] * child_span, (child_coord[d] + 1) * child_span)
                        for d in range(3)
                    )
                    child_counts.append(leaves[slices].sum())
                children = torch.stack(child_counts)
                total_log_prob = total_log_prob + self.split_log_prob(
                    children, N=N, level=level, depth=depth, coord=coord
                )
        return total_log_prob
