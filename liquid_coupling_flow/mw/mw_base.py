"""Uniform base distribution q0 for the mW annealed-SMC path (Task 5). log q0 is a CONSTANT
(uniform-in-box, i.i.d. per particle), so it must drop out of every mutation acceptance ratio —
LOGQ_CONST=True lets mw_smc.mutation_sweeps certify that reduction BEFORE any neural base exists
(Phase-2 fault isolation)."""
from __future__ import annotations
import math, torch


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
