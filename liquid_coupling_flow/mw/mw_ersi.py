"""Deployment wrapper for a trained mW traceable-eRSI flow."""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from liquid_coupling_flow.mw.mw_ersi_common import _add_learndiffeq_path
from liquid_coupling_flow.mw.mw_ersi_train import _make_flow


@dataclass
class MWeRSIFlow:
    """Fixed-solver eRSI proposal with a deterministic composed likelihood.

    RK4 with 40 points is the initial conservative production setting.  The
    solver-convergence gate exposes the setting as data rather than treating it
    as a hidden invariant.
    """

    flow: object
    N: int
    L: float
    device: torch.device
    n_steps: int = 40
    method: str = "rk4"
    LOGQ_CONST = False

    def _species(self, B: int) -> torch.Tensor:
        return torch.zeros(B, self.N, dtype=torch.long, device=self.device)

    def _base_logq(self, B: int, dtype=torch.float32) -> torch.Tensor:
        return torch.full((B,), -self.N * 3 * math.log(self.L), device=self.device, dtype=dtype)

    def _sample_base(self, B: int, gen=None) -> torch.Tensor:
        if gen is not None and gen.device.type != self.device.type:
            raise ValueError(f"generator device {gen.device} does not match model device {self.device}")
        return torch.rand(B, self.N, 3, device=self.device, generator=gen) * self.L

    def _sample_and_logq(self, B: int, gen=None, *, n_steps: int | None = None):
        steps = self.n_steps if n_steps is None else int(n_steps)
        x0 = self._sample_base(B, gen)
        a0 = self._species(B)
        _, x, log_jac = self.flow.sample(
            a0, x0, method=self.method, n_steps=steps, return_log_jac=True,
            keep_intermediates=False, approx=False,
        )
        return torch.remainder(x, self.L), self._base_logq(B, x.dtype) + log_jac.flatten()

    def sample_and_logq(self, B: int, gen=None):
        """Draw ``B`` samples and their forward-composed log density in one ODE pass."""
        self.flow.eval()
        return self._sample_and_logq(B, gen)

    def sample(self, B: int, gen=None):
        return self.sample_and_logq(B, gen)[0]

    def _log_q(self, X: torch.Tensor, *, n_steps: int | None = None) -> torch.Tensor:
        if X.ndim != 3 or X.shape[1:] != (self.N, 3):
            raise ValueError(f"expected X [B,{self.N},3], got {tuple(X.shape)}")
        steps = self.n_steps if n_steps is None else int(n_steps)
        x = torch.remainder(X.to(self.device), self.L)
        a = self._species(x.shape[0])
        return self.flow.log_prob(a, x, method=self.method, n_steps=steps, approx=False)

    def log_q(self, X: torch.Tensor) -> torch.Tensor:
        """Score arbitrary wrapped configurations via the reverse-time ODE."""
        self.flow.eval()
        return self._log_q(X)

    def as_generator_base(self):
        return _GeneratorShim(self)


class _GeneratorShim:
    LOGQ_CONST = False

    def __init__(self, model: MWeRSIFlow):
        self.model, self.N, self.L = model, model.N, model.L

    def sample(self, B, gen=None):
        return self.model.sample(B, gen)

    def log_q(self, x):
        return self.model.log_q(x)


def load_ersi(path: str, device: str | torch.device = "cpu") -> MWeRSIFlow:
    """Load a portable velocity-only checkpoint produced by :func:`train`."""
    _add_learndiffeq_path()
    dev = torch.device(device)
    ck = torch.load(path, map_location=dev, weights_only=False)
    required = {"state_dict", "N", "K", "hidden_nf", "n_layers", "L", "dim_phys", "n_species"}
    missing = required.difference(ck)
    if missing:
        raise ValueError(f"{path} is not an mW eRSI checkpoint; missing {sorted(missing)}")
    if ck["dim_phys"] != 3 or ck["n_species"] != 1:
        raise ValueError("MWeRSIFlow supports only 3-D monatomic checkpoints")
    flow = _make_flow(N=int(ck["N"]), K=int(ck["K"]), hidden_nf=int(ck["hidden_nf"]),
                      n_layers=int(ck["n_layers"]), L=float(ck["L"]), lr=1e-3, ot=bool(ck.get("ot", True)))
    flow.b.load_state_dict(ck["state_dict"], strict=True)
    flow.to(dev).eval()
    solver = ck.get("solver", {})
    return MWeRSIFlow(flow=flow, N=int(ck["N"]), L=float(ck["L"]), device=dev,
                      n_steps=int(solver.get("n_steps", 40)), method=solver.get("method", "rk4"))


def solver_convergence(model: MWeRSIFlow, X: torch.Tensor, steps_list=(10, 20, 40, 80)) -> dict:
    """Return deterministic reverse-time scores for each requested solver resolution."""
    return {int(steps): model._log_q(X, n_steps=int(steps)).detach().cpu() for steps in steps_list}


def normalization_check(model: MWeRSIFlow, n_quad: int = 4, seed: int = 0, *, chunk: int = 512) -> float:
    """Full-box midpoint quadrature of ``q`` for N=2 (six-dimensional) only.

    A previous draft proposed using an N=8 smoke model here; no finite grid can
    quadrature its 24-dimensional joint.  Restricting this gate to N=2 makes
    the integral genuine rather than an unnormalised conditional surrogate.
    """
    del seed  # midpoint quadrature is deterministic
    if model.N != 2:
        raise ValueError("normalization_check is a full-joint N=2 gate; build a dedicated N=2 model")
    if n_quad < 2:
        raise ValueError("n_quad must be >= 2")
    grid_1d = (torch.arange(n_quad, device=model.device, dtype=torch.float32) + 0.5) * (model.L / n_quad)
    mesh = torch.meshgrid(*([grid_1d] * 6), indexing="ij")
    X = torch.stack(mesh, dim=-1).reshape(-1, 2, 3)
    total = torch.zeros((), dtype=torch.float64)
    for start in range(0, X.shape[0], chunk):
        total += torch.exp(model.log_q(X[start:start + chunk]).double()).sum()
    return float(total * (model.L / n_quad) ** 6)


__all__ = ["MWeRSIFlow", "load_ersi", "normalization_check", "solver_convergence"]
