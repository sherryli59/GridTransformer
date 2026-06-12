from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _num_tril_params(spatial_dim: int) -> int:
    d = int(spatial_dim)
    return d * (d + 1) // 2


def _build_scale_tril(raw_tril: torch.Tensor, spatial_dim: int) -> torch.Tensor:
    if raw_tril.ndim != 4:
        raise ValueError(f"raw_tril must be [B,T,M,P], got {tuple(raw_tril.shape)}")
    d = int(spatial_dim)
    expected = _num_tril_params(d)
    if raw_tril.shape[-1] != expected:
        raise ValueError(
            f"raw_tril trailing dim must be {expected} for spatial_dim={d}, got {raw_tril.shape[-1]}"
        )
    prefix = raw_tril.shape[:-1]
    scale_tril = raw_tril.new_zeros(*prefix, d, d)
    tril_idx = torch.tril_indices(d, d, device=raw_tril.device)
    scale_tril[..., tril_idx[0], tril_idx[1]] = raw_tril
    diag_idx = torch.arange(d, device=raw_tril.device)
    raw_diag = scale_tril[..., diag_idx, diag_idx]
    scale_tril[..., diag_idx, diag_idx] = torch.exp(torch.clamp(raw_diag, min=-10.0, max=5.0))
    return scale_tril


class MDNHead(nn.Module):
    def __init__(
        self,
        d_model: int,
        spatial_dim: int,
        num_mixtures: int = 32,
        *,
        full_covariance: bool = False,
    ) -> None:
        super().__init__()
        if int(spatial_dim) <= 0:
            raise ValueError(f"spatial_dim must be positive, got {spatial_dim}")
        if int(num_mixtures) <= 0:
            raise ValueError(f"num_mixtures must be positive, got {num_mixtures}")
        self.spatial_dim = int(spatial_dim)
        self.num_mixtures = int(num_mixtures)
        self.full_covariance = bool(full_covariance)
        self.pi_proj = nn.Linear(d_model, self.num_mixtures)
        self.mu_proj = nn.Linear(d_model, self.num_mixtures * self.spatial_dim)
        if self.full_covariance:
            self.raw_tril_proj = nn.Linear(
                d_model,
                self.num_mixtures * _num_tril_params(self.spatial_dim),
            )
            self.log_sigma_proj = None
        else:
            self.log_sigma_proj = nn.Linear(d_model, self.num_mixtures * self.spatial_dim)
            self.raw_tril_proj = None

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if x.ndim != 3:
            raise ValueError(f"MDNHead expects [B,T,D], got {tuple(x.shape)}")
        B, T, _ = x.shape
        log_pi = F.log_softmax(self.pi_proj(x), dim=-1)
        mu = self.mu_proj(x).view(B, T, self.num_mixtures, self.spatial_dim)
        if self.full_covariance:
            assert self.raw_tril_proj is not None
            raw_tril = self.raw_tril_proj(x).view(
                B,
                T,
                self.num_mixtures,
                _num_tril_params(self.spatial_dim),
            )
            scale_tril = _build_scale_tril(raw_tril, self.spatial_dim)
            return log_pi, mu, scale_tril

        assert self.log_sigma_proj is not None
        log_sigma = self.log_sigma_proj(x).view(B, T, self.num_mixtures, self.spatial_dim)
        sigma = torch.exp(torch.clamp(log_sigma, min=-10.0, max=5.0))
        return log_pi, mu, sigma


def mdn_loss(
    log_pi: torch.Tensor,
    mu: torch.Tensor,
    scale_param: torch.Tensor,
    target: torch.Tensor,
    *,
    pad_mask: Optional[torch.Tensor] = None,
    box_size: Optional[torch.Tensor] = None,
    return_point_nll: bool = False,
) -> tuple[torch.Tensor, ...]:
    """
    Continuous negative log-likelihood under a diagonal-covariance Gaussian mixture.

    Returns:
      loss: scalar mean NLL normalized by number of valid coordinates
      seq_nll: [B] summed NLL over timesteps
      coord_count: [B] number of valid coordinates contributing to each sample
      point_nll: [B, T] per-token NLL (zeroed at padding) — only when
        ``return_point_nll=True`` (used for jump-stratified logging)
    """
    if log_pi.ndim != 3:
        raise ValueError(f"log_pi must be [B,T,M], got {tuple(log_pi.shape)}")
    if mu.ndim != 4:
        raise ValueError(f"mu must be [B,T,M,D], got {tuple(mu.shape)}")
    if target.ndim != 3:
        raise ValueError(f"target must be [B,T,D], got {tuple(target.shape)}")
    if log_pi.shape[:2] != target.shape[:2]:
        raise ValueError(
            f"log_pi shape {tuple(log_pi.shape)} is incompatible with target {tuple(target.shape)}"
        )
    if mu.shape[:3] != (*target.shape[:2], log_pi.shape[-1]):
        raise ValueError(
            f"mu/log_pi/target shapes are incompatible: log_pi={tuple(log_pi.shape)}, "
            f"mu={tuple(mu.shape)}, target={tuple(target.shape)}"
        )
    full_covariance = scale_param.ndim == 5
    if full_covariance:
        if scale_param.shape[:3] != mu.shape[:3] or scale_param.shape[-2:] != (mu.shape[-1], mu.shape[-1]):
            raise ValueError(
                "Full-covariance scale_param must be [B,T,M,D,D], got "
                f"{tuple(scale_param.shape)} for mu {tuple(mu.shape)}"
            )
    else:
        if scale_param.ndim != 4:
            raise ValueError(
                f"Diagonal sigma must be [B,T,M,D] or full covariance [B,T,M,D,D], got {tuple(scale_param.shape)}"
            )
        if scale_param.shape != mu.shape:
            raise ValueError(
                f"Diagonal sigma shape {tuple(scale_param.shape)} must match mu shape {tuple(mu.shape)}"
            )
    if mu.shape[-1] != target.shape[-1]:
        raise ValueError(f"target dim {target.shape[-1]} must match mu dim {mu.shape[-1]}")

    B, _, _, coord_dim = mu.shape
    target_exp = target.unsqueeze(-2)  # [B,T,1,D]
    diff = target_exp - mu

    if box_size is not None:
        box = torch.as_tensor(box_size, device=diff.device, dtype=diff.dtype)
        if box.ndim == 1:
            box = box.view(1, 1, -1)
        elif box.ndim == 2:
            box = box.view(box.shape[0], 1, box.shape[1])
        elif box.ndim != 3:
            raise ValueError(f"box_size must be [D], [B,D], or [B,T,D], got {tuple(box.shape)}")
        if box.shape[0] == 1 and B > 1:
            box = box.expand(B, -1, -1)
        if box.shape[0] != B:
            raise ValueError(f"box_size batch mismatch: expected 1 or {B}, got {box.shape[0]}")
        if box.shape[1] == 1 and target.shape[1] > 1:
            box = box.expand(-1, target.shape[1], -1)
        if box.shape[1] != target.shape[1]:
            raise ValueError(
                f"box_size timestep dim must match target length {target.shape[1]}, got {box.shape[1]}"
            )
        if box.shape[-1] != coord_dim:
            raise ValueError(
                f"box_size coord dim must match target dim {coord_dim}, got {box.shape[-1]}"
            )
        box = box.unsqueeze(-2)
        diff = diff - box * torch.round(diff / box.clamp_min(1e-8))

    if full_covariance:
        scale_tril = scale_param
        diag = torch.diagonal(scale_tril, dim1=-2, dim2=-1).clamp_min(1e-8)
        centered = diff.unsqueeze(-1)  # [B,T,M,D,1]
        solved = torch.linalg.solve_triangular(scale_tril, centered, upper=False)
        quad = solved.squeeze(-1).square().sum(dim=-1)
        comp_log_prob = (
            -0.5 * (quad + float(coord_dim) * math.log(2.0 * math.pi))
            - torch.log(diag).sum(dim=-1)
        )
    else:
        sigma = scale_param.clamp_min(1e-8)
        var = sigma.square()
        log_prob_per_dim = -0.5 * diff.square() / var - torch.log(sigma) - 0.5 * math.log(2.0 * math.pi)
        comp_log_prob = log_prob_per_dim.sum(dim=-1)  # [B,T,M]
    point_nll = -torch.logsumexp(log_pi + comp_log_prob, dim=-1)  # [B,T]

    if pad_mask is not None:
        if pad_mask.shape != point_nll.shape:
            raise ValueError(f"pad_mask must be [B,T], got {tuple(pad_mask.shape)}")
        valid = (~pad_mask.bool()).to(point_nll.dtype)
    else:
        valid = torch.ones_like(point_nll, dtype=point_nll.dtype)

    point_nll = point_nll * valid
    seq_nll = point_nll.sum(dim=1)
    coord_count = valid.sum(dim=1).clamp_min(1.0) * float(coord_dim)
    loss = (seq_nll / coord_count).mean()
    if return_point_nll:
        return loss, seq_nll, coord_count, point_nll
    return loss, seq_nll, coord_count


def stratified_nll_means(
    point_nll: torch.Tensor,
    delta_s: torch.Tensor,
    *,
    threshold: float = 4.0,
    pad_mask: Optional[torch.Tensor] = None,
) -> dict[str, Optional[float]]:
    """Split per-token NLL into Hilbert-jump vs local strata by |Δs| (action-plan Phase 3).

    The jump stratum (|Δs| > threshold) is the cheap early discriminator for the
    geometric variants: they exist to fix jump-step conditionals, and the aggregate
    NLL is dominated by local steps. Returns Python floats (None for an empty
    stratum) so callers can feed self.log directly.
    """
    if point_nll.shape != delta_s.shape:
        raise ValueError(
            f"point_nll {tuple(point_nll.shape)} and delta_s {tuple(delta_s.shape)} must match"
        )
    valid = torch.ones_like(point_nll, dtype=torch.bool)
    if pad_mask is not None:
        valid &= ~pad_mask.bool()
    jump = (delta_s.abs() > float(threshold)) & valid
    local = (~(delta_s.abs() > float(threshold))) & valid
    out: dict[str, Optional[float]] = {
        "nll_jump": float(point_nll[jump].mean()) if bool(jump.any()) else None,
        "nll_local": float(point_nll[local].mean()) if bool(local.any()) else None,
        "jump_fraction": float(jump.sum()) / float(valid.sum().clamp_min(1)),
    }
    return out


def compute_log_weight_variance(
    seq_nll_model: torch.Tensor,
    target_energy: torch.Tensor,
    *,
    kT: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if seq_nll_model.ndim != 1:
        raise ValueError(f"seq_nll_model must be [B], got {tuple(seq_nll_model.shape)}")
    if target_energy.ndim != 1:
        raise ValueError(f"target_energy must be [B], got {tuple(target_energy.shape)}")
    if target_energy.shape[0] != seq_nll_model.shape[0]:
        raise ValueError(
            f"Batch size mismatch between target_energy {tuple(target_energy.shape)} and "
            f"seq_nll_model {tuple(seq_nll_model.shape)}"
        )
    kT = float(kT)
    if kT <= 0.0:
        raise ValueError(f"kT must be positive, got {kT}")
    target_log_p = -target_energy.to(device=seq_nll_model.device, dtype=seq_nll_model.dtype) / kT
    log_q = -seq_nll_model
    log_w = target_log_p - log_q
    var_log_w = torch.var(log_w, unbiased=False)
    batch_size = torch.tensor(log_w.shape[0], device=log_w.device, dtype=log_w.dtype)
    log_ess_fraction = (
        2.0 * torch.logsumexp(log_w, dim=0)
        - torch.logsumexp(2.0 * log_w, dim=0)
        - torch.log(batch_size)
    )
    ess_fraction = torch.exp(log_ess_fraction).clamp(max=1.0)
    return var_log_w, target_log_p, log_q, ess_fraction
