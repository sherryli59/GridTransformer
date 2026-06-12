from __future__ import annotations

import math
from typing import Optional, Sequence, Tuple

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..training.ar import MDNHead, compute_log_weight_variance, mdn_loss, stratified_nll_means
from .curve_rail import CurveRailAttention
from .deep_ida import DeepIDABias
from .ida import PeriodicIDA
from .joint_arc_bias import JointEdgeBias
from .rbf_edge_bias import RBFEdgeBias
from ..utils.spatial import build_attention_coords, id_to_center_xyz, wrap_min_image


class EdgeBias(nn.Module):
    """Turn pairwise distances into additive attention bias."""

    def __init__(
        self,
        n_head: int,
        n_bins: int = 32,
        d_dir: int = 16,
        use_dir: bool = True,
        max_dist: float = 4.0,
    ):
        super().__init__()
        self.n_head = n_head
        self.n_bins = n_bins
        self.use_dir = use_dir
        self.max_dist = max(float(max_dist), 1e-8)
        self.max_log_dist = float(math.log1p(self.max_dist))
        self.bin_embed = nn.Embedding(n_bins, n_head)
        if use_dir:
            self.dir_mlp = nn.Sequential(
                nn.Linear(3, d_dir),
                nn.GELU(),
                nn.Linear(d_dir, n_head),
            )

    def _bin_dist(self, d: torch.Tensor) -> torch.LongTensor:
        if self.max_log_dist <= 0.0:
            return torch.zeros_like(d, dtype=torch.long)
        log_d = torch.log1p(d.clamp_min(0.0))
        bins = (log_d / self.max_log_dist) * (self.n_bins - 1)
        return bins.long().clamp(0, self.n_bins - 1)

    def forward(
        self,
        coords: torch.Tensor,
        causal_mask: torch.Tensor,
        *,
        box_size: Optional[torch.Tensor] = None,
        torus: bool = False,
    ) -> torch.Tensor:
        """
        coords: [B, T, C] where C is 2 or 3.
        causal_mask: [B, T, T] bool (True means allowed attention).
        box_size: optional side lengths (C,) or [B, C] used when torus=True.
        returns: [B, nH, T, T]
        """
        if coords.ndim != 3:
            raise ValueError(f"coords must be [B,T,C], got {tuple(coords.shape)}")

        B, T, C = coords.shape
        if C == 2:
            zeros = torch.zeros(B, T, 1, device=coords.device, dtype=coords.dtype)
            coords3 = torch.cat([coords, zeros], dim=-1)
            if box_size is not None:
                bs = torch.as_tensor(box_size, device=coords.device, dtype=coords.dtype)
                if bs.ndim == 1:
                    bs = torch.cat([bs, torch.ones(1, device=bs.device, dtype=bs.dtype)], dim=0)
                elif bs.ndim == 2:
                    pad = torch.ones(bs.shape[0], 1, device=bs.device, dtype=bs.dtype)
                    bs = torch.cat([bs, pad], dim=1)
                box_size = bs
        elif C == 3:
            coords3 = coords
        else:
            raise ValueError(f"coords last dim must be 2 or 3, got {C}")

        diff = coords3[:, :, None, :] - coords3[:, None, :, :]  # [B,T,T,3]
        if torus:
            if box_size is None:
                raise ValueError("box_size is required when torus=True")
            diff = wrap_min_image(diff, box_size)

        bias = self._bias_from_diff(diff)  # [B,nH,T,T]
        return bias.masked_fill(~causal_mask[:, None, :, :], float("-inf"))

    def _bias_from_diff(self, diff: torch.Tensor) -> torch.Tensor:
        """Shared bias math for the full matrix and the incremental row: [..., Q, K, 3] -> [B, nH, Q, K]."""
        dist = torch.linalg.norm(diff, dim=-1)
        bins = self._bin_dist(dist)
        bias = self.bin_embed(bins).permute(0, 3, 1, 2).contiguous()
        if self.use_dir:
            unit_dir = diff / dist.unsqueeze(-1).clamp_min(1e-6)
            bias = bias + self.dir_mlp(unit_dir).permute(0, 3, 1, 2).contiguous()
        return bias

    def bias_row(
        self,
        coords: torch.Tensor,
        *,
        box_size: Optional[torch.Tensor] = None,
        torus: bool = False,
    ) -> torch.Tensor:
        """Bias row for the LAST position attending to the whole prefix: [B, nH, 1, P].

        ``coords`` is [B, P, C] (prefix INCLUDING the new position, last). Equals the
        last causal row of :meth:`forward` on the same coords — O(P) instead of O(P²),
        for KV-cached generation. No causal masking needed (the last row sees all).
        """
        if coords.ndim != 3:
            raise ValueError(f"coords must be [B,P,C], got {tuple(coords.shape)}")
        B, P, C = coords.shape
        if C == 2:
            zeros = torch.zeros(B, P, 1, device=coords.device, dtype=coords.dtype)
            coords3 = torch.cat([coords, zeros], dim=-1)
            if box_size is not None:
                bs = torch.as_tensor(box_size, device=coords.device, dtype=coords.dtype)
                if bs.ndim == 1:
                    bs = torch.cat([bs, torch.ones(1, device=bs.device, dtype=bs.dtype)], dim=0)
                elif bs.ndim == 2:
                    pad = torch.ones(bs.shape[0], 1, device=bs.device, dtype=bs.dtype)
                    bs = torch.cat([bs, pad], dim=1)
                box_size = bs
        elif C == 3:
            coords3 = coords
        else:
            raise ValueError(f"coords last dim must be 2 or 3, got {C}")

        diff = coords3[:, -1:, None, :] - coords3[:, None, :, :]  # [B,1,P,3]
        if torus:
            if box_size is None:
                raise ValueError("box_size is required when torus=True")
            diff = wrap_min_image(diff, box_size)
        return self._bias_from_diff(diff)  # [B,nH,1,P]


class GenerationCache:
    """Per-layer KV cache for incremental autoregressive generation.

    ``layers[i]`` holds the (k, v) tensors of block i over all positions generated so
    far; ``length`` is the number of positions already in the cache. ``arc_s`` holds
    the cumulative normalized arc length per position [B, length] (only used when the
    joint arc bias is enabled).
    """

    __slots__ = ("layers", "length", "arc_s")

    def __init__(self, n_layer: int):
        self.layers: list[Optional[tuple[torch.Tensor, torch.Tensor]]] = [None] * int(n_layer)
        self.length: int = 0
        self.arc_s: Optional[torch.Tensor] = None


def _knn_causal_mask(
    coords: torch.Tensor,
    k: int,
    causal_mask: torch.Tensor,
    *,
    box_size: Optional[torch.Tensor] = None,
    torus: bool = False,
) -> torch.Tensor:
    """Variant G: restrict each query to its k nearest causal keys (plus itself).

    coords [B,T,C], causal_mask [B,T,T] bool -> [B,T,T] bool subset of causal_mask.
    """
    B, T, _ = coords.shape
    diff = coords[:, :, None, :] - coords[:, None, :, :]
    if torus and box_size is not None:
        diff = wrap_min_image(diff, box_size)
    dist = torch.linalg.norm(diff, dim=-1)
    dist = dist.masked_fill(~causal_mask, float("inf"))
    kk = min(int(k) + 1, T)  # +1: self-distance 0 is always among the smallest
    idx = dist.topk(kk, dim=-1, largest=False).indices
    keep = torch.zeros_like(causal_mask)
    keep.scatter_(-1, idx, True)
    keep &= causal_mask
    # The diagonal is always allowed so every row keeps at least one finite logit.
    eye = torch.eye(T, device=coords.device, dtype=torch.bool).unsqueeze(0)
    return keep | (eye & causal_mask)


class CausalSelfAttnWithBias(nn.Module):
    def __init__(self, d_model: int, n_head: int, dropout: float, use_rope: bool = False, rope_max_period: float = 10000.0):
        super().__init__()
        if d_model % n_head != 0:
            raise ValueError(f"d_model={d_model} must be divisible by n_head={n_head}")
        self.n_head = n_head
        self.d_head = d_model // n_head
        _ = use_rope
        _ = rope_max_period
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        bias: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        past_kv: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        return_kv: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        B, T, D = x.shape
        qkv = self.qkv(x).view(B, T, 3, self.n_head, self.d_head)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)  # [B,nH,T,dH]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        if past_kv is not None:
            # KV-cached generation: x holds only NEW positions; bias is [B,nH,T_new,T_total].
            pk, pv = past_kv
            k = torch.cat([pk, k], dim=2)
            v = torch.cat([pv, v], dim=2)

        att = (q @ k.transpose(-2, -1)) / (self.d_head ** 0.5)
        att = att + bias

        if key_padding_mask is not None:
            att = att.masked_fill(key_padding_mask[:, None, None, :], float("-inf"))

        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, D)
        y = self.resid_drop(self.proj(y))
        if return_kv:
            return y, (k, v)
        return y


class GraphormerAR(pl.LightningModule):
    def __init__(
        self,
        K: int,
        d_model: int = 256,
        n_layer: int = 8,
        n_head: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        lr: float = 3e-4,
        weight_decay: float = 1e-2,
        pad_id: Optional[int] = None,
        sos_id: Optional[int] = None,
        input_vocab_size: Optional[int] = None,
        use_edge_bias: bool = True,
        use_ida_pre: bool = False,
        ida_spatial_dim: int = 2,
        torus: bool = False,
        use_dir_bias: bool = True,
        dist_bins: int = 32,
        edge_max_dist: float = 4.0,
        use_rbf_bias: bool = False,
        use_deep_ida: bool = False,
        rbf_n_centers: int = 64,
        rbf_hidden_dim: int = 64,
        coord_dequantize_train: bool = True,
        coord_dequant_width: float = 0.0,
        id_coord_mode: Optional[str] = None,
        cell_grid_size: Optional[int] = None,
        use_pos_emb: bool = False,
        use_density_cond: bool = False,
        use_rope: bool = False,
        rope_max_period: float = 10000.0,
        is_factorized: bool = False,
        use_continuous_head: bool = False,
        continuous_input: bool = False,
        num_mixtures: int = 32,
        full_covariance: bool = False,
        polar: bool = False,
        discrete: bool = False,
        binned_discrete: bool = False,
        lambda_var: float = 0.0,
        lj_kT: float = 1.0,
        lj_epsilon: float = 1.0,
        lj_sigma: float = 1.0,
        lj_cutoff: Optional[float] = None,
        lj_spring_constant: float = 0.5,
        lj_boxlength: float = 10.0,
        lj_periodic: bool = False,
        use_curve_rail: bool = False,
        curve_rail_k: int = 6,
        curve_rail_n_rbf: int = 32,
        curve_rail_max_dist: float = 4.0,
        curve_rail_hidden: int = 64,
        curve_rail_offsets: Optional[Sequence[int]] = None,
        hilbert_resolution: int = 128,
        cell_size: Optional[float] = None,
        curve_rail_mode: str = "lookahead",
        curve_rail_window: float = 1.0,
        curve_rail_reference: str = "absolute",
        curve_rail_residual_target: bool = False,
        arc_repr: bool = False,
        use_joint_arc_bias: bool = False,
        use_refine_rbf_bias: bool = False,
        knn_mask_k: Optional[int] = None,
        jump_delta_s_threshold: float = 4.0,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.K = int(K)
        self.pad_id = pad_id
        self.sos_id = sos_id
        self.lr = lr
        self.weight_decay = weight_decay

        in_vocab = int(input_vocab_size) if input_vocab_size is not None else int(K)
        self.tok_emb = nn.Embedding(in_vocab, d_model)
        self.use_pos_emb = bool(use_pos_emb)
        self.use_density_cond = bool(use_density_cond)
        self.is_factorized = bool(is_factorized)
        self.pos_emb = nn.Embedding(4096, d_model) if self.use_pos_emb else None
        self.density_emb = nn.Linear(1, d_model) if self.use_density_cond else None

        self.use_edge_bias = bool(use_edge_bias)
        self.use_ida_pre = bool(use_ida_pre)
        self.torus = bool(torus)
        self.use_rbf_bias = bool(use_rbf_bias)
        self.use_deep_ida = bool(use_deep_ida)
        if self.use_rbf_bias and self.use_deep_ida:
            raise ValueError("use_rbf_bias and use_deep_ida cannot both be enabled.")

        if self.use_deep_ida:
            self.edge_bias = DeepIDABias(n_head=n_head)
        elif self.use_rbf_bias:
            self.edge_bias = RBFEdgeBias(
                n_head=n_head,
                n_rbf_centers=int(rbf_n_centers),
                max_dist=edge_max_dist,
                hidden_dim=int(rbf_hidden_dim),
            )
        else:
            self.edge_bias = EdgeBias(
                n_head=n_head,
                n_bins=dist_bins,
                use_dir=use_dir_bias,
                max_dist=edge_max_dist,
            )

        self.coord_dequantize_train = bool(coord_dequantize_train)
        self.coord_dequant_width = max(float(coord_dequant_width), 0.0)
        self.discrete = bool(discrete)
        self.binned_discrete = bool(binned_discrete)
        self.use_continuous_head = bool(use_continuous_head)
        self.full_covariance = bool(full_covariance)
        self.polar = bool(polar)
        if self.discrete and self.use_continuous_head:
            raise ValueError("discrete=True is incompatible with use_continuous_head=True.")
        if self.discrete and self.binned_discrete:
            raise ValueError("discrete=True is incompatible with binned_discrete=True.")
        self.output_spatial_dim = int(ida_spatial_dim)
        self.continuous_out_dim = 1 if self.is_factorized else self.output_spatial_dim
        self.axis_emb = nn.Embedding(self.output_spatial_dim, d_model) if self.is_factorized else None
        # arc_repr: predict (Δs, fine_x, fine_y, fine_z) instead of (Δx, Δy, Δz).
        # EdgeBias still uses 3D xyz coordinates (output_spatial_dim); only
        # delta_in_proj and the MDN head switch to the 4D arc dimension.
        self.arc_repr = bool(arc_repr)
        if self.arc_repr and self.is_factorized:
            raise ValueError("arc_repr=True is not supported with is_factorized=True.")
        if self.arc_repr and self.polar:
            raise ValueError("arc_repr=True is not supported with polar=True.")
        self._delta_dim = 4 if self.arc_repr else self.output_spatial_dim
        # Continuous input feedback: project the (continuous) previous displacement into the
        # residual stream instead of relying on the QUANTIZED discrete token.
        self.continuous_input = bool(continuous_input)
        if self.continuous_input:
            if self.is_factorized:
                raise ValueError("continuous_input is not supported with is_factorized=True.")
            if self.polar:
                raise ValueError("continuous_input is not supported with polar=True.")
            self.delta_in_proj = nn.Linear(self._delta_dim, d_model)
        else:
            self.delta_in_proj = None
        self.num_mixtures = int(num_mixtures)
        self.lambda_var = float(lambda_var)
        self.lj_kT = float(lj_kT)

        # Phase-3 probe variants (action-plan). All are exact no-ops at init
        # (zero-init outputs / parameter-free), so they can be warm-started from a
        # baseline checkpoint with strict=False loading.
        self.use_joint_arc_bias = bool(use_joint_arc_bias)
        self.use_refine_rbf_bias = bool(use_refine_rbf_bias)
        self.knn_mask_k = int(knn_mask_k) if knn_mask_k is not None else None
        self.jump_delta_s_threshold = float(jump_delta_s_threshold)
        if self.use_joint_arc_bias:
            if not (self.arc_repr and self.continuous_input):
                raise ValueError(
                    "use_joint_arc_bias requires arc_repr=True and continuous_input=True "
                    "(arc positions are accumulated from the input deltas)."
                )
            self.joint_arc_bias = JointEdgeBias(n_head=n_head, max_dist=edge_max_dist)
        else:
            self.joint_arc_bias = None
        if self.use_refine_rbf_bias:
            # Variant F: dense RBF refinement concentrated near contact (d <= 2 sigma),
            # where log-binned EdgeBias is coarsest and the LJ wall is stiffest.
            self.refine_rbf_bias = RBFEdgeBias(
                n_head=n_head,
                n_rbf_centers=64,
                max_dist=2.0,
                hidden_dim=int(rbf_hidden_dim),
            )
            nn.init.zeros_(self.refine_rbf_bias.mlp[-1].weight)
            nn.init.zeros_(self.refine_rbf_bias.mlp[-1].bias)
        else:
            self.refine_rbf_bias = None
        if self.use_ida_pre:
            self.ida_pre = PeriodicIDA(
                d_model=d_model,
                num_heads=n_head,
                spatial_dim=int(ida_spatial_dim),
                bias=False,
            )
        else:
            self.ida_pre = None

        # Optional "curve rail" (GPS-guide) cross-attention: one instance applied once
        # before the transformer blocks, separate from the neighborhood edge bias/IDA.
        self.use_curve_rail = bool(use_curve_rail)
        # Geometry needed to recompute the rail at generation time (carried in the
        # checkpoint via save_hyperparameters). offsets default to the dataset default.
        if curve_rail_offsets is None:
            from ..data.lj_transferable import DEFAULT_CURVE_RAIL_OFFSETS
            self.curve_rail_offsets: Tuple[int, ...] = tuple(int(o) for o in DEFAULT_CURVE_RAIL_OFFSETS)
        else:
            self.curve_rail_offsets = tuple(int(o) for o in curve_rail_offsets)
        self.hilbert_resolution = int(hilbert_resolution)
        self.cell_size = float(cell_size) if cell_size is not None else None
        self.curve_rail_mode = str(curve_rail_mode)
        self.curve_rail_window = float(curve_rail_window)
        self.curve_rail_reference = str(curve_rail_reference)
        self.curve_rail_residual_target = bool(curve_rail_residual_target)
        if self.use_curve_rail:
            self.rail_attn = CurveRailAttention(
                d_model=d_model,
                n_head=n_head,
                n_rail=int(curve_rail_k),
                n_rbf=int(curve_rail_n_rbf),
                max_dist=float(curve_rail_max_dist),
                hidden_dim=int(curve_rail_hidden),
                dropout=dropout,
            )
        else:
            self.rail_attn = None
        self.id_coord_mode = id_coord_mode
        self.cell_grid_size = int(cell_grid_size) if cell_grid_size is not None else None

        self.blocks = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        "ln1": nn.LayerNorm(d_model),
                        "attn": CausalSelfAttnWithBias(
                            d_model,
                            n_head,
                            dropout,
                            use_rope=use_rope,
                            rope_max_period=rope_max_period,
                        ),
                        "ln2": nn.LayerNorm(d_model),
                        "mlp": nn.Sequential(
                            nn.Linear(d_model, int(mlp_ratio * d_model)),
                            nn.GELU(),
                            nn.Dropout(dropout),
                            nn.Linear(int(mlp_ratio * d_model), d_model),
                            nn.Dropout(dropout),
                        ),
                    }
                )
                for _ in range(n_layer)
            ]
        )
        self.ln_f = nn.LayerNorm(d_model)
        if self.use_continuous_head:
            _head_spatial = self._delta_dim if self.arc_repr else self.continuous_out_dim
            self.head = MDNHead(
                d_model=d_model,
                spatial_dim=_head_spatial,
                num_mixtures=self.num_mixtures,
                full_covariance=self.full_covariance,
            )
        else:
            self.head = nn.Linear(d_model, K)

    def on_fit_start(self) -> None:
        dm = self.trainer.datamodule
        codebook = getattr(dm, "codebook_weight", None)
        if codebook is None:
            return
        codebook = codebook.to(self.device, dtype=self.tok_emb.weight.dtype)
        if codebook.ndim != 2:
            raise ValueError(f"codebook_weight must be rank-2 [K,D], got {tuple(codebook.shape)}")
        if codebook.shape[0] != self.K:
            raise ValueError(f"codebook K mismatch: model K={self.K}, codebook K={codebook.shape[0]}")
        if codebook.shape[1] != self.tok_emb.embedding_dim:
            raise ValueError(
                "Codebook embedding dim must equal AR d_model: "
                f"{codebook.shape[1]} != {self.tok_emb.embedding_dim}"
            )
        with torch.no_grad():
            self.tok_emb.weight[: self.K].copy_(codebook)

    def _shift_with_sos(self, seq: torch.LongTensor) -> torch.LongTensor:
        if self.sos_id is None:
            return torch.cat([seq[:, :1], seq[:, :-1]], dim=1)
        sos = torch.full((seq.size(0), 1), int(self.sos_id), dtype=seq.dtype, device=seq.device)
        return torch.cat([sos, seq[:, :-1]], dim=1)

    def _box_like_coords(self, coords: torch.Tensor, box_size: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if box_size is None:
            return None
        bs = torch.as_tensor(box_size, device=coords.device, dtype=coords.dtype)
        if bs.ndim == 1:
            bs = bs.view(1, -1)
        if bs.ndim != 2:
            raise ValueError(f"box_size must be rank-1 or rank-2, got shape {tuple(bs.shape)}")
        if bs.shape[0] == 1 and coords.shape[0] > 1:
            bs = bs.expand(coords.shape[0], -1)
        if bs.shape[0] != coords.shape[0]:
            raise ValueError(f"box_size batch mismatch: coords B={coords.shape[0]}, box B={bs.shape[0]}")

        c = coords.shape[-1]
        if bs.shape[1] == c:
            return bs
        if bs.shape[1] > c:
            return bs[:, :c]
        pad = torch.ones(bs.shape[0], c - bs.shape[1], device=bs.device, dtype=bs.dtype)
        return torch.cat([bs, pad], dim=1)

    def _apply_training_coord_dequantization(
        self,
        coords: Optional[torch.Tensor],
        box_size: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        if coords is None or (not self.training) or (not self.coord_dequantize_train):
            return coords

        box_for_coords = self._box_like_coords(coords, box_size)
        noise_width: Optional[torch.Tensor] = None

        if box_for_coords is not None and self.cell_grid_size is not None and self.cell_grid_size > 0:
            noise_width = (box_for_coords / float(self.cell_grid_size)).unsqueeze(1)  # [B,1,C]
        elif self.coord_dequant_width > 0.0:
            noise_width = torch.full(
                (1, 1, coords.shape[-1]),
                fill_value=float(self.coord_dequant_width),
                device=coords.device,
                dtype=coords.dtype,
            )

        if noise_width is None:
            return coords

        noisy_coords = coords + (torch.rand_like(coords) - 0.5) * noise_width
        if box_for_coords is not None:
            noisy_coords = torch.remainder(noisy_coords, box_for_coords.unsqueeze(1))
        return noisy_coords

    def _causal_bias(
        self,
        B: int,
        T: int,
        coords: Optional[torch.Tensor],
        box_size: Optional[torch.Tensor],
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        causal = torch.tril(torch.ones(T, T, device=device, dtype=torch.bool)).unsqueeze(0).expand(B, -1, -1)
        if not self.use_edge_bias or coords is None:
            bias = torch.zeros(B, 1, T, T, device=device, dtype=dtype)
            return bias.masked_fill(~causal[:, None, :, :], float("-inf"))
        return self.edge_bias(coords, causal, box_size=box_size, torus=self.torus)

    def _prepare_attention_coords(
        self,
        coords: Optional[torch.Tensor],
        *,
        seq_len: int,
    ) -> Optional[torch.Tensor]:
        return build_attention_coords(
            coords,
            seq_len=seq_len,
            spatial_dim=self.output_spatial_dim,
            factorized=self.is_factorized,
            polar=self.polar,
        )

    def forward(
        self,
        seq_in: torch.LongTensor,
        *,
        coords: Optional[torch.Tensor] = None,
        box_size: Optional[torch.Tensor] = None,
        density: Optional[torch.Tensor] = None,
        pad_mask: Optional[torch.Tensor] = None,
        curve_waypoints: Optional[torch.Tensor] = None,
        input_deltas: Optional[torch.Tensor] = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, T = seq_in.shape
        coords = self._prepare_attention_coords(coords, seq_len=T)
        if self.continuous_input:
            # Token content comes from the CONTINUOUS previous displacement; position 0 keeps
            # the (SOS) token embedding so the start of sequence is still marked.
            if input_deltas is None:
                raise ValueError("continuous_input=True requires input_deltas in forward().")
            if input_deltas.shape[:2] != (B, T) or input_deltas.shape[-1] != self._delta_dim:
                raise ValueError(
                    f"input_deltas must be [B,T,{self._delta_dim}], got {tuple(input_deltas.shape)}"
                )
            assert self.delta_in_proj is not None
            cont = self.delta_in_proj(input_deltas.to(dtype=self.delta_in_proj.weight.dtype))
            sos_e = self.tok_emb(seq_in[:, :1])  # [B,1,d] reuse the SOS token embedding
            x = torch.cat([sos_e, cont[:, 1:, :]], dim=1)
            if self.use_pos_emb:
                assert self.pos_emb is not None
                if T > self.pos_emb.num_embeddings:
                    raise ValueError(f"Sequence length {T} exceeds max position embeddings {self.pos_emb.num_embeddings}")
                x = x + self.pos_emb(torch.arange(T, device=seq_in.device))[None, :, :]
        elif self.use_pos_emb:
            assert self.pos_emb is not None
            if T > self.pos_emb.num_embeddings:
                raise ValueError(f"Sequence length {T} exceeds max position embeddings {self.pos_emb.num_embeddings}")
            pos = torch.arange(T, device=seq_in.device)
            x = self.tok_emb(seq_in) + self.pos_emb(pos)[None, :, :]
        else:
            x = self.tok_emb(seq_in)

        if getattr(self, "is_factorized", False) and self.axis_emb is not None:
            assert self.axis_emb is not None
            axis_ids = torch.arange(T, device=seq_in.device) % self.output_spatial_dim
            x = x + self.axis_emb(axis_ids)[None, :, :]

        if self.use_density_cond:
            if density is None:
                raise ValueError("density must be provided when use_density_cond=True")
            assert self.density_emb is not None
            dens = torch.as_tensor(density, device=seq_in.device, dtype=x.dtype)
            if dens.ndim == 0:
                dens = dens.view(1, 1)
            elif dens.ndim == 1:
                dens = dens[:, None]
            elif dens.ndim == 2 and dens.shape[1] == 1:
                pass
            else:
                raise ValueError(f"density must be scalar, [B], or [B,1], got shape {tuple(dens.shape)}")
            if dens.shape[0] == 1 and B > 1:
                dens = dens.expand(B, 1)
            if dens.shape[0] != B:
                raise ValueError(f"density batch size mismatch: expected {B}, got {dens.shape[0]}")
            x = x + self.density_emb(dens)[:, None, :]

        if coords is None and self.id_coord_mode is not None:
            if self.cell_grid_size is None:
                raise ValueError("cell_grid_size must be set when id_coord_mode is enabled")
            coords = id_to_center_xyz(
                seq_in,
                grid_size=self.cell_grid_size,
                box_size=box_size,
                mapping=self.id_coord_mode,
                sos_id=self.sos_id,
            )

        if self.use_ida_pre and self.ida_pre is not None and coords is not None:
            if coords.shape[-1] != self.ida_pre.spatial_dim:
                raise ValueError(
                    f"coords last dim ({coords.shape[-1]}) must match ida_spatial_dim ({self.ida_pre.spatial_dim})"
                )
            causal_mask_bool = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool))
            x = self.ida_pre(
                x=x,
                coords=coords,
                causal_mask=causal_mask_bool,
                box_size=box_size,
                torus=self.torus,
            )

        if self.use_curve_rail and self.rail_attn is not None and curve_waypoints is not None:
            if curve_waypoints.shape[:2] != (B, T):
                raise ValueError(
                    f"curve_waypoints [B,T] {tuple(curve_waypoints.shape[:2])} must match seq_in {(B, T)}"
                )
            x = self.rail_attn(x, curve_waypoints.to(dtype=x.dtype))

        bias = self._causal_bias(B, T, coords, box_size, x.device, x.dtype)
        if coords is not None and (
            self.joint_arc_bias is not None
            or self.refine_rbf_bias is not None
            or self.knn_mask_k is not None
        ):
            causal = (
                torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool))
                .unsqueeze(0)
                .expand(B, -1, -1)
            )
            if self.refine_rbf_bias is not None:
                bias = bias + self.refine_rbf_bias(
                    coords, causal, box_size=box_size, torus=self.torus
                )
            if self.joint_arc_bias is not None:
                if input_deltas is None:
                    raise ValueError("use_joint_arc_bias requires input_deltas in forward().")
                arc_s = torch.cumsum(input_deltas[..., 0].to(dtype=x.dtype), dim=1)  # [B,T]
                bias = bias + self.joint_arc_bias(
                    coords, arc_s, causal, box_size=box_size, torus=self.torus
                )
            if self.knn_mask_k is not None:
                knn = _knn_causal_mask(
                    coords, self.knn_mask_k, causal, box_size=box_size, torus=self.torus
                )
                bias = bias.masked_fill(~knn[:, None, :, :], float("-inf"))
        for blk in self.blocks:
            x = x + blk["attn"](blk["ln1"](x), bias, key_padding_mask=pad_mask)
            x = x + blk["mlp"](blk["ln2"](x))
        x = self.ln_f(x)
        return self.head(x)

    def new_generation_cache(self) -> GenerationCache:
        return GenerationCache(len(self.blocks))

    def forward_step(
        self,
        seq_in_t: torch.LongTensor,
        *,
        cache: GenerationCache,
        coords: Optional[torch.Tensor] = None,
        box_size: Optional[torch.Tensor] = None,
        density: Optional[torch.Tensor] = None,
        curve_waypoints: Optional[torch.Tensor] = None,
        input_deltas: Optional[torch.Tensor] = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Incremental forward for ONE new position, using (and updating) a KV cache.

        Numerically equivalent to calling :meth:`forward` on the full prefix and taking
        the last position (see tests/test_kv_cache.py), at O(t) per step instead of
        O(t²): past K/V are exact because there is no RoPE and all input embeddings are
        position-local; EdgeBias contributes only its last causal row.

        Args mirror :meth:`forward` but per step: ``seq_in_t`` is [B, 1] (the token at
        position t = cache.length), ``input_deltas`` is [B, 1, delta_dim],
        ``curve_waypoints`` is [B, 1, K, 3]; ``coords`` covers the FULL prefix
        [B, t+1, D] (needed for the bias row — only the last row is computed).
        """
        if self.use_ida_pre and self.ida_pre is not None:
            raise NotImplementedError("forward_step does not support use_ida_pre; use forward().")
        B, T_new = seq_in_t.shape
        if T_new != 1:
            raise ValueError(f"forward_step expects one position at a time, got T={T_new}")
        t = int(cache.length)

        if self.continuous_input:
            if t == 0:
                x = self.tok_emb(seq_in_t)  # SOS marks the start of sequence
            else:
                if input_deltas is None:
                    raise ValueError("continuous_input=True requires input_deltas in forward_step().")
                if input_deltas.shape != (B, 1, self._delta_dim):
                    raise ValueError(
                        f"input_deltas must be [B,1,{self._delta_dim}], got {tuple(input_deltas.shape)}"
                    )
                assert self.delta_in_proj is not None
                x = self.delta_in_proj(input_deltas.to(dtype=self.delta_in_proj.weight.dtype))
        else:
            x = self.tok_emb(seq_in_t)
        if self.use_pos_emb:
            assert self.pos_emb is not None
            if t >= self.pos_emb.num_embeddings:
                raise ValueError(f"Position {t} exceeds max position embeddings {self.pos_emb.num_embeddings}")
            x = x + self.pos_emb(torch.tensor([t], device=seq_in_t.device))[None, :, :]
        if getattr(self, "is_factorized", False) and self.axis_emb is not None:
            axis_ids = torch.tensor([t % self.output_spatial_dim], device=seq_in_t.device)
            x = x + self.axis_emb(axis_ids)[None, :, :]
        if self.use_density_cond:
            if density is None:
                raise ValueError("density must be provided when use_density_cond=True")
            assert self.density_emb is not None
            dens = torch.as_tensor(density, device=seq_in_t.device, dtype=x.dtype).reshape(-1, 1)
            if dens.shape[0] == 1 and B > 1:
                dens = dens.expand(B, 1)
            x = x + self.density_emb(dens)[:, None, :]

        if self.use_curve_rail and self.rail_attn is not None and curve_waypoints is not None:
            x = self.rail_attn(x, curve_waypoints.to(dtype=x.dtype))

        coords_prep = self._prepare_attention_coords(coords, seq_len=t + 1)
        if self.use_edge_bias and coords_prep is not None:
            bias = self.edge_bias.bias_row(coords_prep, box_size=box_size, torus=self.torus)
        else:
            bias = torch.zeros(B, 1, 1, t + 1, device=x.device, dtype=x.dtype)

        if coords_prep is not None and (
            self.joint_arc_bias is not None
            or self.refine_rbf_bias is not None
            or self.knn_mask_k is not None
        ):
            if self.refine_rbf_bias is not None:
                bias = bias + self.refine_rbf_bias.bias_row(
                    coords_prep, box_size=box_size, torus=self.torus
                )
            if self.joint_arc_bias is not None:
                # Accumulate the cumulative arc coordinate across steps in the cache
                # (mirrors forward()'s cumsum over input_deltas[..., 0]).
                if input_deltas is not None:
                    delta_s_new = input_deltas[:, 0, 0].to(dtype=x.dtype)
                else:
                    delta_s_new = torch.zeros(B, device=x.device, dtype=x.dtype)
                prev_s = (
                    cache.arc_s[:, -1]
                    if cache.arc_s is not None
                    else torch.zeros(B, device=x.device, dtype=x.dtype)
                )
                s_new = (prev_s + delta_s_new).unsqueeze(1)  # [B,1]
                arc_prefix = s_new if cache.arc_s is None else torch.cat([cache.arc_s, s_new], dim=1)
                cache.arc_s = arc_prefix
                bias = bias + self.joint_arc_bias.bias_row(
                    coords_prep, arc_prefix, box_size=box_size, torus=self.torus
                )
            if self.knn_mask_k is not None:
                diff = coords_prep[:, -1:, :] - coords_prep  # [B,P,C]
                if self.torus and box_size is not None:
                    diff = wrap_min_image(diff, box_size)
                dist = torch.linalg.norm(diff, dim=-1)  # [B,P]
                kk = min(self.knn_mask_k + 1, dist.shape[1])
                idx = dist.topk(kk, dim=-1, largest=False).indices
                keep = torch.zeros_like(dist, dtype=torch.bool)
                keep.scatter_(-1, idx, True)
                keep[:, -1] = True  # self
                bias = bias.masked_fill(~keep[:, None, None, :], float("-inf"))

        for i, blk in enumerate(self.blocks):
            y, kv = blk["attn"](blk["ln1"](x), bias, past_kv=cache.layers[i], return_kv=True)
            cache.layers[i] = kv
            x = x + y
            x = x + blk["mlp"](blk["ln2"](x))
        cache.length = t + 1
        x = self.ln_f(x)
        return self.head(x)

    @torch.no_grad()
    def nll(
        self,
        seq: torch.LongTensor,
        *,
        coords: Optional[torch.Tensor] = None,
        box_size: Optional[torch.Tensor] = None,
        density: Optional[torch.Tensor] = None,
        pad_mask: Optional[torch.Tensor] = None,
        seq_in: Optional[torch.LongTensor] = None,
        logits: Optional[torch.Tensor] = None,
        curve_waypoints: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.use_continuous_head:
            raise NotImplementedError("nll() is only implemented for the discrete classification head.")
        if seq_in is None:
            seq_in = self._shift_with_sos(seq)
        if logits is None:
            logits = self.forward(
                seq_in,
                coords=coords,
                box_size=box_size,
                density=density,
                pad_mask=pad_mask,
                curve_waypoints=curve_waypoints,
            )
        logp = F.log_softmax(logits, dim=-1)
        tok_logp = logp.gather(-1, seq.unsqueeze(-1)).squeeze(-1)
        if pad_mask is not None:
            tok_logp = tok_logp.masked_fill(pad_mask, 0.0)
        return -tok_logp.sum(dim=1)

    def training_step(self, batch, idx):
        seq = batch.get("target_idx", batch.get("seq"))
        if seq is None:
            raise KeyError("Batch must include `target_idx` or `seq` for AR training")
        seq = seq.long()

        seq_in = batch.get("input_idx")
        if seq_in is None:
            seq_in = self._shift_with_sos(seq)
        else:
            seq_in = seq_in.long()

        coords = batch.get("abs_coords")
        if coords is None:
            coords = batch.get("absolute_coords")
        if coords is None:
            coords = batch.get("token_coords")
        if coords is not None:
            coords = coords.to(self.device, dtype=torch.float32)

        box_size = batch.get("box_size")
        if box_size is not None:
            box_size = box_size.to(self.device)

        density = batch.get("density")
        if density is not None:
            density = density.to(self.device)

        pad = batch.get("pad_mask")
        if pad is not None:
            pad = pad.bool()

        curve_waypoints = batch.get("curve_waypoints")
        if curve_waypoints is not None:
            curve_waypoints = curve_waypoints.to(self.device, dtype=torch.float32)

        coords = self._prepare_attention_coords(coords, seq_len=seq_in.shape[1])
        coords = self._apply_training_coord_dequantization(coords, box_size)
        input_deltas = None
        if self.continuous_input:
            _delta_key = "arc_delta" if self.arc_repr else "deltas"
            tgt_deltas = batch.get(_delta_key)
            if tgt_deltas is None:
                raise KeyError(f"continuous_input=True requires batch[{_delta_key!r}].")
            tgt_deltas = tgt_deltas.to(self.device, dtype=torch.float32)
            # Shift so position 0 = SOS (zeros), position t = delta_{t-1} (continuous feedback).
            zero = torch.zeros_like(tgt_deltas[:, :1, :])
            input_deltas = torch.cat([zero, tgt_deltas[:, :-1, :]], dim=1)
        outputs = self.forward(
            seq_in,
            coords=coords,
            box_size=box_size,
            density=density,
            pad_mask=pad,
            curve_waypoints=curve_waypoints,
            input_deltas=input_deltas,
        )
        if self.discrete:
            logits = outputs
            tok_nll = F.cross_entropy(
                logits.reshape(-1, self.K),
                seq.reshape(-1),
                reduction="none",
            ).view_as(seq).float()
            if pad is not None:
                tok_nll = tok_nll.masked_fill(pad, 0.0)
                coord_count = (~pad).sum(dim=1).to(tok_nll.dtype).clamp_min(1.0)
            else:
                coord_count = torch.full(
                    (seq.size(0),),
                    fill_value=seq.size(1),
                    device=seq.device,
                    dtype=tok_nll.dtype,
                )
            seq_nll_model = tok_nll.sum(dim=1)
            loss = (seq_nll_model / coord_count).mean()
            seq_nll_exact = seq_nll_model.detach()
        elif self.use_continuous_head:
            _delta_key = "arc_delta" if self.arc_repr else "deltas"
            deltas = batch.get(_delta_key)
            if deltas is None:
                raise KeyError(f"Continuous head requires batch[{_delta_key!r}] targets.")
            deltas = deltas.to(self.device, dtype=torch.float32)
            log_pi, mu, scale_param = outputs
            # arc_repr: box_size is 3D xyz, but targets are 4D (Δs, fine_xyz) — no torus wrap
            box_size_loss = (box_size if self.torus else None) if not self.arc_repr else None
            if self.is_factorized:
                batch_size, n_predict, coord_dim = deltas.shape
                deltas = deltas.reshape(batch_size, n_predict * coord_dim, 1)
                if self.torus and box_size is not None:
                    box_size_loss = (
                        box_size.unsqueeze(1)
                        .expand(-1, n_predict, -1)
                        .reshape(batch_size, n_predict * coord_dim, 1)
                    )
            if deltas.shape[:2] != log_pi.shape[:2] or deltas.shape[-1] != mu.shape[-1]:
                raise ValueError(
                    f"Continuous delta targets shape {tuple(deltas.shape)} does not match "
                    f"model outputs log_pi={tuple(log_pi.shape)}, mu={tuple(mu.shape)}, scale={tuple(scale_param.shape)}."
                )
            loss, seq_nll_model, coord_count, point_nll = mdn_loss(
                log_pi,
                mu,
                scale_param,
                deltas,
                pad_mask=pad,
                box_size=box_size_loss,
                return_point_nll=True,
            )
            seq_nll_exact = seq_nll_model.detach()
            if self.arc_repr:
                # Jump-stratified NLL: the cheap discriminator for the geometric
                # probe variants (action-plan Phase 3). Δs is the arc target's dim 0.
                strat = stratified_nll_means(
                    point_nll.detach(),
                    deltas[..., 0].detach(),
                    threshold=self.jump_delta_s_threshold,
                    pad_mask=pad,
                )
                if strat["nll_jump"] is not None:
                    self.log(
                        "train/nll_jump",
                        strat["nll_jump"],
                        on_step=False,
                        on_epoch=True,
                        batch_size=seq.size(0),
                    )
                if strat["nll_local"] is not None:
                    self.log(
                        "train/nll_local",
                        strat["nll_local"],
                        on_step=False,
                        on_epoch=True,
                        batch_size=seq.size(0),
                    )
                self.log(
                    "train/jump_fraction",
                    strat["jump_fraction"],
                    on_step=False,
                    on_epoch=True,
                    batch_size=seq.size(0),
                )
        else:
            logits = outputs
            tok_nll = F.cross_entropy(
                logits.reshape(-1, self.K),
                seq.reshape(-1),
                reduction="none",
            ).view_as(seq).float()  # [B,T], differentiable
            if pad is not None:
                tok_nll = tok_nll.masked_fill(pad, 0.0)
                coord_count = (~pad).sum(dim=1).to(tok_nll.dtype).clamp_min(1.0)
            else:
                coord_count = torch.full(
                    (seq.size(0),),
                    fill_value=seq.size(1),
                    device=seq.device,
                    dtype=tok_nll.dtype,
                )
            seq_nll_model = tok_nll.sum(dim=1)
            loss = (seq_nll_model / coord_count).mean()
            seq_nll_exact = seq_nll_model.detach()

        if self.lambda_var > 0.0:
            target_energy = batch.get("target_energy")
            if target_energy is None:
                raise KeyError(
                    "lambda_var > 0 requires batch['target_energy']. "
                    "Regenerate the lj_transferable cache with RUN_PREPROCESS=1."
                )
            target_energy = target_energy.to(self.device, dtype=torch.float32)
            var_log_w, target_log_p, _, ess_fraction = compute_log_weight_variance(
                seq_nll_model,
                target_energy,
                kT=self.lj_kT,
            )
            loss = loss + (self.lambda_var * var_log_w)
            self.log(
                "train/energy_var_loss",
                var_log_w.detach(),
                on_step=True,
                on_epoch=True,
                batch_size=target_energy.size(0),
            )
            self.log(
                "train/target_log_p",
                target_log_p.mean().detach(),
                on_step=True,
                on_epoch=True,
                batch_size=target_energy.size(0),
            )
            self.log(
                "train/ess_fraction",
                ess_fraction.detach(),
                on_step=True,
                on_epoch=True,
                batch_size=target_energy.size(0),
            )
        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=True, batch_size=seq.size(0))
        self.log("train/nll", seq_nll_exact.mean(), prog_bar=True, on_step=True, on_epoch=True, batch_size=seq.size(0))
        self.log(
            "train/bpd",
            (seq_nll_exact / (coord_count * torch.log(torch.tensor(2.0, device=seq_nll_exact.device)))).mean(),
            on_step=True,
            on_epoch=True,
            batch_size=seq.size(0),
        )
        return loss

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
