from __future__ import annotations

import math
from typing import Optional

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..training.ar import MDNHead, compute_log_weight_variance, mdn_loss
from .deep_ida import DeepIDABias
from .ida import PeriodicIDA
from .rbf_edge_bias import RBFEdgeBias
from ..utils.spatial import id_to_center_xyz, wrap_min_image


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

        dist = torch.linalg.norm(diff, dim=-1)  # [B,T,T]

        bins = self._bin_dist(dist)  # [B,T,T]
        bias_dist = self.bin_embed(bins).permute(0, 3, 1, 2).contiguous()  # [B,nH,T,T]

        if self.use_dir:
            dir_scale = dist.unsqueeze(-1).clamp_min(1e-6)
            unit_dir = diff / dir_scale
            dir_bias = self.dir_mlp(unit_dir).permute(0, 3, 1, 2).contiguous()  # [B,nH,T,T]
            bias = bias_dist + dir_bias
        else:
            bias = bias_dist

        return bias.masked_fill(~causal_mask[:, None, :, :], float("-inf"))


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

    def forward(self, x: torch.Tensor, bias: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, T, D = x.shape
        qkv = self.qkv(x).view(B, T, 3, self.n_head, self.d_head)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)  # [B,nH,T,dH]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) / (self.d_head ** 0.5)
        att = att + bias

        if key_padding_mask is not None:
            att = att.masked_fill(key_padding_mask[:, None, None, :], float("-inf"))

        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, D)
        return self.resid_drop(self.proj(y))


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
        num_mixtures: int = 32,
        lambda_var: float = 0.0,
        lj_kT: float = 1.0,
        lj_epsilon: float = 1.0,
        lj_sigma: float = 1.0,
        lj_cutoff: Optional[float] = None,
        lj_spring_constant: float = 0.5,
        lj_boxlength: float = 10.0,
        lj_periodic: bool = False,
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
        self.axis_emb = nn.Embedding(3, d_model) if self.is_factorized else None

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
        self.use_continuous_head = bool(use_continuous_head)
        self.output_spatial_dim = int(ida_spatial_dim)
        self.num_mixtures = int(num_mixtures)
        self.lambda_var = float(lambda_var)
        self.lj_kT = float(lj_kT)
        if self.use_ida_pre:
            self.ida_pre = PeriodicIDA(
                d_model=d_model,
                num_heads=n_head,
                spatial_dim=int(ida_spatial_dim),
                bias=False,
            )
        else:
            self.ida_pre = None
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
            self.head = MDNHead(
                d_model=d_model,
                spatial_dim=self.output_spatial_dim,
                num_mixtures=self.num_mixtures,
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

    def forward(
        self,
        seq_in: torch.LongTensor,
        *,
        coords: Optional[torch.Tensor] = None,
        box_size: Optional[torch.Tensor] = None,
        density: Optional[torch.Tensor] = None,
        pad_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, T = seq_in.shape
        if self.use_pos_emb:
            assert self.pos_emb is not None
            if T > self.pos_emb.num_embeddings:
                raise ValueError(f"Sequence length {T} exceeds max position embeddings {self.pos_emb.num_embeddings}")
            pos = torch.arange(T, device=seq_in.device)
            x = self.tok_emb(seq_in) + self.pos_emb(pos)[None, :, :]
        else:
            x = self.tok_emb(seq_in)

        if self.is_factorized:
            assert self.axis_emb is not None
            axis_ids = torch.arange(T, device=seq_in.device) % 3
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

        bias = self._causal_bias(B, T, coords, box_size, x.device, x.dtype)
        for blk in self.blocks:
            x = x + blk["attn"](blk["ln1"](x), bias, key_padding_mask=pad_mask)
            x = x + blk["mlp"](blk["ln2"](x))
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
    ) -> torch.Tensor:
        if self.use_continuous_head:
            raise NotImplementedError("nll() is only implemented for the discrete classification head.")
        if seq_in is None:
            seq_in = self._shift_with_sos(seq)
        if logits is None:
            logits = self.forward(seq_in, coords=coords, box_size=box_size, density=density, pad_mask=pad_mask)
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

        coords = batch.get("token_coords")
        if coords is not None:
            coords = coords.to(self.device)

        box_size = batch.get("box_size")
        if box_size is not None:
            box_size = box_size.to(self.device)

        density = batch.get("density")
        if density is not None:
            density = density.to(self.device)

        pad = batch.get("pad_mask")
        if pad is not None:
            pad = pad.bool()

        coords = self._apply_training_coord_dequantization(coords, box_size)
        outputs = self.forward(seq_in, coords=coords, box_size=box_size, density=density, pad_mask=pad)
        if self.use_continuous_head:
            deltas = batch.get("deltas")
            if deltas is None:
                raise KeyError("Continuous head requires batch['deltas'] targets.")
            deltas = deltas.to(self.device, dtype=torch.float32)
            log_pi, mu, sigma = outputs
            if deltas.shape[:2] != log_pi.shape[:2]:
                raise ValueError(
                    f"Continuous delta targets shape {tuple(deltas.shape)} does not match "
                    f"model outputs {tuple(log_pi.shape)}. Continuous head currently requires "
                    "non-factorized lj_transferable inputs."
                )
            loss, seq_nll_model, coord_count = mdn_loss(
                log_pi,
                mu,
                sigma,
                deltas,
                pad_mask=pad,
            )
            seq_nll_exact = seq_nll_model.detach()
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
