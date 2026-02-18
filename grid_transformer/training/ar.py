from __future__ import annotations

from functools import lru_cache
from typing import Optional

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F


def wrap_min_image(dr: torch.Tensor, box_size: torch.Tensor) -> torch.Tensor:
    """Apply minimum-image wrapping along each coordinate axis."""
    box_size = torch.as_tensor(box_size, device=dr.device, dtype=dr.dtype)
    if box_size.ndim == 1:
        box_size = box_size.view(*([1] * (dr.ndim - 1)), box_size.shape[0])
    elif box_size.ndim == 2:
        box_size = box_size.view(box_size.shape[0], *([1] * (dr.ndim - 2)), box_size.shape[1])
    else:
        raise ValueError(f"box_size must be rank 1 or 2, got rank {box_size.ndim}")
    return dr - box_size * torch.round(dr / box_size.clamp_min(1e-8))


def _is_power_of_two(v: int) -> bool:
    return v > 0 and (v & (v - 1)) == 0


def _hilbert_d2xy(order_n: int, d: int) -> tuple[int, int]:
    """Distance-to-2D Hilbert coordinate for square side length `order_n`."""
    x = 0
    y = 0
    t = int(d)
    s = 1
    while s < order_n:
        rx = (t // 2) & 1
        ry = (t ^ rx) & 1
        if ry == 0:
            if rx == 1:
                x = s - 1 - x
                y = s - 1 - y
            x, y = y, x
        x += s * rx
        y += s * ry
        t //= 4
        s <<= 1
    return x, y


@lru_cache(maxsize=16)
def _hilbert_id_to_xy_lut(side: int) -> torch.Tensor:
    if not _is_power_of_two(side):
        raise ValueError(f"Hilbert mapping requires power-of-two side length, got {side}")
    lut = torch.empty((side * side, 2), dtype=torch.float32)
    for d in range(side * side):
        x, y = _hilbert_d2xy(side, d)
        lut[d, 0] = float(x)
        lut[d, 1] = float(y)
    return lut


def id_to_center_xyz(
    token_ids: torch.LongTensor,
    *,
    grid_size: int,
    box_size: Optional[torch.Tensor] = None,
    mapping: str = "hilbert",
    sos_id: Optional[int] = None,
) -> torch.Tensor:
    """
    Convert discrete token IDs to physical 2D cell-center coordinates.

    token_ids: [B, T], values in [0, K-1] (optionally includes SOS token).
    returns:   [B, T, 2] where coords are in the same physical units as box_size.
    """
    if token_ids.ndim != 2:
        raise ValueError(f"token_ids must be [B,T], got {tuple(token_ids.shape)}")
    if grid_size <= 0:
        raise ValueError(f"grid_size must be positive, got {grid_size}")

    K = int(grid_size * grid_size)
    ids = token_ids.long()
    invalid = (ids < 0) | (ids >= K)
    if sos_id is not None:
        invalid = invalid | (ids == int(sos_id))
    ids_safe = ids.clamp(min=0, max=K - 1)

    if mapping == "hilbert":
        lut = _hilbert_id_to_xy_lut(int(grid_size)).to(device=ids.device)
        xy = lut[ids_safe]  # [B,T,2], columns=(ix, iy)
    elif mapping == "raster":
        ix = (ids_safe % int(grid_size)).to(dtype=torch.float32)
        iy = torch.div(ids_safe, int(grid_size), rounding_mode="floor").to(dtype=torch.float32)
        xy = torch.stack([ix, iy], dim=-1)
    else:
        raise ValueError(f"Unsupported id->coord mapping '{mapping}'")

    if box_size is None:
        bs = torch.tensor([float(grid_size), float(grid_size)], device=ids.device, dtype=xy.dtype).view(1, 1, 2)
    else:
        bs = torch.as_tensor(box_size, device=ids.device, dtype=xy.dtype)
        if bs.ndim == 1:
            if bs.numel() != 2:
                raise ValueError(f"box_size must have 2 entries, got shape {tuple(bs.shape)}")
            bs = bs.view(1, 1, 2)
        elif bs.ndim == 2:
            if bs.shape[1] != 2:
                raise ValueError(f"box_size must be [B,2], got shape {tuple(bs.shape)}")
            bs = bs[:, None, :]
        else:
            raise ValueError(f"box_size must be rank 1 or 2, got rank {bs.ndim}")

    centers = (xy + 0.5) * (bs / float(grid_size))
    centers = centers.masked_fill(invalid.unsqueeze(-1), 0.0)
    return centers


class EdgeBias(nn.Module):
    """Turn pairwise distances into additive attention bias."""

    def __init__(self, n_head: int, n_bins: int = 32, d_dir: int = 16, use_dir: bool = True):
        super().__init__()
        self.n_head = n_head
        self.n_bins = n_bins
        self.use_dir = use_dir
        self.bin_embed = nn.Embedding(n_bins, n_head)
        if use_dir:
            self.dir_mlp = nn.Sequential(
                nn.Linear(3, d_dir),
                nn.GELU(),
                nn.Linear(d_dir, n_head),
            )

    @staticmethod
    def _bin_dist(d: torch.Tensor, n_bins: int) -> torch.LongTensor:
        d = d.clamp(min=0.0, max=1.0)
        return (d * (n_bins - 1)).long().clamp(0, n_bins - 1)

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

        if torus:
            bs = torch.as_tensor(box_size, device=coords3.device, dtype=coords3.dtype)
            if bs.ndim == 1:
                scale = torch.linalg.norm(bs).clamp_min(1e-6)
                dnorm = dist / scale
            else:
                scale = torch.linalg.norm(bs, dim=-1).clamp_min(1e-6)
                dnorm = dist / scale[:, None, None]
        else:
            scale = dist.detach().amax(dim=(1, 2), keepdim=True).clamp_min(1e-6)
            dnorm = dist / scale

        bins = self._bin_dist(dnorm, self.n_bins)  # [B,T,T]
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
    def __init__(self, d_model: int, n_head: int, dropout: float):
        super().__init__()
        if d_model % n_head != 0:
            raise ValueError(f"d_model={d_model} must be divisible by n_head={n_head}")
        self.n_head = n_head
        self.d_head = d_model // n_head
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
        torus: bool = False,
        use_dir_bias: bool = True,
        dist_bins: int = 32,
        id_coord_mode: Optional[str] = None,
        cell_grid_size: Optional[int] = None,
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
        self.pos_emb = nn.Embedding(4096, d_model)

        self.use_edge_bias = bool(use_edge_bias)
        self.torus = bool(torus)
        self.edge_bias = EdgeBias(n_head=n_head, n_bins=dist_bins, use_dir=use_dir_bias)
        self.id_coord_mode = id_coord_mode
        self.cell_grid_size = int(cell_grid_size) if cell_grid_size is not None else None

        self.blocks = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        "ln1": nn.LayerNorm(d_model),
                        "attn": CausalSelfAttnWithBias(d_model, n_head, dropout),
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
        pad_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, T = seq_in.shape
        if T > self.pos_emb.num_embeddings:
            raise ValueError(f"Sequence length {T} exceeds max position embeddings {self.pos_emb.num_embeddings}")

        pos = torch.arange(T, device=seq_in.device)
        x = self.tok_emb(seq_in) + self.pos_emb(pos)[None, :, :]

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
        pad_mask: Optional[torch.Tensor] = None,
        seq_in: Optional[torch.LongTensor] = None,
        logits: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if seq_in is None:
            seq_in = self._shift_with_sos(seq)
        if logits is None:
            logits = self.forward(seq_in, coords=coords, box_size=box_size, pad_mask=pad_mask)
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

        pad = batch.get("pad_mask")
        if pad is not None:
            pad = pad.bool()

        logits = self.forward(seq_in, coords=coords, box_size=box_size, pad_mask=pad)
        tok_nll = F.cross_entropy(
            logits.reshape(-1, self.K),
            seq.reshape(-1),
            reduction="none",
        ).view_as(seq).float()  # [B,T], differentiable
        if pad is not None:
            tok_nll = tok_nll.masked_fill(pad, 0.0)
            tok_count = (~pad).sum(dim=1).to(tok_nll.dtype).clamp_min(1.0)
        else:
            tok_count = torch.full(
                (seq.size(0),),
                fill_value=seq.size(1),
                device=seq.device,
                dtype=tok_nll.dtype,
            )
        seq_nll_train = tok_nll.sum(dim=1)
        loss = (seq_nll_train / tok_count).mean()
        seq_nll_exact = seq_nll_train.detach()
        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=True, batch_size=seq.size(0))
        self.log("train/nll", seq_nll_exact.mean(), prog_bar=True, on_step=True, on_epoch=True, batch_size=seq.size(0))
        self.log(
            "train/bpd",
            (seq_nll_exact / (tok_count * torch.log(torch.tensor(2.0, device=seq_nll_exact.device)))).mean(),
            on_step=True,
            on_epoch=True,
            batch_size=seq.size(0),
        )
        return loss

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
