from __future__ import annotations

from typing import Optional

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F


class VanillaTransformerAR(pl.LightningModule):
    """
    Notebook-style autoregressive transformer over discrete token sequences.

    This mirrors the mcmc_lj_13.ipynb architecture:
      - token + absolute position embeddings
      - causal nn.TransformerEncoder stack
      - optional group-wise permutation of tokens after SOS during training
    """

    def __init__(
        self,
        K: int,
        d_model: int = 256,
        n_layer: int = 6,
        n_head: int = 4,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        lr: float = 3e-4,
        weight_decay: float = 1e-2,
        pad_id: Optional[int] = None,
        sos_id: Optional[int] = None,
        input_vocab_size: Optional[int] = None,
        use_pos_emb: bool = True,
        permute_train: bool = False,
        permute_group_size: int = 1,
        max_seq_len: int = 4096,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.K = int(K)
        self.pad_id = pad_id
        self.sos_id = sos_id
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.use_pos_emb = bool(use_pos_emb)
        self.permute_train = bool(permute_train)
        self.permute_group_size = int(permute_group_size)
        if self.permute_group_size <= 0:
            raise ValueError(f"permute_group_size must be positive, got {self.permute_group_size}")
        self.max_seq_len = int(max_seq_len)
        if self.max_seq_len <= 0:
            raise ValueError(f"max_seq_len must be positive, got {self.max_seq_len}")

        in_vocab = int(input_vocab_size) if input_vocab_size is not None else int(K)
        self.tok_emb = nn.Embedding(in_vocab, d_model)
        self.pos_emb = nn.Embedding(self.max_seq_len, d_model) if self.use_pos_emb else None
        self.drop = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_head,
            dim_feedforward=int(mlp_ratio * d_model),
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layer)
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, K)

        self._init_weights()
        causal_mask = torch.triu(torch.ones(self.max_seq_len, self.max_seq_len), diagonal=1).bool()
        self.register_buffer("causal_mask", causal_mask, persistent=False)

    def _init_weights(self) -> None:
        nn.init.normal_(self.tok_emb.weight, std=1.0 / max(float(self.hparams.d_model), 1.0) ** 0.5)
        if self.pos_emb is not None:
            nn.init.normal_(self.pos_emb.weight, std=0.02)

    def _shift_with_sos(self, seq: torch.LongTensor) -> torch.LongTensor:
        if self.sos_id is None:
            return torch.cat([seq[:, :1], seq[:, :-1]], dim=1)
        sos = torch.full((seq.size(0), 1), int(self.sos_id), dtype=seq.dtype, device=seq.device)
        return torch.cat([sos, seq[:, :-1]], dim=1)

    def _permute_sequence_groups(self, sequence: torch.LongTensor) -> torch.LongTensor:
        tail = sequence[:, 1:]
        if tail.numel() == 0:
            return sequence
        if tail.size(1) % self.permute_group_size != 0:
            raise ValueError(
                "Cannot group-permute sequence tail of length "
                f"{tail.size(1)} with permute_group_size={self.permute_group_size}"
            )
        n_groups = tail.size(1) // self.permute_group_size
        grouped = tail.view(tail.size(0), n_groups, self.permute_group_size)
        perms = torch.argsort(torch.rand(grouped.size(0), n_groups, device=sequence.device), dim=-1)
        grouped = grouped.gather(1, perms.unsqueeze(-1).expand_as(grouped))
        return torch.cat([sequence[:, :1], grouped.reshape(tail.size(0), -1)], dim=1)

    def _sequence_from_batch(self, batch: dict[str, torch.Tensor]) -> torch.LongTensor:
        sequence = batch.get("sequence")
        if sequence is not None:
            return sequence.long()

        target = batch.get("target_idx", batch.get("seq"))
        if target is None:
            raise KeyError("Batch must include `sequence`, `target_idx`, or `seq` for AR training")
        target = target.long()

        seq_in = batch.get("input_idx")
        if seq_in is None:
            return torch.cat([self._shift_with_sos(target)[:, :1], target], dim=1)
        return torch.cat([seq_in.long()[:, :1], target], dim=1)

    def forward(
        self,
        seq_in: torch.LongTensor,
        *,
        coords: Optional[torch.Tensor] = None,
        box_size: Optional[torch.Tensor] = None,
        density: Optional[torch.Tensor] = None,
        pad_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        del coords
        del box_size
        del density

        _, T = seq_in.shape
        if T > self.max_seq_len:
            raise ValueError(f"Sequence length {T} exceeds max_seq_len={self.max_seq_len}")

        x = self.tok_emb(seq_in)
        if self.pos_emb is not None:
            pos = torch.arange(T, device=seq_in.device)
            x = x + self.pos_emb(pos)[None, :, :]
        x = self.drop(x)
        x = self.transformer(
            x,
            mask=self.causal_mask[:T, :T],
            src_key_padding_mask=pad_mask,
            is_causal=True,
        )
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
        del idx
        sequence = self._sequence_from_batch(batch).to(self.device)
        if self.permute_train:
            sequence = self._permute_sequence_groups(sequence)

        seq_in = sequence[:, :-1]
        seq = sequence[:, 1:]

        pad = batch.get("pad_mask")
        if pad is not None:
            pad = pad.bool().to(self.device)

        logits = self.forward(seq_in, pad_mask=pad)
        tok_nll = F.cross_entropy(
            logits.reshape(-1, self.K),
            seq.reshape(-1),
            reduction="none",
        ).view_as(seq).float()

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

        seq_nll = tok_nll.sum(dim=1)
        loss = (seq_nll / tok_count).mean()
        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=True, batch_size=seq.size(0))
        self.log("train/nll", seq_nll.detach().mean(), prog_bar=True, on_step=True, on_epoch=True, batch_size=seq.size(0))
        self.log(
            "train/bpd",
            (seq_nll.detach() / (tok_count * torch.log(torch.tensor(2.0, device=seq.device)))).mean(),
            on_step=True,
            on_epoch=True,
            batch_size=seq.size(0),
        )
        return loss

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
