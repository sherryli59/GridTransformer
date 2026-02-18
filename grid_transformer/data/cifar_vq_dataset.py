from __future__ import annotations

import os
import tempfile
from functools import lru_cache
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


def _resolve_torch_dtype(name: str | None) -> Optional[torch.dtype]:
    if not name or name.lower() == "auto":
        return None
    mapping: Dict[str, torch.dtype] = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    key = name.lower()
    if key not in mapping:
        raise ValueError(f"Unsupported torch dtype '{name}'.")
    return mapping[key]


def _next_pow2(v: int) -> int:
    n = 1
    while n < v:
        n <<= 1
    return n


def _hilbert_d2xy(order_n: int, d: int) -> tuple[int, int]:
    """Distance-to-2D Hilbert coordinate for square side length `order_n` (power of two)."""
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


@lru_cache(maxsize=32)
def get_hilbert_indices(h: int, w: int) -> torch.LongTensor:
    """
    Return `[h*w, 2]` `(y, x)` coordinates traversed in Hilbert order.
    Supports rectangular grids by generating on the next power-of-two square
    and filtering points outside the target extent.
    """
    if h <= 0 or w <= 0:
        raise ValueError(f"Grid shape must be positive, got {(h, w)}")
    side = _next_pow2(max(h, w))
    coords: list[tuple[int, int]] = []
    total = side * side
    for d in range(total):
        x, y = _hilbert_d2xy(side, d)
        if y < h and x < w:
            coords.append((y, x))
            if len(coords) == h * w:
                break
    if len(coords) != h * w:
        raise RuntimeError(f"Failed to generate Hilbert ordering for {(h, w)}")
    return torch.tensor(coords, dtype=torch.long)

class CIFARVQLatentDataset(Dataset):
    """
    CIFAR-10 -> autoregressive VQ-token sequences.

    Returns:
      input_idx:    [T] shifted-right sequence with SOS prepended.
      target_idx:   [T] token IDs in generation order.
      seq:          [T] alias of target_idx (for AR modules expecting `seq`).
      token_coords: [T, 2] (y, x) sequence coordinates.
      box_size:     [2] torus side lengths for optional minimum-image distances.
    """

    def __init__(
        self,
        data_root: str,
        vq_model: str,
        train: bool = True,
        limit: Optional[int] = None,
        mask_ratio: float = 0.9,  # retained for backward CLI compatibility (unused in AR mode)
        mask_ratio_max: Optional[float] = None,  # retained for backward compatibility
        seed: int = 0,
        vq_subfolder: Optional[str] = None,
        vq_dtype: Optional[str] = None,
        vq_device: str = "cuda",
        cache_latents: bool = False,
        tmp_dir: Optional[str] = None,
        use_hilbert: bool = True,
    ) -> None:
        try:
            from torchvision import datasets, transforms  # type: ignore
        except ImportError as exc:
            raise ImportError("torchvision is required to load the CIFAR-10 dataset.") from exc

        self.rng = np.random.default_rng(seed)
        _ = (mask_ratio, mask_ratio_max)
        self.vq_model_id = vq_model
        self.vq_subfolder = vq_subfolder
        self.vq_dtype = _resolve_torch_dtype(vq_dtype)
        self.device = torch.device(vq_device)
        self.cache_latents = cache_latents
        self.use_hilbert = bool(use_hilbert)
        self._latent_cache: Dict[int, Dict[str, torch.Tensor]] = {}
        self._token_order: Optional[torch.LongTensor] = None

        transform = transforms.ToTensor()  # [0,1], [3,32,32]
        self._dataset = datasets.CIFAR10(root=data_root, train=train, download=True, transform=transform)

        total = len(self._dataset)
        if limit is not None:
            n = max(1, min(int(limit), total))
            self._indices = list(range(n))
        else:
            self._indices = list(range(total))

        self._vq_model: Optional["VQModel"] = None
        self._latent_shape: Optional[Tuple[int, int, int]] = None  # (D,Hc,Wc)
        self.latent_downsample: Optional[float] = None
        self._codebook_size: Optional[int] = None
        self._codebook_weight: Optional[torch.Tensor] = None  # [K, D], on self.device
        self._tmp_dir = tmp_dir

    # ---- internals ---------------------------------------------------------------

    def _ensure_tmpdir(self) -> str:
        path = self._tmp_dir or os.path.join(self._dataset.root, "_vq_tmp")
        os.makedirs(path, exist_ok=True)
        for env_name in ("TMPDIR", "TEMP", "TMP"):
            os.environ.setdefault(env_name, path)
        tempfile.tempdir = path
        return path

    def _lazy_vq_model(self) -> "VQModel":
        if self._vq_model is None:
            self._ensure_tmpdir()
            from diffusers import VQModel  # lazy import

            model = VQModel.from_pretrained(
                self.vq_model_id,
                subfolder=self.vq_subfolder,
                torch_dtype=self.vq_dtype,
            ).to(self.device)
            model.eval().requires_grad_(False)
            self._vq_model = model

            # Probe latent shape with zeros
            with torch.no_grad():
                dummy = torch.zeros(1, 3, 32, 32, device=self.device)
                z_q = model.encode(dummy).latents  # already quantized in diffusers VQModel
            # shapes
            D, Hc, Wc = z_q.shape[1], z_q.shape[2], z_q.shape[3]
            self._latent_shape = (D, Hc, Wc)
            self.latent_downsample = 32.0 / float(Wc)

            # codebook
            E = model.quantize.embedding.weight  # [K, D]
            self._codebook_weight = E.detach().to(self.device)
            self._codebook_size = int(E.shape[0])
        return self._vq_model  # type: ignore[return-value]

    @property
    def latent_shape(self) -> Tuple[int, int, int]:
        if self._latent_shape is None:
            self._lazy_vq_model()
        assert self._latent_shape is not None
        return self._latent_shape

    @property
    def input_channels(self) -> int:
        D, _, _ = self.latent_shape
        return D

    @property
    def codebook_size(self) -> int:
        if self._codebook_size is None:
            self._lazy_vq_model()
        assert self._codebook_size is not None
        return self._codebook_size

    # ---- helpers -----------------------------------------------------------------

    @torch.no_grad()
    def _encode_to_indices_and_embeddings(self, img: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        img: [3,32,32] in [0,1]  (CPU)
        returns:
        indices: [Hc, Wc] (Long, CPU)
        emb:     [D, Hc, Wc] (Float32, CPU)  == z_q.squeeze(0)
        """
        model = self._lazy_vq_model()                 # moves VQ to self.device
        x = (img.unsqueeze(0) * 2.0 - 1.0).to(self.device)  # [1,3,32,32]

        # explicit quantize path to keep idx and emb consistent
        z_e = model.quant_conv(model.encoder(x))      # [1,D,Hc,Wc] pre-quant
        z_q, _, info = model.quantize(z_e)            # z_q quantized, info[2] = indices
        idx = info[2].squeeze(0).to("cpu", torch.long)   # [Hc, Wc]
        emb = z_q.squeeze(0).to("cpu", torch.float32)    # [D, Hc, Wc]
        idx = idx.view(z_q.shape[2], z_q.shape[3])  # [Hc, Wc]
        return idx, emb

    def _get_token_order(self, h: int, w: int) -> torch.LongTensor:
        if self._token_order is not None and self._token_order.shape[0] == h * w:
            return self._token_order
        if self.use_hilbert:
            order = get_hilbert_indices(h, w)
        else:
            yy, xx = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
            order = torch.stack([yy.reshape(-1), xx.reshape(-1)], dim=1).long()
        self._token_order = order
        return order

    # ---- dataset protocol --------------------------------------------------------

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
        real_idx = self._indices[i]
        img, label = self._dataset[real_idx]  # img: [3,32,32] float in [0,1]

        # 1) Get code indices (and optionally emb for Hc,Wc probing only)
        if self.cache_latents and real_idx in self._latent_cache:
            cache = self._latent_cache[real_idx]
            indices = cache["indices"]      # [Hc,Wc] Long (CPU)
            # emb may or may not be present; don't rely on it anymore
            emb = cache.get("emb", None)    # [D,Hc,Wc] Float32 (CPU) or None
        else:
            indices, emb = self._encode_to_indices_and_embeddings(img)  # indices: [Hc,Wc]
            if self.cache_latents:
                self._latent_cache[real_idx] = {"indices": indices, "emb": emb}

        # 2) Shapes
        if emb is not None:
            _, Hc, Wc = emb.shape
        else:
            # fall back when emb is not cached
            Hc, Wc = indices.shape

        # 3) Flatten to an AR sequence using optional Hilbert ordering
        order = self._get_token_order(Hc, Wc)  # [T,2] as (y,x)
        ys = order[:, 0]
        xs = order[:, 1]
        target_idx = indices[ys, xs].long()  # [T]

        # 4) Shift right and prepend SOS token (ID=K)
        K = self.codebook_size
        sos_id = K
        input_idx = torch.empty_like(target_idx)
        input_idx[0] = sos_id
        if target_idx.numel() > 1:
            input_idx[1:] = target_idx[:-1]

        # 5) Pack AR sample
        sample = {
            "input_idx": input_idx,                 # [T], values in [0..K] (K is SOS)
            "target_idx": target_idx,               # [T], values in [0..K-1]
            "seq": target_idx,                      # alias for AR APIs
            "token_coords": order.float(),          # [T,2] in latent-grid coordinates
            "box_size": torch.tensor([float(Hc), float(Wc)], dtype=torch.float32),
            "meta": torch.tensor(
                [float(Hc), float(Wc), float(self.latent_downsample or (32.0 / Wc))],
                dtype=torch.float32,
            ),
            "label": torch.tensor(int(label)),
            "K": torch.tensor(K, dtype=torch.long),
            "sos_id": torch.tensor(sos_id, dtype=torch.long),
        }
        return sample
