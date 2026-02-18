import argparse
import os
import tempfile
from typing import Optional, Tuple

import numpy as np
import torch
from PIL import Image

from grid_transformer.data.cifar_vq_dataset import get_hilbert_indices
from grid_transformer.training.ar import GraphormerAR
from grid_transformer.training.lightning_module import VQLatentTransformerModule


def _resolve_dtype(name: Optional[str]) -> Optional[torch.dtype]:
    if not name or name.lower() == "auto":
        return None
    mapping = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    key = name.lower()
    if key not in mapping:
        raise ValueError(f"Unknown torch dtype alias '{name}'.")
    return mapping[key]


def _token_order(h: int, w: int, use_hilbert: bool) -> torch.LongTensor:
    if use_hilbert:
        return get_hilbert_indices(h, w)
    yy, xx = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    return torch.stack([yy.reshape(-1), xx.reshape(-1)], dim=1).long()


def _seq_to_grid(seq: torch.LongTensor, h: int, w: int, order: torch.LongTensor) -> torch.LongTensor:
    grid = torch.empty((seq.size(0), h, w), dtype=seq.dtype, device=seq.device)
    ys = order[:, 0].to(device=seq.device)
    xs = order[:, 1].to(device=seq.device)
    for b in range(seq.size(0)):
        grid[b, ys, xs] = seq[b]
    return grid


@torch.no_grad()
def _infer_latent_shape_from_vq(vq_model, image_size: int = 32) -> Tuple[int, int, int]:
    dev = next(vq_model.parameters()).device
    dummy = torch.zeros(1, 3, image_size, image_size, device=dev)
    z_q = vq_model.encode(dummy).latents
    return int(z_q.shape[1]), int(z_q.shape[2]), int(z_q.shape[3])


def _load_lightning_checkpoint(path: str, device: torch.device):
    try:
        model = GraphormerAR.load_from_checkpoint(path, map_location=device)
        model.to(device).eval()
        return model, "graphormer_ar"
    except Exception as ar_exc:
        try:
            model = VQLatentTransformerModule.load_from_checkpoint(path, map_location=device)
            model.to(device).eval()
            return model, "vq_latent"
        except Exception as vq_exc:
            raise RuntimeError(
                f"Failed to load checkpoint '{path}' as GraphormerAR or VQLatentTransformerModule."
            ) from vq_exc


@torch.no_grad()
def iterative_binary_sample(
    model,
    H,
    W,
    steps=50,
    reveal_schedule="cosine",
    return_history=False,
    max_history=10,
):
    """MaskGIT-style sampling for binary occupancy grids."""
    device = next(model.parameters()).device
    x = torch.full((1, 1, H, W), -1.0, device=device)
    revealed = torch.zeros((1, 1, H, W), dtype=torch.bool, device=device)

    def frac(t):
        if reveal_schedule == "linear":
            return 1.0 / max(steps, 1)
        return 1.0 - np.cos(0.5 * np.pi * (t + 1) / max(steps, 1))

    history = [] if return_history else None

    for t in range(steps):
        logits = model(x)
        probs = torch.sigmoid(logits)
        mask = ~revealed
        if mask.sum() == 0:
            break
        conf = torch.abs(probs - 0.5)
        k = min(int(mask.sum().item()), max(1, int(mask.sum().item() * frac(t))))
        conf_masked = torch.where(mask, conf, torch.full_like(conf, -1.0))
        _, idx = torch.topk(conf_masked.view(-1), k)
        sel = torch.zeros_like(conf_masked, dtype=torch.bool).view(-1)
        sel[idx] = True
        sel = sel.view_as(conf_masked)
        samples = torch.bernoulli(probs)
        x = torch.where(sel, samples * 1.0, x)
        revealed = revealed | sel
        if return_history and len(history) < max_history:
            history.append(x.detach().cpu().clone())

    if (~revealed).sum() > 0:
        logits = model(x)
        probs = torch.sigmoid(logits)
        samples = torch.bernoulli(probs)
        x = torch.where(~revealed, samples * 1.0, x)

    if return_history:
        final_frame = x.detach().cpu().clone()
        if not history:
            history = [final_frame]
        elif not torch.equal(history[-1], final_frame):
            if len(history) < max_history:
                history.append(final_frame)
            else:
                history[-1] = final_frame
        return x, history

    return x, None


@torch.no_grad()
def iterative_latent_regression_sample(
    model,
    latent_channels,
    H,
    W,
    steps=12,
    reveal_schedule="cosine",
):
    """Iterative masked sampling when predicting latent values directly."""
    device = next(model.parameters()).device
    latents = torch.zeros((1, latent_channels, H, W), device=device)
    mask = torch.ones((1, 1, H, W), device=device)

    def frac(t):
        if reveal_schedule == "linear":
            return 1.0 / max(steps, 1)
        return 1.0 - np.cos(0.5 * np.pi * (t + 1) / max(steps, 1))

    for t in range(steps):
        preds = model(torch.cat([latents, mask], dim=1))
        masked_sites = mask > 0.5
        num_tokens = masked_sites.sum().item()
        if num_tokens == 0:
            break
        k = min(int(num_tokens), max(1, int(num_tokens * frac(t))))
        conf = torch.linalg.norm(preds, dim=1, keepdim=True)
        conf_masked = torch.where(masked_sites, conf, torch.full_like(conf, -1.0))
        _, idx = torch.topk(conf_masked.view(-1), k)
        sel = torch.zeros_like(conf_masked, dtype=torch.bool).view(-1)
        sel[idx] = True
        sel = sel.view_as(conf_masked)
        latents = torch.where(sel.expand(-1, latent_channels, -1, -1), preds, latents)
        mask = torch.where(sel, torch.zeros_like(mask), mask)

    if (mask > 0.5).any():
        preds = model(torch.cat([latents, mask], dim=1))
        latents = torch.where(mask.expand(-1, latent_channels, -1, -1) > 0.5, preds, latents)
        mask.zero_()

    return latents


@torch.no_grad()
def iterative_latent_logits_sample(
    model,
    vq_model,
    latent_channels,
    H,
    W,
    steps=12,
    reveal_schedule="cosine",
):
    """Iterative masked sampling when predicting VQ codebook logits."""
    device = next(model.parameters()).device
    codebook = vq_model.quantize.embedding.weight.to(device)
    latents = torch.zeros((1, latent_channels, H, W), device=device)
    mask = torch.ones((1, 1, H, W), device=device)
    indices = torch.full((1, H, W), -1, dtype=torch.long, device=device)

    def frac(t):
        if reveal_schedule == "linear":
            return 1.0 / max(steps, 1)
        return 1.0 - np.cos(0.5 * np.pi * (t + 1) / max(steps, 1))

    def update_selected(sel_mask: torch.Tensor, logits: torch.Tensor) -> None:
        if sel_mask.sum() == 0:
            return
        probs = torch.softmax(logits, dim=1)
        probs_hwk = probs.permute(0, 2, 3, 1)
        selected_probs = probs_hwk[sel_mask]
        sampled_codes = torch.multinomial(selected_probs, num_samples=1).squeeze(-1)
        positions = sel_mask.nonzero(as_tuple=False)
        b_idx = positions[:, 0]
        y_idx = positions[:, 1]
        x_idx = positions[:, 2]
        indices[b_idx, y_idx, x_idx] = sampled_codes
        embeds = codebook[sampled_codes]
        latents[b_idx, :, y_idx, x_idx] = embeds
        mask[b_idx, 0, y_idx, x_idx] = 0.0

    for t in range(steps):
        logits = model(torch.cat([latents, mask], dim=1))
        masked_sites = mask.squeeze(1) > 0.5
        num_tokens = masked_sites.sum().item()
        if num_tokens == 0:
            break
        k = min(int(num_tokens), max(1, int(num_tokens * frac(t))))
        probs = torch.softmax(logits, dim=1)
        conf = probs.max(dim=1).values
        conf_masked = torch.where(masked_sites, conf, torch.full_like(conf, -1.0))
        _, top_idx = torch.topk(conf_masked.view(-1), k)
        sel = torch.zeros_like(conf_masked, dtype=torch.bool).view(-1)
        sel[top_idx] = True
        sel = sel.view_as(conf_masked)
        update_selected(sel, logits)

    if (mask > 0.5).any():
        logits = model(torch.cat([latents, mask], dim=1))
        masked_sites = mask.squeeze(1) > 0.5
        update_selected(masked_sites, logits)

    return latents, indices

@torch.no_grad()
def iterative_token_ids_sample(
    lit_module,               # the Lightning module (not lit_module.model)
    Hc, Wc,                   # latent grid size
    steps=12,
    reveal_schedule="cosine",
    sample_mode="multinomial",  # or "argmax"
    temperature=1.0,
):
    """
    MaskGIT-style iterative reveal in *token-ID* space.
    Returns: indices  [1, Hc, Wc] (long in [0..K-1])
    """
    device = next(lit_module.parameters()).device
    K = lit_module.K                      # number of real tokens
    MASK_ID = K                           # we reserved K as [MASK] during training

    # start fully masked
    input_idx = torch.full((1, Hc, Wc), MASK_ID, dtype=torch.long, device=device)
    masked = torch.ones((1, Hc, Wc), dtype=torch.bool, device=device)

    def step_fraction(t):
        if reveal_schedule == "linear":
            return 1.0 / max(steps, 1)
        # cosine schedule in [0,1]
        return 1.0 - np.cos(0.5 * np.pi * (t + 1) / max(steps, 1))

    for t in range(steps):
        logits = lit_module(input_idx)           # [1, K, Hc, Wc]
        if temperature != 1.0:
            logits = logits / float(max(1e-8, temperature))
        prob = torch.softmax(logits, dim=1)      # [1, K, Hc, Wc]
        conf = prob.max(dim=1).values            # [1, Hc, Wc]

        # how many masked tokens to reveal this step
        num_masked = masked.sum().item()
        if num_masked == 0:
            break
        k = min(int(num_masked), max(1, int(num_masked * step_fraction(t))))

        # pick top-k most confident among masked
        conf_masked = torch.where(masked, conf, torch.full_like(conf, -1.0))
        flat = conf_masked.view(-1)
        _, top_idx = torch.topk(flat, k)
        sel = torch.zeros_like(flat, dtype=torch.bool)
        sel[top_idx] = True
        sel = sel.view_as(conf_masked)           # [1, Hc, Wc]

        # sample (or argmax) on those positions
        prob_hwk = prob.permute(0, 2, 3, 1)      # [1, Hc, Wc, K]
        picked_probs = prob_hwk[sel]             # [k, K]
        if sample_mode == "argmax":
            chosen = picked_probs.argmax(dim=1)          # [k]
        else:
            chosen = torch.multinomial(picked_probs, 1).squeeze(1)

        # write chosen token IDs into input_idx
        pos = sel.nonzero(as_tuple=False)        # [k, 3], columns: b,y,x
        b = pos[:, 0]; y = pos[:, 1]; x = pos[:, 2]
        input_idx[b, y, x] = chosen
        masked[b, y, x] = False

    # if anything remains masked, fill by one last sample
    if masked.any():
        logits = lit_module(input_idx)
        prob = torch.softmax(logits, dim=1).permute(0, 2, 3, 1)  # [1, Hc, Wc, K]
        remain = masked.nonzero(as_tuple=False)
        chosen = torch.multinomial(prob[masked], 1).squeeze(1)
        b = remain[:, 0]; y = remain[:, 1]; x = remain[:, 2]
        input_idx[b, y, x] = chosen

    return input_idx  # [1, Hc, Wc]


@torch.no_grad()
def autoregressive_token_sample(
    lit_module: GraphormerAR,
    Hc: int,
    Wc: int,
    *,
    use_hilbert: bool = True,
    sample_mode: str = "multinomial",
    temperature: float = 1.0,
    top_k: Optional[int] = None,
) -> Tuple[torch.LongTensor, torch.LongTensor]:
    """
    Autoregressive token sampling for GraphormerAR.
    Returns:
      seq:  [1, T] sampled token sequence
      grid: [1, Hc, Wc] unflattened token grid
    """
    device = next(lit_module.parameters()).device
    T = int(Hc * Wc)
    order = _token_order(Hc, Wc, use_hilbert=use_hilbert)
    coords_full = order.to(device=device, dtype=torch.float32).unsqueeze(0)  # [1,T,2]
    box_size = torch.tensor([[float(Hc), float(Wc)]], device=device, dtype=torch.float32)

    if lit_module.sos_id is None:
        raise ValueError("GraphormerAR checkpoint has no sos_id; cannot autoregressively sample.")
    sos_id = int(lit_module.sos_id)

    seq = torch.empty((1, T), dtype=torch.long, device=device)
    seq_in = torch.empty((1, T), dtype=torch.long, device=device)
    seq_in[:, 0] = sos_id

    for t in range(T):
        logits = lit_module(
            seq_in[:, : t + 1],
            coords=coords_full[:, : t + 1, :],
            box_size=box_size,
        )[:, -1, :]  # [1,K]
        if temperature != 1.0:
            logits = logits / float(max(1e-8, temperature))
        if top_k is not None and top_k > 0 and top_k < logits.shape[-1]:
            v, _ = torch.topk(logits, k=top_k, dim=-1)
            logits = torch.where(logits < v[..., -1:].expand_as(logits), torch.full_like(logits, -1e9), logits)
        probs = torch.softmax(logits, dim=-1)
        if sample_mode == "argmax":
            nxt = probs.argmax(dim=-1)
        else:
            nxt = torch.multinomial(probs, num_samples=1).squeeze(1)
        seq[:, t] = nxt
        if t + 1 < T:
            seq_in[:, t + 1] = nxt

    grid = _seq_to_grid(seq, Hc, Wc, order)
    return seq, grid

def load_vq_model(repo_or_path: str, subfolder: Optional[str], dtype: Optional[str], device: str, tmp_dir: Optional[str]):
    os.makedirs(tmp_dir, exist_ok=True)
    for env_name in ("TMPDIR", "TEMP", "TMP"):
        os.environ.setdefault(env_name, tmp_dir)
    tempfile.tempdir = tmp_dir
    from diffusers import VQModel  # delayed import

    torch_dtype = _resolve_dtype(dtype)
    model = VQModel.from_pretrained(repo_or_path, subfolder=subfolder, torch_dtype=torch_dtype)
    model = model.to(device)
    model.eval()
    model.requires_grad_(False)
    return model


def _decode_to_unit_interval(decoded: torch.Tensor) -> torch.Tensor:
    """Convert VQ decoder output from [-1, 1] to [0, 1]."""
    return ((decoded + 1.0) * 0.5).clamp(0.0, 1.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--Lx", type=float, required=True)
    ap.add_argument("--Ly", type=float, required=True)
    ap.add_argument("--pixel_size", type=float, default=0.05)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--nsamples", type=int, default=10, help="Number of independent samples to generate.")
    ap.add_argument("--save", type=str, default="out.npz")
    ap.add_argument("--image_dir", type=str, default=None, help="If set, save generated grids/images as PNG.")
    ap.add_argument("--reveal_schedule", type=str, default="cosine", choices=("cosine", "linear"))
    ap.add_argument("--vq_model", type=str, default=None, help="VQ-GAN repo id/path (required for latent checkpoints).")
    ap.add_argument("--vq_subfolder", type=str, default=None)
    ap.add_argument("--vq_dtype", type=str, default=None)
    ap.add_argument("--vq_device", type=str, default="cpu")
    ap.add_argument("--vq_tmp_dir", type=str, default=None, help="Optional temp dir override for diffusers.")
    ap.add_argument("--disable_hilbert", action="store_true", help="Use raster ordering instead of Hilbert for AR CIFAR sampling.")
    ap.add_argument("--sample_mode", type=str, default="multinomial", choices=("multinomial", "argmax"))
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top_k", type=int, default=None, help="Optional top-k truncation for AR sampling.")
    ap.add_argument("--image_size", type=int, default=32, help="Image size used to infer latent grid from VQ model for AR checkpoints.")
    args = ap.parse_args()

    W = int(np.floor(args.Lx / args.pixel_size))
    H = int(np.floor(args.Ly / args.pixel_size))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lit_module, model_kind = _load_lightning_checkpoint(args.ckpt, device)
    model = lit_module.model if model_kind == "vq_latent" else lit_module
    model.eval()

    if model_kind == "graphormer_ar":
        output_type = "latent_logits_ar"
    else:
        output_type = getattr(lit_module, "output_type", lit_module.hparams.get("output_type", "binary"))
    nsamples = max(1, args.nsamples)

    results = {}
    images_to_save = []

    if output_type == "latent_logits_ar":
        if not args.vq_model:
            raise ValueError("--vq_model is required to decode AR latent-token checkpoints.")
        tmp_dir = args.vq_tmp_dir or os.path.join(os.path.dirname(os.path.abspath(args.save)), "_vq_tmp")
        target_device = args.vq_device if args.vq_device is not None else str(device)
        vq_model = load_vq_model(args.vq_model, args.vq_subfolder, args.vq_dtype, target_device, tmp_dir)
        latent_channels, latent_H, latent_W = _infer_latent_shape_from_vq(vq_model, image_size=max(1, args.image_size))

        decoded_images = []
        latent_grids = []
        indices_list = []
        seq_list = []
        for _ in range(nsamples):
            seq, indices = autoregressive_token_sample(
                lit_module,
                Hc=latent_H,
                Wc=latent_W,
                use_hilbert=(not args.disable_hilbert),
                sample_mode=args.sample_mode,
                temperature=args.temperature,
                top_k=args.top_k,
            )  # seq:[1,T], indices:[1,Hc,Wc]
            E = vq_model.quantize.embedding.weight.to(vq_model.device)  # [K, D]
            latents = (
                E[indices.to(vq_model.device).view(-1)]
                .view(1, latent_H, latent_W, E.size(1))
                .permute(0, 3, 1, 2)
                .contiguous()
            )  # [1,D,Hc,Wc]
            decoded = _decode_to_unit_interval(vq_model.decode(latents, force_not_quantize=True).sample)

            latent_grids.append(latents.squeeze(0).detach().cpu().numpy().astype(np.float32))
            indices_list.append(indices.squeeze(0).detach().cpu().numpy().astype(np.int32))
            seq_list.append(seq.squeeze(0).detach().cpu().numpy().astype(np.int32))
            decoded_tensor = decoded.squeeze(0).detach().cpu()
            decoded_images.append(decoded_tensor)
            images_to_save.append(decoded_tensor)

        results["latents"] = np.stack(latent_grids, axis=0)
        results["indices"] = np.stack(indices_list, axis=0)
        results["seq"] = np.stack(seq_list, axis=0)
        imgs = torch.stack(decoded_images, dim=0)
        results["images"] = imgs.cpu().numpy()
        results["token_order"] = _token_order(latent_H, latent_W, use_hilbert=(not args.disable_hilbert)).cpu().numpy().astype(np.int32)
        results["use_hilbert"] = np.array(not args.disable_hilbert, dtype=np.bool_)
    elif output_type == "latent_logits":
        latent_shape = lit_module.hparams.get("latent_shape")
        if latent_shape is None:
            raise RuntimeError("Checkpoint missing latent_shape information.")
        latent_channels, latent_H, latent_W = latent_shape
        tmp_dir = args.vq_tmp_dir or os.path.join(os.path.dirname(os.path.abspath(args.save)), "_vq_tmp")
        if not args.vq_model:
            raise ValueError("--vq_model is required to decode latent checkpoints.")
        target_device = args.vq_device if args.vq_device is not None else str(device)
        vq_model = load_vq_model(args.vq_model, args.vq_subfolder, args.vq_dtype, target_device, tmp_dir)
        decoded_images = []
        latent_grids = []
        indices_list = []
        for _ in range(nsamples):
            indices = iterative_token_ids_sample(
                lit_module,
                Hc=latent_H, Wc=latent_W,
                steps=max(1, args.steps),
                reveal_schedule=args.reveal_schedule,
                sample_mode=args.sample_mode,
                temperature=args.temperature,
            )  # [1, Hc, Wc]
            E = vq_model.quantize.embedding.weight.to(vq_model.device)   # [K, D]
            latents = E[indices.to(vq_model.device).view(-1)].view(1, latent_H, latent_W, E.size(1)).permute(0, 3, 1, 2).contiguous()  # [1,D,Hc,Wc]
            # decode
            decoded = _decode_to_unit_interval(vq_model.decode(latents, force_not_quantize=True).sample)
            latent_grids.append(latents.squeeze(0).detach().cpu().numpy().astype(np.float32))
            indices_list.append(indices.squeeze(0).detach().cpu().numpy().astype(np.int32))
            decoded_tensor = decoded.squeeze(0).detach().cpu()
            decoded_images.append(decoded_tensor)
            images_to_save.append(decoded_tensor)
        results["latents"] = np.stack(latent_grids, axis=0)
        results["indices"] = np.stack(indices_list, axis=0)
        imgs = torch.stack(decoded_images, dim=0)
        results["images"] = imgs.cpu().numpy()
    elif output_type == "latent":
        latent_shape = lit_module.hparams.get("latent_shape")
        if latent_shape is None:
            raise RuntimeError("Checkpoint missing latent_shape information.")
        latent_channels, latent_H, latent_W = latent_shape
        tmp_dir = args.vq_tmp_dir or os.path.join(os.path.dirname(os.path.abspath(args.save)), "_vq_tmp")
        if not args.vq_model:
            raise ValueError("--vq_model is required to decode latent checkpoints.")
        target_device = args.vq_device if args.vq_device is not None else str(device)
        vq_model = load_vq_model(args.vq_model, args.vq_subfolder, args.vq_dtype, target_device, tmp_dir)
        decoded_images = []
        latent_grids = []
        for _ in range(nsamples):
            latents = iterative_latent_regression_sample(
                model,
                latent_channels=latent_channels,
                H=latent_H,
                W=latent_W,
                steps=max(1, args.steps),
                reveal_schedule=args.reveal_schedule,
            )
            latents = latents.to(vq_model.device)
            latents, _ = vq_model.quantize(latents)
            decoded = _decode_to_unit_interval(vq_model.decode(latents).sample)
            latents_np = latents.squeeze(0).detach().cpu().numpy().astype(np.float32)
            latent_grids.append(latents_np)
            decoded_tensor = decoded.squeeze(0).detach().cpu()
            decoded_images.append(decoded_tensor)
            images_to_save.append(decoded_tensor)
        results["latents"] = np.stack(latent_grids, axis=0)
        imgs = torch.stack(decoded_images, dim=0)
        results["images"] = imgs.cpu().numpy()
    else:
        final_grids = []
        coords_list = []
        history_to_save = None
        for i in range(nsamples):
            capture_history = args.image_dir is not None and history_to_save is None
            x, history = iterative_binary_sample(
                model,
                H=H,
                W=W,
                steps=args.steps,
                reveal_schedule=args.reveal_schedule,
                return_history=capture_history,
                max_history=10,
            )
            grid = x.squeeze().cpu().numpy().astype(np.uint8)
            final_grids.append(grid)
            ys, xs = np.where(grid > 0.5)
            coords = np.stack([
                (xs + 0.5) * args.pixel_size,
                (ys + 0.5) * args.pixel_size,
            ], axis=1).astype(np.float32)
            coords_list.append(coords)
            if capture_history and history:
                history_to_save = history
        if nsamples == 1:
            results["grid"] = final_grids[0]
            results["coords"] = coords_list[0]
        else:
            results["grid"] = np.stack(final_grids, axis=0)
            results["coords"] = np.array(coords_list, dtype=object)
        results["L"] = np.array([args.Lx, args.Ly], dtype=np.float32)
        results["pixel_size"] = np.array(args.pixel_size, dtype=np.float32)
        if args.image_dir and history_to_save:
            for tensor in history_to_save[:10]:
                images_to_save.append((tensor.squeeze() > 0.5).float())
        for grid in final_grids[:10]:
            images_to_save.append(torch.from_numpy(grid[None, ...].astype(np.float32)))

    np.savez(args.save, **results)
    print(f"Saved {args.save}  (samples: {nsamples})")

    if args.image_dir:
        os.makedirs(args.image_dir, exist_ok=True)
        for idx, tensor in enumerate(images_to_save[:10]):
            if isinstance(tensor, torch.Tensor):
                tensor = tensor.cpu()
            data = tensor.numpy()
            if data.ndim == 3 and data.shape[0] in (1, 3):
                if data.shape[0] == 1:
                    array = (data.squeeze(0).clip(0.0, 1.0) * 255.0).astype(np.uint8)
                else:
                    array = (data.transpose(1, 2, 0).clip(0.0, 1.0) * 255.0).astype(np.uint8)
            else:
                array = ((data.squeeze() > 0.5).astype(np.uint8)) * 255
            Image.fromarray(array).save(os.path.join(args.image_dir, f"sample_{idx:02d}.png"))
        print(f"Saved image previews to {args.image_dir}")


if __name__ == "__main__":
    main()
