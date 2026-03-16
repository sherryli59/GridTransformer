from __future__ import annotations

from typing import Optional

import torch

from .transformer import GraphormerAR as GraphormerARStd
from .transformer_ida import GraphormerAR as GraphormerARIDA
from .vanilla_transformer import VanillaTransformerAR

AR_ARCH_CHOICES = ("auto", "standard", "ida", "vanilla")

_AR_ARCH_ALIASES = {
    "auto": "auto",
    "standard": "standard",
    "std": "standard",
    "transformer": "standard",
    "edge_bias": "standard",
    "baseline_edge_bias": "standard",
    "ida": "ida",
    "transformer_ida": "ida",
    "vanilla": "vanilla",
    "notebook": "vanilla",
    "mcmc_lj_13": "vanilla",
    "simple": "vanilla",
}


def normalize_ar_arch(name: Optional[str]) -> str:
    if name is None:
        return "auto"
    key = str(name).strip().lower()
    try:
        return _AR_ARCH_ALIASES[key]
    except KeyError as exc:
        raise ValueError(f"Unknown AR architecture '{name}'. Expected one of {AR_ARCH_CHOICES}.") from exc


def resolve_ar_arch(ar_arch: Optional[str], *, use_ida: Optional[bool] = None) -> str:
    normalized = normalize_ar_arch(ar_arch)
    if normalized != "auto":
        return normalized
    if use_ida is None:
        return "auto"
    return "ida" if bool(use_ida) else "standard"


def get_ar_model_class(ar_arch: str):
    arch = normalize_ar_arch(ar_arch)
    if arch == "standard":
        return GraphormerARStd
    if arch == "ida":
        return GraphormerARIDA
    if arch == "vanilla":
        return VanillaTransformerAR
    raise ValueError(f"Cannot resolve model class for architecture '{ar_arch}'.")


def infer_ar_arch_from_checkpoint(path: str) -> str:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")

    hparams = payload.get("hyper_parameters", {})
    if hasattr(hparams, "items"):
        hparams = dict(hparams)
    else:
        hparams = {}

    explicit = hparams.get("ar_arch")
    if explicit is not None:
        return normalize_ar_arch(str(explicit))

    if "permute_train" in hparams or "permute_group_size" in hparams:
        return "vanilla"
    if any(k in hparams for k in ("use_edge_bias", "use_rbf_bias", "use_deep_ida", "use_ida_pre")):
        return "standard"
    if "use_ida" in hparams:
        return "ida"

    state_dict = payload.get("state_dict", {})
    keys = tuple(state_dict.keys()) if hasattr(state_dict, "keys") else ()
    if any(k.startswith("transformer.layers.") for k in keys):
        return "vanilla"
    if any(k.startswith("ida.") for k in keys):
        return "ida"
    return "standard"


def load_ar_checkpoint(
    path: str,
    device: torch.device,
    *,
    ar_arch: str = "auto",
    use_ida: Optional[bool] = None,
):
    resolved_arch = resolve_ar_arch(ar_arch, use_ida=use_ida)
    if resolved_arch == "auto":
        resolved_arch = infer_ar_arch_from_checkpoint(path)
    model_cls = get_ar_model_class(resolved_arch)
    try:
        model = model_cls.load_from_checkpoint(path, map_location=device)
    except RuntimeError as exc:
        msg = str(exc)
        far_bias_mismatch = (
            "Missing key(s) in state_dict: \"edge_bias.far_bias\"" in msg
            or "Unexpected key(s) in state_dict: \"edge_bias.far_bias\"" in msg
        )
        if resolved_arch != "standard" or not far_bias_mismatch:
            raise RuntimeError(
                f"Failed to load checkpoint {path} with architecture '{resolved_arch}'."
            ) from exc
        model = model_cls.load_from_checkpoint(path, map_location=device, strict=False)
    model.to(device).eval()
    return model, resolved_arch
