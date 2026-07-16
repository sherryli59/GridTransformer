"""Balanced-size MLE trainer for the exact global mW cell q0 base."""
from __future__ import annotations

import argparse
from pathlib import Path
import time

import torch

from liquid_coupling_flow.mw.mw_cell_data import (
    frame_banks,
    fresh_augmented_state,
    load_or_build_raw_cache,
)
from liquid_coupling_flow.mw.mw_cell_q0 import MWCellQ0
from liquid_coupling_flow.mw.mw_cell_transformer import (
    DEFAULT_SELECTED_SCAFFOLD,
    JULY14_MLE_SCAFFOLD,
    warm_start_cell_from_scaffold,
)


ART = Path(__file__).resolve().parent / "artifacts" / "mw_cell_q0_runs"


def _grid_for_N(N: int) -> int:
    if int(N) == 64:
        return 2
    if int(N) == 512:
        return 4
    raise ValueError("v1 cell training supports N=64 or N=512")


def _draw(frames: torch.Tensor, batch: int, device, gen):
    indices = torch.randint(len(frames), (int(batch),), generator=gen)
    return frames[indices].to(device, non_blocking=(device.type == "cuda"))


@torch.no_grad()
def evaluate(model, banks, batch_per_size, device, cpu_gen, dev_gen):
    was_training = model.training
    model.eval()
    result, diagnostics = {}, {}
    for bank in banks:
        x = _draw(bank.validation, batch_per_size, device, cpu_gen)
        state = fresh_augmented_state(x, bank.L, _grid_for_N(bank.N), gen=dev_gen)
        G = _grid_for_N(bank.N)
        result[bank.N] = float((-model.log_prob(state, bank.L, G) / bank.N).mean())
        _, terms = model.position_terms_by_occupancy(state, bank.L, G)
        count_nll = (-model.count_log_probs(state, bank.L, G) / (G ** 3 - 1)).mean()
        diagnostics[bank.N] = {
            "count_nll_per_factor": float(count_nll),
            "position_nll_by_K": {
                int(K): float((-values / K).mean()) for K, values in terms.items()
            },
            "cell_count_by_K": {int(K): int(values.numel()) for K, values in terms.items()},
        }
    model.train(was_training)
    return result, diagnostics


def _checkpoint(model, opt, step, vals, metadata):
    return {
        "state_dict": model.state_dict(), "optimizer": opt.state_dict(),
        "step": int(step), "val_nll_per_particle": vals,
        "val_nll_mean": sum(vals.values()) / len(vals), **metadata,
    }


def train(*, sources, cache, out="mw_cell_q0.pt", steps=20_000,
          batch_per_size=2, validation_batch=4, lr=2e-5,
          cache_train_per_size=4096, cache_validation_per_size=256,
          eval_every=100, seed=20260715, device=None, amp=True,
          warm=DEFAULT_SELECTED_SCAFFOLD, resume=None, balanced_k=True,
          count_weight=1.0, count_lr=5e-4):
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    cache_payload = load_or_build_raw_cache(
        sources, cache, train_per_size=cache_train_per_size,
        validation_per_size=cache_validation_per_size, seed=seed,
    )
    banks = frame_banks(cache_payload)
    sizes = [b.N for b in banks]
    if len(set(sizes)) != len(sizes):
        raise ValueError("at most one source per particle count in v1")
    model = MWCellQ0().to(device)
    warm_meta = warm_start_cell_from_scaffold(model, warm)
    # Coordinate likelihood needs the conservative scaffold fine-tune rate.
    # The count residual starts at zero around a sound multinomial base, so a
    # separate faster rate is needed to learn liquid count correlations on a
    # visible timescale (and to wake its hidden layers after the first update).
    position_params = [
        parameter for name, parameter in model.named_parameters()
        if not name.startswith("count_model.")
    ]
    opt = torch.optim.AdamW([
        {"params": position_params, "lr": float(lr)},
        {"params": model.count_model.parameters(), "lr": float(count_lr)},
    ], weight_decay=1e-4)
    start = 0
    if resume:
        payload = torch.load(resume, map_location=device, weights_only=False)
        model.load_state_dict(payload["state_dict"], strict=True)
        opt.load_state_dict(payload["optimizer"])
        start = int(payload["step"])
    cpu_gen = torch.Generator().manual_seed(int(seed) + 1)
    dev_gen = torch.Generator(device=device).manual_seed(int(seed) + 2)
    metadata = {
        "architecture": "MWCellQ0_periodic_KNN_Cat3", "sources": [str(p) for p in sources],
        "cache": str(cache), "sizes": sizes, "G_by_N": {b.N: _grid_for_N(b.N) for b in banks},
        "batch_per_size": int(batch_per_size), "effective_batch": int(batch_per_size) * len(banks),
        "augmentation": cache_payload["augmentation"], "warm_start": warm_meta,
        "july14_mle_ablation": str(JULY14_MLE_SCAFFOLD), "lr": float(lr),
        "amp_bf16": bool(amp and device.type == "cuda"),
        "pos_temp": model.pos_temp, "mw_pair_tilt": model.mw_pair_tilt,
        "mw_three_tilt": model.mw_three_tilt,
        "balanced_K_position_loss": bool(balanced_k),
        "count_tree_weight": float(count_weight),
        "count_tree_lr": float(count_lr),
    }
    out_path = Path(out) if Path(out).is_absolute() else ART / out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    last_path = out_path.with_name(out_path.stem + "_last.pt")
    best = float("inf")
    print(f"cell q0 MLE sizes={sizes} batch_per_size={batch_per_size} "
          f"effective_batch={metadata['effective_batch']} cache={cache} "
          f"warm={warm_meta['checkpoint']} source_step={warm_meta['source_step']} "
          f"balanced_K={bool(balanced_k)} count_lr={float(count_lr):g}", flush=True)
    for step in range(start, start + int(steps) + 1):
        if step == start or step % int(eval_every) == 0 or step == start + int(steps):
            vals, k_diag = evaluate(model, banks, validation_batch, device, cpu_gen, dev_gen)
            mean = sum(vals.values()) / len(vals)
            print("step %6d val_nll/N={%s} mean=%+.5f" % (
                step, ",".join(f"{n}:{v:+.4f}" for n, v in vals.items()), mean), flush=True)
            for N, diag in k_diag.items():
                text = ",".join(f"{K}:{v:+.3f}" for K, v in sorted(diag["position_nll_by_K"].items()))
                print(f"step {step:6d} val_count_nll/factor N={N} "
                      f"{diag['count_nll_per_factor']:+.5f}", flush=True)
                print(f"step {step:6d} val_position_nll/K N={N} {{{text}}}", flush=True)
            state = _checkpoint(model, opt, step, vals, metadata)
            state["val_position_nll_by_K"] = k_diag
            torch.save(state, last_path)
            if mean < best:
                best = mean
                torch.save(state, out_path)
            if step == start + int(steps):
                break

        opt.zero_grad(set_to_none=True)
        losses, count_losses, position_losses = {}, {}, {}
        t0 = time.perf_counter()
        for bank in banks:
            x = _draw(bank.train, batch_per_size, device, cpu_gen)
            state = fresh_augmented_state(x, bank.L, _grid_for_N(bank.N), gen=dev_gen)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                enabled=bool(amp and device.type == "cuda")):
                if balanced_k:
                    nll, balance_diag = model.balanced_nll(
                        state, bank.L, _grid_for_N(bank.N), count_weight=count_weight
                    )
                else:
                    nll = (-model.log_prob(state, bank.L, _grid_for_N(bank.N)) / bank.N).mean()
                    balance_diag = None
            if not torch.isfinite(nll):
                raise FloatingPointError(f"non-finite NLL at step={step+1} N={bank.N}")
            (nll / len(banks)).backward()
            losses[bank.N] = float(nll.detach())
            if balance_diag is not None:
                count_losses[bank.N] = float(balance_diag["count_nll"])
                position_losses[bank.N] = float(balance_diag["position_nll"])
        grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        if (step + 1) % 5 == 0:
            print("train %6d objective/N={%s} count/factor={%s} position={%s} grad=%.3f sec=%.3f" % (
                step + 1, ",".join(f"{n}:{v:+.4f}" for n, v in losses.items()),
                ",".join(f"{n}:{v:+.4f}" for n, v in count_losses.items()),
                ",".join(f"{n}:{v:+.4f}" for n, v in position_losses.items()),
                float(grad), time.perf_counter() - t0), flush=True)
    return model


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sources", nargs="+", default=[
        "liquid_coupling_flow/mw/artifacts/mw_bank_ambient_N64.pt",
        "liquid_coupling_flow/mw/artifacts/mw_ref_N512_traj_train_v1.pt",
    ])
    p.add_argument("--cache", default="liquid_coupling_flow/mw/artifacts/mw_cell_q0_runs/raw_frames.pt")
    p.add_argument("--out", default="mw_cell_q0.pt")
    p.add_argument("--steps", type=int, default=20_000)
    p.add_argument("--batch-per-size", type=int, default=2)
    p.add_argument("--validation-batch", type=int, default=4)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--cache-train-per-size", type=int, default=4096)
    p.add_argument("--cache-validation-per-size", type=int, default=256)
    p.add_argument("--eval-every", type=int, default=100)
    p.add_argument("--seed", type=int, default=20260715)
    p.add_argument("--device", default=None)
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--warm", default=str(DEFAULT_SELECTED_SCAFFOLD))
    p.add_argument("--resume", default=None)
    p.add_argument("--physical-k-weighting", action="store_true",
                   help="disable balanced occupied-K position reweighting")
    p.add_argument("--count-weight", type=float, default=1.0,
                   help="weight for count NLL normalized per categorical factor")
    p.add_argument("--count-lr", type=float, default=5e-4,
                   help="optimizer LR for the count-tree residual (independent of coordinate LR)")
    a = p.parse_args()
    train(sources=a.sources, cache=a.cache, out=a.out, steps=a.steps,
          batch_per_size=a.batch_per_size, validation_batch=a.validation_batch, lr=a.lr,
          cache_train_per_size=a.cache_train_per_size,
          cache_validation_per_size=a.cache_validation_per_size,
          eval_every=a.eval_every, seed=a.seed, device=a.device,
          amp=not a.no_amp, warm=a.warm, resume=a.resume,
          balanced_k=not a.physical_k_weighting, count_weight=a.count_weight,
          count_lr=a.count_lr)


if __name__ == "__main__":
    main()
