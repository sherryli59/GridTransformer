"""Teacher-forced NLL of an arc_repr checkpoint on a preprocessed cache.

Size-transfer parity check (roadmap Step 4a): evaluate the SAME checkpoint on caches
built at different box sizes (constant cell via pow2 R). Identical normalization to
training (per-coordinate mean from training_step), so values are comparable across
sizes and against the run's logged train/loss_epoch (which doubles as a correctness
cross-check for this script at the training size).

Example:
  python eval_arc_nll.py --ckpt /tmp/arc_repr_ep210.ckpt \
    --cache lj_caches_arc_transfer/lj64_L4_R128_cache.pt --limit 2000
"""
from __future__ import annotations

import argparse

import torch

from grid_transformer.models.transformer import GraphormerAR
from grid_transformer.training.lightning_module import LJTransferableDataModule


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--cache", type=str, required=True)
    ap.add_argument("--limit", type=int, default=2000, help="Number of cached configs to evaluate.")
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--num_workers", type=int, default=2)
    args = ap.parse_args()

    model = GraphormerAR.load_from_checkpoint(args.ckpt, map_location="cpu")
    model.eval()
    # training_step calls self.log; outside a Trainer that's a no-op stub.
    model.log = lambda *a, **k: None  # type: ignore[method-assign]

    dm = LJTransferableDataModule(
        preprocessed_path=args.cache,
        arc_repr=bool(getattr(model, "arc_repr", False)),
        batch_size=args.batch_size,
        train_limit=args.limit,
        num_workers=args.num_workers,
        pin_memory=False,
    )

    total, count = 0.0, 0
    with torch.no_grad():
        for batch in dm.train_dataloader():
            loss = model.training_step(batch, 0)
            bsz = int(batch["box_size"].shape[0])
            total += float(loss) * bsz
            count += bsz
    print(f"cache={args.cache}")
    print(f"configs={count}  teacher-forced NLL (training_step loss, per-coordinate): {total / max(count, 1):.4f}")


if __name__ == "__main__":
    main()
