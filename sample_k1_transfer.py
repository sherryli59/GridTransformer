# sample_k1_transfer.py
"""Zero-shot sampling of the trained K=1 rail checkpoint at a larger box.

Forces the rail grid resolution so the physical cell size matches training
(CELL_TRAIN), since the checkpoint stores cell_size=None (which would otherwise
pin R=64 and let the cell grow with the box). Usage mirrors sample_lj.py; add
--transfer_strategy {pow2,exact}.

Example:
  python sample_k1_transfer.py --transfer_strategy pow2 \
    --mode relative --ckpt <best.ckpt> --Lx 4 --Ly 4 --Lz 4 \
    --coord_dim 3 --num_particles 64 --periodic --ar_arch standard \
    --use_continuous_head --full_covariance --nsamples 256 \
    --sample_batch_size 128 --temperature 0.9 --relative_window 3.0 \
    --relative_bins 64 --device cpu --save <out.npz>
"""
import sys
import math
import numpy as np

import sample_lj

CELL_TRAIN = 3.0 / 64.0


def _strategy_R(L: float, strategy: str) -> int:
    n = max(2, int(round(L / CELL_TRAIN)))
    if strategy == "exact":
        return n
    return int(1 << int(math.ceil(math.log2(n))))  # pow2


def main() -> None:
    # Pop our extra flag before sample_lj parses argv.
    strategy = "pow2"
    if "--transfer_strategy" in sys.argv:
        i = sys.argv.index("--transfer_strategy")
        strategy = sys.argv[i + 1]
        del sys.argv[i:i + 2]

    _orig = sample_lj._rail_resolution_for_box

    def _forced(box_np, hilbert_resolution, cell_size):
        L = float(np.max(np.asarray(box_np, dtype=np.float64)))
        R = _strategy_R(L, strategy)
        print(f"[transfer] strategy={strategy} L={L} -> R={R} cell={L / R:.5f}")
        return int(R)

    sample_lj._rail_resolution_for_box = _forced
    try:
        sample_lj.main()
    finally:
        sample_lj._rail_resolution_for_box = _orig


if __name__ == "__main__":
    main()
