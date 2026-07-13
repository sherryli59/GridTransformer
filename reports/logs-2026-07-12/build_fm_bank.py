"""Precompute the FM training bank: the AR base is FROZEN, so (gated AR draw, OT-matched data block,
fixed-size cage) tuples are flow-independent and reusable across every retrain. Profile
(profile_train_step.py): data pipeline = 6.5-11.3 s/step of the 8-10 s step; loss fwd+bwd only ~1.8 s.
Banking moves ~80% of the step offline. t-sampling, interpolant, and SO(3) augmentation stay ONLINE
(fresh every step), so only the AR draws are reused — with >=100k rows revisited ~10x over a full run,
standard and harmless.

Row: x0_block[K,3] (gated AR), x1_block[K,3] (per-species-Hungarian-matched data), sp_block[K],
cage[48,3], sp_cage[48]. Shards of SHARD_CAV cavities saved incrementally (resumable: existing shards
are skipped). Train bank from the 112 train chains; held bank from the 48 held configs (chain-level
split preserved).

Usage: python build_fm_bank.py [--n_cav 6250] [--held_n_cav 128] [--gate_cut 0.55]
"""
import argparse
import sys
import time
from pathlib import Path
import torch

sys.path.insert(0, "reports/logs-2026-07-12")
from train_cavity_egnn_flow import (load_ar_model, load_split, carve_random_cavity, fixed_size_cage,
                                    TRAIN_DATA, HELD_DATA)
from liquid_coupling_flow.ka3d_gated_base import GatedARBase
from fm_data import fm_batch

P = argparse.ArgumentParser()
P.add_argument("--n_cav", type=int, default=6250)          # x M=16 draws -> ~100k train rows
P.add_argument("--held_n_cav", type=int, default=128)      # x 16 -> ~2k held rows
P.add_argument("--m", type=int, default=16)
P.add_argument("--k", type=int, default=8)
P.add_argument("--n_cage", type=int, default=48)
P.add_argument("--gate_cut", type=float, default=0.55)
P.add_argument("--gate_rounds", type=int, default=32)
P.add_argument("--pos_temp", type=float, default=0.4)
P.add_argument("--radii", type=float, nargs="+", default=[1.6, 2.0, 2.5, 3.0])
P.add_argument("--shard_cav", type=int, default=256)
P.add_argument("--device", default="cuda")
P.add_argument("--seed", type=int, default=0)
args = P.parse_args()
dev = args.device

OUT_DIR = Path("liquid_coupling_flow/artifacts/fm_bank")
OUT_DIR.mkdir(exist_ok=True)

ar = GatedARBase(load_ar_model(dev), cut=args.gate_cut, max_rounds=args.gate_rounds)


def build(split_name, data_path, expect, n_cav_total, seed):
    X, S, L = load_split(data_path, dev, expect_chains=expect)
    gen = torch.Generator(device=dev).manual_seed(seed)
    n_shards = (n_cav_total + args.shard_cav - 1) // args.shard_cav
    t0 = time.time()
    for sh in range(n_shards):
        out = OUT_DIR / f"{split_name}_shard{sh:04d}.pt"
        ncav_shard = min(args.shard_cav, n_cav_total - sh * args.shard_cav)
        if out.exists():
            print(f"[bank] {out.name} exists -- skip (resume)", flush=True)
            # keep the RNG stream position deterministic on resume: burn the shard's cavities
            for _ in range(ncav_shard):
                cav = carve_random_cavity(X, S, L, gen, args.k, args.radii)
                if cav is not None:
                    fm_batch(ar, cav, gen, pos_temp=args.pos_temp, M=args.m)
            continue
        rows = {kk: [] for kk in ("x0", "x1", "sp", "cage", "sp_cage")}
        made = 0
        for _ in range(ncav_shard):
            cav = carve_random_cavity(X, S, L, gen, args.k, args.radii)
            if cav is None:
                continue
            o = fm_batch(ar, cav, gen, pos_temp=args.pos_temp, M=args.m)
            keep = ar.last_pass_mask
            if not keep.any():
                continue
            centroid = cav["xo"][cav["block_mask"]].mean(0)
            cfx, cfs = fixed_size_cage(o["cage_x"], o["sp_cage"], centroid, args.n_cage,
                                       o["cage_x"].device, o["cage_x"].dtype)
            rows["x0"].append(o["x0_block"][keep].cpu())
            rows["x1"].append(o["x1_block"][keep].cpu())
            rows["sp"].append(o["sp_block"][keep].cpu())
            rows["cage"].append(cfx[keep].cpu())
            rows["sp_cage"].append(cfs[keep].cpu())
            made += 1
        shard = {kk: torch.cat(vv, 0) for kk, vv in rows.items()}
        shard["meta"] = {"gate_cut": args.gate_cut, "pos_temp": args.pos_temp, "k": args.k,
                         "n_cage": args.n_cage, "radii": args.radii, "split": split_name,
                         "n_cavities": made, "seed": seed, "shard": sh}
        torch.save(shard, out)
        nrows = shard["x0"].shape[0]
        el = time.time() - t0
        print(f"[bank] {split_name} shard {sh+1}/{n_shards}: {made} cavities -> {nrows} rows "
              f"SAVED ({el/60:.1f} min elapsed)", flush=True)


build("held", HELD_DATA, None, args.held_n_cav, seed=args.seed + 7000)
build("train", TRAIN_DATA, 112, args.n_cav, seed=args.seed)
print("[bank] DONE", flush=True)
