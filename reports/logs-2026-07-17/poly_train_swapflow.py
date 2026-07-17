"""FM trainer for SwapBlockFlow (liquid_coupling_flow/poly/swap_flow.py, poly Task J3b, PHASE 1 of
docs/superpowers/specs/2026-07-17-joint-sigma-x-continuous-flow-amendment.md's DECISION 2026-07-17).

Mirrors poly_train_blockflow.py's FM trainer conventions (build-batch-then-single-vel_div-call,
masked-MSE over real movers, incremental best-ckpt save, loss + v=0 floor prints) but conditions on
the swap-endpoint sigma(t) PATH instead of a fixed post-permutation species assignment:

FM setup: x0 = x_old + BASE_W*randn (short-transport base, SAME prior SwapBlockFlow.propose_batch
samples from), x1 = x_new, t ~ U(0,1) PER PAIR (one scalar t per block, matching
poly_train_blockflow.py's convention -- NOT one t per particle), x_t = (1-t)*x0 + t*x1, target
v = x1 - x0. Mover species labels at THIS t are `labels_at_t(sig_start, sig_end, t, n_sig_bins)` --
the exact conditioning `SwapBlockFlow._integrate` recomputes at every RK4 sub-stage from its own
running t, so training must evaluate species at the SAME (x_t, t) pair the field sees at inference,
not at the path's t=1 endpoint (unlike block_flow.py's fixed-post-permutation species, swap_flow's
species assignment genuinely varies along the path). Env species are t-independent (sig_start==
sig_end there): labels_at_t(env_sig, env_sig, 0.0, n_sig_bins), same call SwapBlockFlow._integrate
itself makes.

Loss regresses flow.ce.vel_div directly -- the EXACT field the RK4 integrator in propose_batch/
logq_of_batch uses, never a separate forward -- masked-MSE over real (non-dummy) mover rows only
(dummies excluded from numerator AND denominator via the real-mover mask), matching
poly_train_blockflow.py's / train_cavity_ersi.py's convention.

Padding reuses SwapBlockFlow's own batched `_prep_movers`/`_prep_env` (never a hand-rolled padding
routine) so training-time padding is byte-identical to inference-time padding, including the
sig_start=sig_end=SIG_MIN dummy-row convention documented in swap_flow.py's module docstring.

Usage:
    poly_train_swapflow.py --T 0.085 --k 8 --steps 300 --pairs_file <path> [--out PATH] [--m_env 64] ...
"""
from __future__ import annotations
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path("/mnt/ssd/GridTransformer")
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "reports/logs-2026-07-17"))

from liquid_coupling_flow.poly.swap_flow import SwapBlockFlow, labels_at_t, BASE_W  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--T", type=float, required=True)
p.add_argument("--k", type=int, required=True)
p.add_argument("--steps", type=int, default=20000)
p.add_argument("--pairs_file", type=str, required=True)
p.add_argument("--out", type=str, default=None)
p.add_argument("--m_env", type=int, default=64)
p.add_argument("--hidden", type=int, default=128)
p.add_argument("--layers", type=int, default=4)
p.add_argument("--n_sig_bins", type=int, default=8)
p.add_argument("--lr", type=float, default=3e-4)
p.add_argument("--batch", type=int, default=32)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--print_every", type=int, default=100)
a = p.parse_args()

OUT = Path(a.out) if a.out else REPO / f"liquid_coupling_flow/artifacts/poly_swapflow_T{a.T}_k{a.k}_best.pt"
dev = "cuda" if torch.cuda.is_available() else "cpu"


def load_pairs(path):
    """poly_swap_pairs.py saves {'pairs': [...], 'meta': vars(args)} -- unlike poly_block_data.py's
    plain-list format, so unwrap the dict here."""
    d = torch.load(path, weights_only=False)
    pairs = d["pairs"] if isinstance(d, dict) and "pairs" in d else d
    assert len(pairs) > 0, f"empty pairs file {path}"
    assert pairs[0]["x_old"].shape[0] == a.k, (
        f"pairs file block size {pairs[0]['x_old'].shape[0]} != --k {a.k}")
    return pairs


def build_batch(pairs, idxs, flow, gen):
    """One FM minibatch. Per pair: draw base noise + random t, form x_t/target_v, pad movers/env via
    flow's OWN batched _prep_movers/_prep_env (called with an explicit B=1 leading dim then squeezed --
    those methods are batched-only, unlike PolyBlockFlow's single-sample _prep_movers/_prep_env), then
    compute this pair's mover species labels at ITS OWN t via labels_at_t (the conditioning signal
    genuinely varies per row since each row has its own t, unlike block_flow.py's fixed post-perm
    species). Stacked into one batch so a single flow.ce.vel_div call covers the whole minibatch --
    same structure as poly_train_blockflow.py's build_batch."""
    k = flow.k_max
    xt_rows, sp_rows, cage_rows, spc_rows, tv_rows, t_rows, real_rows = [], [], [], [], [], [], []
    for i in idxs:
        pr = pairs[i]
        x_old = torch.as_tensor(pr["x_old"], dtype=torch.float32)
        x_new = torch.as_tensor(pr["x_new"], dtype=torch.float32)
        sig_start = torch.as_tensor(pr["sig_start"], dtype=torch.float64)
        sig_end = torch.as_tensor(pr["sig_end"], dtype=torch.float64)
        env_x = torch.as_tensor(pr["env_x"], dtype=torch.float32)
        env_sig = torch.as_tensor(pr["env_sig"], dtype=torch.float64)

        n = x_old.shape[0]
        noise = torch.randn(n, 3, generator=gen)
        x0 = x_old + BASE_W * noise
        x1 = x_new
        t = torch.rand((), generator=gen).item()
        x_t = (1.0 - t) * x0 + t * x1
        target_v = x1 - x0

        movers_full, s0_full, s1_full, n_real = flow._prep_movers(
            x_t.unsqueeze(0), sig_start.unsqueeze(0), sig_end.unsqueeze(0))
        movers_full = movers_full.squeeze(0); s0_full = s0_full.squeeze(0); s1_full = s1_full.squeeze(0)
        mover_labels = labels_at_t(s0_full, s1_full, t, flow.n_sig_bins)          # [k_max] long

        env_full, env_sig_full, _ = flow._prep_env(env_x.unsqueeze(0), env_sig.unsqueeze(0))
        env_full = env_full.squeeze(0); env_sig_full = env_sig_full.squeeze(0)
        env_labels = labels_at_t(env_sig_full, env_sig_full, 0.0, flow.n_sig_bins)  # [m_env] long

        real = torch.zeros(k, dtype=torch.bool); real[:n_real] = True
        tv_full = torch.zeros(k, 3); tv_full[:n_real] = target_v[:n_real]

        xt_rows.append(movers_full); sp_rows.append(mover_labels)
        cage_rows.append(env_full); spc_rows.append(env_labels)
        tv_rows.append(tv_full); t_rows.append(t); real_rows.append(real)

    x_t_b = torch.stack(xt_rows).to(dev)
    cage_b = torch.stack(cage_rows).to(dev)
    sp_b = torch.cat([torch.stack(sp_rows), torch.stack(spc_rows)], dim=1).to(dev)
    tv_b = torch.stack(tv_rows).to(dev)
    t_b = torch.tensor(t_rows, dtype=torch.float32, device=dev)
    real_b = torch.stack(real_rows).to(dev)
    return x_t_b, cage_b, sp_b, tv_b, t_b, real_b


def train():
    pairs = load_pairs(a.pairs_file)
    print(f"poly swap-flow FM training: T={a.T} k={a.k} m_env={a.m_env} hidden={a.hidden} "
          f"layers={a.layers} n_sig_bins={a.n_sig_bins} | {len(pairs)} pairs | steps={a.steps} "
          f"batch={a.batch} | device={dev}", flush=True)

    flow = SwapBlockFlow(k_max=a.k, m_env=a.m_env, hidden_nf=a.hidden, n_layers=a.layers,
                          n_sig_bins=a.n_sig_bins).to(dev)
    opt = torch.optim.Adam(flow.parameters(), lr=a.lr)
    gen = torch.Generator().manual_seed(a.seed)             # CPU generator (cuda generator crashes randn)
    np_rng = np.random.default_rng(a.seed)

    best = float("inf")
    loss_hist, floor_hist = [], []
    t0 = time.time()
    for step in range(1, a.steps + 1):
        idxs = np_rng.integers(0, len(pairs), size=a.batch)
        x_t_b, cage_b, sp_b, tv_b, t_b, real_b = build_batch(pairs, idxs, flow, gen)
        cloud = torch.cat([x_t_b, cage_b], dim=1)
        vel, _ = flow.ce.vel_div(cloud, t_b, sp_b, flow.k_max)          # the EXACT field the RK4 integrates
        sqerr = ((vel - tv_b) ** 2).sum(-1)                             # [B,k] per-particle squared error
        real_f = real_b.float()
        loss = (sqerr * real_f).sum() / real_f.sum().clamp(min=1)
        floor = ((tv_b ** 2).sum(-1) * real_f).sum() / real_f.sum().clamp(min=1)   # v=0 baseline, same mask

        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(flow.parameters(), 5.0)
        opt.step()

        loss_hist.append(float(loss)); floor_hist.append(float(floor))
        if step % a.print_every == 0 or step == a.steps:
            m_loss = float(np.mean(loss_hist[-a.print_every:]))
            m_floor = float(np.mean(floor_hist[-a.print_every:]))
            if m_loss < best:
                best = m_loss
                OUT.parent.mkdir(parents=True, exist_ok=True)
                torch.save({"state_dict": flow.state_dict(),
                            "args": {"k_max": a.k, "m_env": a.m_env, "hidden_nf": a.hidden,
                                     "n_layers": a.layers, "n_sig_bins": a.n_sig_bins, "T": a.T},
                            "step": step, "fm_loss": m_loss, "v0_floor": m_floor}, OUT)
            print(f"  step {step:>6}: FM loss {m_loss:.4f} (best {best:.4f}) | v=0 floor {m_floor:.4f} "
                  f"({time.time()-t0:.0f}s)", flush=True)
    print(f"done. best FM {best:.4f} | final v=0 floor {float(np.mean(floor_hist[-a.print_every:])):.4f} "
          f"-> {OUT}", flush=True)


if __name__ == "__main__":
    train()
