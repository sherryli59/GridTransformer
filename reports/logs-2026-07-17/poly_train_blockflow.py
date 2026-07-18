"""FM trainer for PolyBlockFlow (liquid_coupling_flow/poly/block_flow.py) on Task-3 block pairs
(reports/logs-2026-07-17/poly_block_data.py's make_block_pair output, pickled as a list of dicts by
poly_block_data.py's __main__ or the --pairs_file this script is pointed at).

FM setup: x0 = x_old + BASE_W*randn (short-transport base, SAME prior propose() samples from), x1 = x_new,
t ~ U(0,1), x_t = (1-t)*x0 + t*x1, target v = x1 - x0. Mover species = sigma_to_bin(sig_new) (the species
identity AFTER the block's permutation -- what the relaxed x_new is conditioned on); env species =
sigma_to_bin(env_sig). Loss regresses flow.ce.vel_div directly -- the EXACT field the RK4 integrator in
block_flow.py's propose/logq_of uses, never a separate forward -- masked-MSE over real (non-dummy) mover
rows, matching reports/logs-2026-07-15/train_cavity_ersi.py's training convention.

`--diversity_check`: loads a checkpoint and runs the Ciarella mode-collapse diagnostic on ONE pair's
block -- per-particle position std across 64 flow proposals vs across 64 FRESH `_block_relax`
re-relaxations of the identical (positions, permuted species) start state. Healthy ratio in [0.5, 2].

Usage:
    poly_train_blockflow.py --T 0.2 --k 8 --steps 300 --pairs_file <path> [--out PATH] [--m_env 64] ...
    poly_train_blockflow.py --diversity_check --ckpt <path> --pairs_file <path>
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

from liquid_coupling_flow.poly.block_flow import PolyBlockFlow, sigma_to_bin, BASE_W
from liquid_coupling_flow.poly.model import seed_numba
from poly_block_data import _block_relax, RELAX_SW  # noqa: E402  (needs sys.path insert above)

p = argparse.ArgumentParser()
p.add_argument("--T", type=float, required=True)
p.add_argument("--k", type=int, required=True)
p.add_argument("--steps", type=int, default=20000)
p.add_argument("--pairs_file", type=str, required=True)
p.add_argument("--out", type=str, default=None)
p.add_argument("--m_env", type=int, default=64)
p.add_argument("--hidden", type=int, default=128)
p.add_argument("--layers", type=int, default=4)
p.add_argument("--lr", type=float, default=3e-4)
p.add_argument("--batch", type=int, default=32)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--print_every", type=int, default=100)
p.add_argument("--diversity_check", action="store_true")
p.add_argument("--ckpt", type=str, default=None, help="checkpoint to load for --diversity_check")
p.add_argument("--n_samples", type=int, default=64, help="draws per side for --diversity_check")
a = p.parse_args()

OUT = Path(a.out) if a.out else REPO / f"liquid_coupling_flow/artifacts/poly_blockflow_T{a.T}_k{a.k}_best.pt"
dev = "cuda" if torch.cuda.is_available() else "cpu"


def load_pairs(path):
    pairs = torch.load(path, weights_only=False)
    assert len(pairs) > 0, f"empty pairs file {path}"
    assert pairs[0]["x_old"].shape[0] == a.k, (
        f"pairs file block size {pairs[0]['x_old'].shape[0]} != --k {a.k}")
    return pairs


def build_batch(pairs, idxs, flow, gen):
    """One FM minibatch: x_t (interpolant), target_v, species, cage -- all fixed-size padded/truncated by
    PolyBlockFlow's own helpers so a single batched flow.ce.vel_div call covers the whole batch. Also
    returns `real_b[B,k_max]` bool: True on the first n_real mover slots of each row (n_real == x_old's
    block size, <= k_max; make_block_pair currently always emits exactly k_max rows so this is all-True
    in practice, but the mask is required by spec point 3 (train_cavity_ersi.py's masked-MSE convention)
    and costs nothing -- it also makes the trainer correct the moment a variable-k pairs file shows up."""
    k = flow.k_max
    xt_rows, sp_rows, cage_rows, spc_rows, tv_rows, t_rows, real_rows = [], [], [], [], [], [], []
    for i in idxs:
        pr = pairs[i]
        x_old = torch.as_tensor(pr["x_old"], dtype=torch.float32)
        # WRAP FIX (2026-07-18): min-image the DIFFERENCE (see poly_train_swapflow.py note)
        _L = float(pr["L"])
        _d = torch.as_tensor(pr["x_new"], dtype=torch.float32) - x_old
        _d = _d - _L * torch.round(_d / _L)
        x_new = x_old + _d
        sig_new_bin = torch.as_tensor(sigma_to_bin(pr["sig_new"]), dtype=torch.long)
        env_x = torch.as_tensor(pr["env_x"], dtype=torch.float32)
        env_bin = torch.as_tensor(sigma_to_bin(pr["env_sig"]), dtype=torch.long)

        n = x_old.shape[0]
        noise = torch.randn(n, 3, generator=gen)
        x0 = x_old + BASE_W * noise
        x1 = x_new
        t = torch.rand((), generator=gen).item()
        x_t = (1.0 - t) * x0 + t * x1
        target_v = x1 - x0

        movers_full, sp_movers, n_real = flow._prep_movers(x_t, sig_new_bin)
        cage_full, sp_cage = flow._prep_env(env_x, env_bin)
        real = torch.zeros(k, dtype=torch.bool); real[:n_real] = True
        tv_full = torch.zeros(k, 3); tv_full[:n_real] = target_v[:n_real]

        xt_rows.append(movers_full); sp_rows.append(sp_movers)
        cage_rows.append(cage_full); spc_rows.append(sp_cage)
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
    print(f"poly block-flow FM training: T={a.T} k={a.k} m_env={a.m_env} hidden={a.hidden} "
          f"layers={a.layers} | {len(pairs)} pairs | steps={a.steps} batch={a.batch} | device={dev}",
          flush=True)

    flow = PolyBlockFlow(k_max=a.k, m_env=a.m_env, hidden_nf=a.hidden, n_layers=a.layers).to(dev)
    opt = torch.optim.Adam(flow.parameters(), lr=a.lr)
    gen = torch.Generator().manual_seed(a.seed)
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
        # masked-MSE over REAL (non-dummy) mover rows only, matching train_cavity_ersi.py's convention --
        # denominator is the real-mover count, not B*k, so dummy rows (target_v==0 by construction, see
        # _prep_movers) never dilute the loss/floor. All-real in the current fixed-k pairs files (real_b
        # all True) so this is a no-op today, but required the moment k < k_max pairs appear.
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
                                     "n_layers": a.layers, "T": a.T},
                            "step": step, "fm_loss": m_loss, "v0_floor": m_floor}, OUT)
            print(f"  step {step:>6}: FM loss {m_loss:.4f} (best {best:.4f}) | v=0 floor {m_floor:.4f} "
                  f"({time.time()-t0:.0f}s)", flush=True)
    print(f"done. best FM {best:.4f} | final v=0 floor {float(np.mean(floor_hist[-a.print_every:])):.4f} "
          f"-> {OUT}", flush=True)


def _load_flow_for_diversity():
    ckpt_path = a.ckpt or str(OUT)
    ck = torch.load(ckpt_path, map_location=dev, weights_only=False)
    ar = ck["args"]
    flow = PolyBlockFlow(k_max=ar["k_max"], m_env=ar["m_env"], hidden_nf=ar["hidden_nf"],
                          n_layers=ar["n_layers"]).to(dev)
    flow.load_state_dict(ck["state_dict"])
    flow.eval()
    print(f"loaded {ckpt_path} (step {ck.get('step')}, fm_loss {ck.get('fm_loss'):.4f})", flush=True)
    return flow


def diversity_check():
    pairs = load_pairs(a.pairs_file)
    flow = _load_flow_for_diversity()
    pr = pairs[0]
    k = pr["x_old"].shape[0]
    x_old = pr["x_old"]; env_x = pr["env_x"]; env_sig = pr["env_sig"]
    sig_new = pr["sig_new"]; L = float(pr["L"]); beta = float(pr["beta"])
    sig_new_bin = sigma_to_bin(sig_new)
    env_bin = sigma_to_bin(env_sig)

    # --- 64 flow proposals ---
    gen = torch.Generator().manual_seed(12345)
    flow_samples = np.stack([flow.propose(x_old, sig_new_bin, env_x, env_bin, gen)[0]
                              for _ in range(a.n_samples)])             # [M,k,3]

    # --- 64 fresh _block_relax re-relaxations of the SAME (positions, permuted species) start state ---
    x_full = np.concatenate([x_old, env_x], axis=0).astype(np.float64)
    sig_full = np.concatenate([sig_new, env_sig], axis=0).astype(np.float64)
    idx = np.arange(k, dtype=np.int64)
    relax_samples = np.zeros((a.n_samples, k, 3))
    for m in range(a.n_samples):
        xw = x_full.copy(); sw = sig_full.copy()
        seed_numba(20000 + m)
        _block_relax(xw, sw, L, beta, idx, RELAX_SW, 0.1)
        relax_samples[m] = xw[idx]

    def per_particle_std(samples):
        # samples [M,k,3] -> scalar: mean over particles of sqrt(sum_dim var)
        v = samples.var(axis=0)                     # [k,3]
        return float(np.sqrt(v.sum(-1)).mean())

    flow_std = per_particle_std(flow_samples)
    relax_std = per_particle_std(relax_samples)
    ratio = flow_std / relax_std if relax_std > 0 else float("inf")
    print(f"diversity check: flow std {flow_std:.4f} | block-relax std {relax_std:.4f} | "
          f"ratio {ratio:.4f} | healthy range [0.5, 2] -> {'PASS' if 0.5 <= ratio <= 2.0 else 'FAIL'}",
          flush=True)
    return flow_std, relax_std, ratio


if __name__ == "__main__":
    if a.diversity_check:
        diversity_check()
    else:
        train()
