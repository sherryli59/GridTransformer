"""Task 5: conditional flow-matching TRAINING loop for the cavity block-flow corrector
(`CavityBlockFlow`, `liquid_coupling_flow/ka3d_cavity_egnn.py`). See
docs/superpowers/specs/2026-07-12-egnn-cavity-block-flow-corrector-design.md ("Training objective")
and reports/logs-2026-07-12/fm_data.py (Task 4: minibatch-OT FM data pipeline, AR-block base ->
data-block target).

FIXED-SIZE SCHEME (why -- this is the crux of this script, do not deviate)
----------------------------------------------------------------------------
`CavityCondEGNN` wraps an `EGNN_dynamics` whose particle count `n_particles` is baked in at
construction (`ka3d_cavity_egnn.py`): every forward pass MUST see exactly `P = k + n_cage`
particles. Carved cavities vary in size (radius in {1.6, 2.0, 2.5, 3.0}, chain-to-chain density
fluctuation), so this script freezes:
  - K = 8         movers   -- matches the deployment hot-rung block move BLOCK_K=8
                               (reports/logs-2026-07-12/frontier_threearm.py).
  - N_CAGE = 48   cage slots (boundary shell + retained interior), TRUNCATED to the 48 cage
                  particles NEAREST to the cavity's DATA block centroid (`cav["xo"][block_mask].mean(0)`;
                  a single fixed point per cavity -- fm_batch's `cage_x` is already identical across
                  its M AR draws, so truncating once per cavity, not per draw, is the natural and only
                  self-consistent choice). If a cavity has FEWER than 48 cage particles (small R), PAD
                  with dummies at radius 50 along a fixed direction (+x), species 0. At R<=3.0,
                  RCTX=2.5 (fm_data.py), the true cage extends to at most ~5.5 from the cavity center,
                  so a radius-50 dummy is farther from every real particle than
                  `max_neighbors`-nearest-neighbour (16) trim could ever reach -- dummies never enter
                  any real particle's k-NN neighbourhood and are therefore inert padding, not signal.
  - Cavities with `n_in < K + 3` are skipped (a little slack beyond `carve_cavity_block`'s own
    `n_in >= K + 1` floor, so the per-species OT/Hungarian pairing in `fm_batch` has room).
This is justified by k-NN locality -- `CavityCondEGNN`'s own `max_neighbors` trim only ever looks at
the nearest ~2-3 shells, so a local corrector does not need the FULL cage, only its nearest slots --
and mirrors the fixed-local-context, degree-capped size-transfer approach validated in the
ka-block-transfer memory (train-small, sample-big, zero-shot, no retrain). The flow is therefore ONE
fixed `CavityBlockFlow(k=K, n_cage=N_CAGE, max_neighbors=16)` for every cavity radius/size.

TRAINED VELOCITY == DEPLOYED VELOCITY (the other load-bearing requirement)
----------------------------------------------------------------------------
`CavityBlockFlow.flow` / `.composed_logq` are `@torch.no_grad` (inference-only) and integrate
`CavityCondEGNN.vel_div`'s velocity, which is the PAIRWISE-ONLY central-force term -- in isolated
mode, `EGNN_dynamics.forward_and_perparticle_divergence` does NOT add the `com_pot`
COM-directed term that plain `.forward()` adds (see the CAVEAT in ka3d_cavity_egnn.py's module
docstring). Training therefore calls `flow_model.ce.vel_div(cloud, t, sp, k)` DIRECTLY -- never
`flow_model.ce.egnn.forward(...)` -- so the FM loss regresses the EXACT field the deployment ODE
integrates. `vel_div` branches on `torch.is_grad_enabled()`: under an active autograd context (the
default here -- the training step is never wrapped in `torch.no_grad()`) it returns a differentiable
velocity (grad flows into `pot_model`) and, as a side effect, also computes the exact per-particle
divergence -- wasted compute during training, accepted for correctness (never let the trained field
diverge from the one the ODE integrates at deployment).

Per-sample t: `EGNN_dynamics._expand_t` natively accepts a `t` tensor of shape `[B]` (one t per
batch row), so `fm_batch`'s per-row `t[M]` is passed straight through to `vel_div` -- no python loop
over the batch is needed.

Data / split: train = the 112-chain `ka3d_train_N4096_T0.5_rho1.15.pt` (source-cavities only);
held = the 48 held-set configs (16 reference chains x 3 snapshots) in
`ka3d_dataset_N4096_T0.5_rho1.15.pt` -- CHAIN-LEVEL zero-leakage split, same as the AR retrain. AR
base: frozen `KA3DScaffoldEBMBatched`, ckpt `ka3d_cavity_ebm3ax_rho115_best.pt` (load pattern per
reports/logs-2026-07-12/mtm_early_read.py).

Usage:
  python reports/logs-2026-07-12/train_cavity_egnn_flow.py --smoke
  python reports/logs-2026-07-12/train_cavity_egnn_flow.py [--steps N ...]   # full run
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from liquid_coupling_flow.ka3d_ebm_batched import KA3DScaffoldEBMBatched
from liquid_coupling_flow.ka3d_cavity_egnn import CavityBlockFlow

from fm_data import carve_cavity_block, fm_batch

REPO_ROOT = Path(__file__).resolve().parents[2]
AR_CKPT = REPO_ROOT / "liquid_coupling_flow/artifacts/ka3d_cavity_ebm3ax_rho115_best.pt"
TRAIN_DATA = REPO_ROOT / "liquid_coupling_flow/artifacts/ka3d_train_N4096_T0.5_rho1.15.pt"
HELD_DATA = REPO_ROOT / "liquid_coupling_flow/artifacts/ka3d_dataset_N4096_T0.5_rho1.15.pt"
OUT_CKPT = REPO_ROOT / "liquid_coupling_flow/artifacts/ka3d_cavity_egnn_flow.pt"
OUT_CKPT_BEST = REPO_ROOT / "liquid_coupling_flow/artifacts/ka3d_cavity_egnn_flow_best.pt"

DUMMY_RADIUS = 50.0
MIN_SLACK = 3   # skip cavities with n_in < K + MIN_SLACK


# --------------------------------------------------------------------------------------------------
# data loading
# --------------------------------------------------------------------------------------------------
def load_ar_model(device):
    ck = torch.load(AR_CKPT, map_location=device, weights_only=False)
    ar = KA3DScaffoldEBMBatched(cat_bins=128, cat_range=2.5).to(device)
    ar.load_state_dict(ck["state_dict"], strict=False)
    ar.eval(); ar.use_frame = False
    for p in ar.parameters():
        p.requires_grad_(False)
    return ar


def load_split(path, device, expect_chains=None):
    d = torch.load(path, map_location=device, weights_only=False)
    X, S, L = d["x"].to(device).float(), d["s"].to(device).long(), float(d["L"])
    if expect_chains is not None:
        assert X.shape[0] == expect_chains, f"{path}: expected {expect_chains} configs, got {X.shape[0]}"
    return X, S, L


# --------------------------------------------------------------------------------------------------
# fixed-size cage + one-cavity batch builder (requirement 1)
# --------------------------------------------------------------------------------------------------
def carve_random_cavity(X, S, L, gen, K, radii, max_tries=50):
    """Try random (chain, center, radius) until a cavity with n_in >= K + MIN_SLACK carves; else None."""
    n_chains = X.shape[0]
    for _ in range(max_tries):
        ci = int(torch.randint(n_chains, (), generator=gen, device=X.device))
        R = float(radii[int(torch.randint(len(radii), (), generator=gen, device=X.device))])
        c = torch.rand(3, generator=gen, device=X.device) * L
        cav = carve_cavity_block(X, S, L, ci, c, R, K, gen)
        if cav is not None and cav["n"] >= K + MIN_SLACK:
            return cav
    return None


def fixed_size_cage(cage_x, sp_cage, block_centroid, n_cage, device, dtype):
    """Truncate cage_x[M,m,3]/sp_cage[M,m] to the n_cage cage slots nearest `block_centroid`[3]
    (a single per-cavity point -- cage_x is identical across the M rows by construction, see the
    module docstring), padding with far-away (radius DUMMY_RADIUS, +x direction) dummies of species 0
    if m < n_cage. Returns (cage_fix[M,n_cage,3], sp_fix[M,n_cage])."""
    M, m, _ = cage_x.shape
    if m >= n_cage:
        dist = (cage_x[0] - block_centroid).norm(dim=-1)          # [m], identical across rows -> use row 0
        idx = torch.topk(dist, n_cage, largest=False).indices
        return cage_x[:, idx], sp_cage[:, idx]
    pad_n = n_cage - m
    dummy_dir = torch.tensor([1.0, 0.0, 0.0], device=device, dtype=dtype)
    dummy_pos = (dummy_dir * DUMMY_RADIUS)[None, None, :].expand(M, pad_n, 3)
    dummy_sp = torch.zeros(M, pad_n, dtype=sp_cage.dtype, device=device)
    return torch.cat([cage_x, dummy_pos], dim=1), torch.cat([sp_cage, dummy_sp], dim=1)


def build_cavity_rows(ar_model, cav, gen, K, n_cage, M, pos_temp):
    """One cavity -> M training rows (x_t, t, target_v, sp_block, fixed-size cage/sp_cage)."""
    out = fm_batch(ar_model, cav, gen, pos_temp=pos_temp, M=M)
    assert out["n_identity_fallback"] == 0, (
        f"OT identity fallback should be impossible by construction (Task-4 carryover; species "
        f"coupling must be exact) -- got {out['n_identity_fallback']}/{M}")
    device, dtype = out["cage_x"].device, out["cage_x"].dtype
    block_centroid = cav["xo"][cav["block_mask"]].mean(0)   # DATA block centroid, fixed per cavity
    cage_fix, sp_fix = fixed_size_cage(out["cage_x"], out["sp_cage"], block_centroid, n_cage, device, dtype)
    return {"x_t": out["x_t"], "t": out["t"], "target_v": out["target_v"],
            "sp_block": out["sp_block"], "cage": cage_fix, "sp_cage": sp_fix}


def sample_training_batch(ar_model, X, S, L, gen, K, n_cage, radii, n_cav, M, pos_temp):
    """n_cav independently-carved cavities x M AR draws each, concatenated along dim 0."""
    rows = []
    for _ in range(n_cav):
        cav = carve_random_cavity(X, S, L, gen, K, radii)
        if cav is None:
            continue
        rows.append(build_cavity_rows(ar_model, cav, gen, K, n_cage, M, pos_temp))
    if not rows:
        raise RuntimeError("sample_training_batch: could not carve any valid cavity")
    return {k: torch.cat([r[k] for r in rows], dim=0) for k in rows[0]}


# --------------------------------------------------------------------------------------------------
# random SO(3) augmentation (block + cage rotated TOGETHER, per row)
# --------------------------------------------------------------------------------------------------
def random_so3(B, device, dtype, gen):
    """Batch of B proper rotation matrices [B,3,3] (QR of a Gaussian matrix -> Haar-random O(3);
    fix det to +1 by flipping the last column's sign when det==-1, the standard O(n)->SO(n) trick)."""
    A = torch.randn(B, 3, 3, generator=gen, device=device, dtype=dtype)
    Q, Rm = torch.linalg.qr(A)
    d = torch.diagonal(Rm, dim1=-2, dim2=-1).sign()
    Q = Q * d.unsqueeze(-2)
    det = torch.det(Q).sign()
    Q = Q.clone()
    Q[:, :, -1] = Q[:, :, -1] * det.unsqueeze(-1)
    return Q


def augment_so3(batch, gen):
    """Rotate x_t, target_v, cage by an INDEPENDENT random rotation per row, block+cage TOGETHER
    (isotropy: cage context must rotate with the block it conditions, or the pairwise geometry the
    EGNN reads would be wrong). x_t/target_v are both linear in (x0_block, x1_block), so rotating them
    directly is equivalent to rotating x0_block/x1_block before interpolating."""
    x_t, target_v, cage = batch["x_t"], batch["target_v"], batch["cage"]
    B = x_t.shape[0]
    Q = random_so3(B, x_t.device, x_t.dtype, gen)
    rot = lambda p: torch.einsum("bij,bkj->bki", Q, p)
    out = dict(batch)
    out["x_t"] = rot(x_t)
    out["target_v"] = rot(target_v)
    out["cage"] = rot(cage)
    return out


# --------------------------------------------------------------------------------------------------
# the FM loss (requirement 2: exactly the deployed velocity)
# --------------------------------------------------------------------------------------------------
def fm_loss(flow_model, batch, K):
    cloud = torch.cat([batch["x_t"], batch["cage"]], dim=1)
    sp = torch.cat([batch["sp_block"], batch["sp_cage"]], dim=1)
    vel, _div = flow_model.ce.vel_div(cloud, batch["t"], sp, K)   # vel_div, NOT egnn.forward() -- see docstring
    return F.mse_loss(vel, batch["target_v"])


# --------------------------------------------------------------------------------------------------
# smoke: tiny FIXED pool, must overfit
# --------------------------------------------------------------------------------------------------
def run_smoke(args):
    device = args.device
    gen = torch.Generator(device=device).manual_seed(args.seed)
    ar_model = load_ar_model(device)
    X, S, L = load_split(TRAIN_DATA, device, expect_chains=112)

    flow_model = CavityBlockFlow(n_cage=args.n_cage, k=args.k, r_c=args.r_c, hidden_nf=args.hidden_nf,
                                  n_layers=args.n_layers, n_species=2,
                                  max_neighbors=args.max_neighbors).to(device)
    opt = torch.optim.AdamW(flow_model.parameters(), lr=args.smoke_lr)

    # ---- fixed pool: build ONE batch once, train on the SAME data every step (classic overfit gate) ----
    batch = sample_training_batch(ar_model, X, S, L, gen, args.k, args.n_cage, args.radii,
                                   n_cav=args.smoke_ncav, M=args.smoke_m, pos_temp=args.pos_temp)
    n_rows = batch["x_t"].shape[0]
    print(f"[smoke] fixed pool: {args.smoke_ncav} cavities x {args.smoke_m} draws -> {n_rows} rows, "
          f"K={args.k} n_cage={args.n_cage}", flush=True)

    losses = []
    t0 = time.time()
    for step in range(1, args.smoke_steps + 1):
        loss = fm_loss(flow_model, batch, args.k)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        losses.append(float(loss.item()))
        if step == 1 or step % 25 == 0 or step == args.smoke_steps:
            print(f"[smoke] step {step:4d}  loss {losses[-1]:.6f}", flush=True)
    wall = time.time() - t0
    per_step = wall / args.smoke_steps

    first, final = losses[0], losses[-1]
    ratio = final / first
    print(f"[smoke] first={first:.6f} final={final:.6f} ratio={ratio:.4f} "
          f"wall={wall:.1f}s ({per_step*1000:.1f} ms/step)", flush=True)
    ok = final < 0.5 * first
    print(f"[smoke] {'PASS' if ok else 'FAIL'}: final {'<' if ok else '>='} 0.5*first", flush=True)
    assert ok, f"smoke FAILED: final loss {final:.6f} not < 0.5 * first loss {first:.6f}"
    return {"losses": losses, "first": first, "final": final, "ratio": ratio,
            "wall": wall, "per_step_s": per_step, "n_rows": n_rows}


# --------------------------------------------------------------------------------------------------
# held eval
# --------------------------------------------------------------------------------------------------
@torch.no_grad()
def eval_held(flow_model, ar_model, eval_cavities, gen, K, n_cage, M, pos_temp):
    flow_model.eval()
    losses, ns = [], []
    for cav in eval_cavities:
        rows = build_cavity_rows(ar_model, cav, gen, K, n_cage, M, pos_temp)
        l = fm_loss(flow_model, rows, K)
        losses.append(float(l.item()) * rows["x_t"].shape[0])
        ns.append(rows["x_t"].shape[0])
    flow_model.train()
    return sum(losses) / sum(ns)


# --------------------------------------------------------------------------------------------------
# full training loop
# --------------------------------------------------------------------------------------------------
def run_train(args):
    device = args.device
    gen = torch.Generator(device=device).manual_seed(args.seed)
    gen_eval = torch.Generator(device=device).manual_seed(args.seed + 1)

    ar_model = load_ar_model(device)
    X_tr, S_tr, L_tr = load_split(TRAIN_DATA, device, expect_chains=112)
    X_he, S_he, L_he = load_split(HELD_DATA, device)
    assert abs(L_he - L_tr) < 1e-6, f"train/held L mismatch: {L_tr} vs {L_he}"

    flow_model = CavityBlockFlow(n_cage=args.n_cage, k=args.k, r_c=args.r_c, hidden_nf=args.hidden_nf,
                                  n_layers=args.n_layers, n_species=2,
                                  max_neighbors=args.max_neighbors).to(device)
    opt = torch.optim.AdamW(flow_model.parameters(), lr=args.lr)

    # FIXED held-cavity pool (chain-level split: X_he/S_he never touch the 112 training chains) so the
    # early-stop signal isn't polluted by cavity-selection noise across evals; AR draws are still fresh
    # each eval call (build_cavity_rows re-samples the AR base every time).
    gen_pool = torch.Generator(device=device).manual_seed(args.seed + 1000)
    eval_cavities = []
    for _ in range(args.eval_n_cav):
        cav = carve_random_cavity(X_he, S_he, L_he, gen_pool, args.k, args.radii)
        if cav is not None:
            eval_cavities.append(cav)
    assert len(eval_cavities) >= max(4, args.eval_n_cav // 2), \
        f"could only carve {len(eval_cavities)}/{args.eval_n_cav} held cavities"
    print(f"[train] held eval pool: {len(eval_cavities)} fixed cavities", flush=True)

    best_held, best_step, patience_ctr = float("inf"), -1, 0
    history = []
    t0 = time.time()
    for step in range(1, args.steps + 1):
        batch = sample_training_batch(ar_model, X_tr, S_tr, L_tr, gen, args.k, args.n_cage, args.radii,
                                       n_cav=args.n_cav, M=args.m, pos_temp=args.pos_temp)
        batch = augment_so3(batch, gen)
        loss = fm_loss(flow_model, batch, args.k)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        train_loss = float(loss.item())
        row = {"step": step, "train_loss": train_loss}

        if step % args.eval_every == 0 or step == args.steps:
            held_loss = eval_held(flow_model, ar_model, eval_cavities, gen_eval, args.k, args.n_cage,
                                   args.eval_m, args.pos_temp)
            gap = held_loss - train_loss
            row.update({"held_loss": held_loss, "gap": gap})
            wall = time.time() - t0
            print(f"[train] step {step:6d}  train {train_loss:.5f}  held {held_loss:.5f}  "
                  f"gap {gap:+.5f}  wall {wall:.0f}s", flush=True)

            ckpt = {"state_dict": flow_model.state_dict(), "step": step, "held_loss": held_loss,
                    "train_loss": train_loss, "history": history + [row], "args": vars(args)}
            torch.save(ckpt, OUT_CKPT)   # last -- checkpoint incrementally, every eval
            if held_loss < best_held:
                best_held, best_step, patience_ctr = held_loss, step, 0
                torch.save(ckpt, OUT_CKPT_BEST)
            else:
                patience_ctr += 1
                if patience_ctr >= args.patience:
                    print(f"[train] EARLY STOP at step {step} (best step {best_step}, "
                          f"held {best_held:.5f})", flush=True)
                    history.append(row)
                    break
        history.append(row)
    return history


def build_argparser():
    p = argparse.ArgumentParser()
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=0)
    # fixed-size scheme
    p.add_argument("--k", type=int, default=8)
    p.add_argument("--n_cage", type=int, default=48)
    p.add_argument("--r_c", type=float, default=2.5)
    p.add_argument("--hidden_nf", type=int, default=64)
    p.add_argument("--n_layers", type=int, default=4)
    p.add_argument("--max_neighbors", type=int, default=16)
    p.add_argument("--radii", type=float, nargs="+", default=[1.6, 2.0, 2.5, 3.0])
    p.add_argument("--pos_temp", type=float, default=1.0)
    # full-run loop
    p.add_argument("--steps", type=int, default=5000)
    p.add_argument("--n_cav", type=int, default=4)
    p.add_argument("--m", type=int, default=8)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--eval_every", type=int, default=200)
    p.add_argument("--eval_n_cav", type=int, default=16)
    p.add_argument("--eval_m", type=int, default=4)
    p.add_argument("--patience", type=int, default=20)
    # smoke
    p.add_argument("--smoke_steps", type=int, default=300)
    p.add_argument("--smoke_lr", type=float, default=1e-3)
    p.add_argument("--smoke_ncav", type=int, default=8)
    p.add_argument("--smoke_m", type=int, default=2)
    p.add_argument("--out_json", type=str, default="")
    return p


def main():
    args = build_argparser().parse_args()
    if args.smoke:
        result = run_smoke(args)
        if args.out_json:
            Path(args.out_json).write_text(json.dumps(result))
    else:
        run_train(args)


if __name__ == "__main__":
    main()
