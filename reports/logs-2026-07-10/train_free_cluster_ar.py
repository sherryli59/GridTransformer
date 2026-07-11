"""Held-out architecture gate for the boundary-free 3D KA cavity AR model.

This deliberately removes the frozen exterior.  It asks whether the geometry,
canonical-slot representation, and autoregressive rollout work on carved bulk
clusters before boundary conditioning is reintroduced.

Primary diagnostics (all held out by bulk frame):
  * true-prefix one-step clash rate vs full-rollout energy/clash rate;
  * fraction of generated particles inside the requested spherical radius;
  * Hungarian scaffold-assignment recovery of the fixed generation labels;
  * exact sample-path logq vs a fresh preordered likelihood evaluation.

The free-cluster distribution is a diagnostic marginal, not the final cavity
target.  The final target still requires the frozen boundary and energy/SMC.
"""
from __future__ import annotations

import argparse
import math
import statistics
import time
from pathlib import Path

import torch
from liquid_coupling_flow.ka3d_cavity_ar import _mic
from liquid_coupling_flow.ka3d_cavity_carve import carve
from liquid_coupling_flow.ka3d_scaffold_ar import (
    KA3DScaffoldAR,
    assignment_recovery,
    ball_squash,
    ball_unsquash,
    label_to_scaffold,
)
from liquid_coupling_flow.ka_energy import ka_energy


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=Path,
                   default=Path("liquid_coupling_flow/artifacts/ka3d_dataset_N512_T0.5.pt"))
    p.add_argument("--artifact", type=Path,
                   default=Path("liquid_coupling_flow/artifacts/ka3d_free_scaffold_ar.pt"))
    p.add_argument("--steps", type=int, default=20_000)
    p.add_argument("--eval-every", type=int, default=1_000)
    p.add_argument("--batch", type=int, default=6)
    p.add_argument("--train-frames", type=int, default=900)
    p.add_argument("--centers-per-frame", type=int, default=2)
    p.add_argument("--eval-n", type=int, default=40)
    p.add_argument("--eval-samples", type=int, default=1)
    p.add_argument("--radii", type=float, nargs="+", default=[1.6, 2.0, 2.4])
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--clash-distance", type=float, default=0.8)
    p.add_argument("--rotation-aug", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def random_rotation(gen, device, dtype):
    a = torch.randn(3, 3, generator=gen, device=device, dtype=dtype)
    q, r = torch.linalg.qr(a)
    sign = torch.where(torch.diagonal(r) >= 0, torch.ones(3, device=device, dtype=dtype),
                       -torch.ones(3, device=device, dtype=dtype))
    q = q * sign[None]
    if torch.det(q) < 0:
        q[:, 0] = -q[:, 0]
    return q


def build_pool(X, S, L, frame_ids, centers_per_frame, radii, gen):
    pool = []
    for frame in frame_ids:
        for _ in range(centers_per_frame):
            center = torch.rand(3, generator=gen, device=X.device, dtype=X.dtype) * L
            rid = int(torch.randint(len(radii), (), generator=gen, device=X.device))
            R = float(radii[rid])
            carved = carve(X[frame], S[frame], center, R, L)
            if carved["n_in"] < 4:
                continue
            x_raw = _mic(carved["x_in"], center, L)
            s_raw = carved["s_in"]
            x, s, _ = label_to_scaffold(x_raw, s_raw, R)
            n_B = int((s == 1).sum())
            pool.append({"x": x, "s": s, "R": R, "n": int(x.shape[0]),
                         "n_A": int(x.shape[0]) - n_B, "n_B": n_B})
    return pool


def cluster_energy(x, s):
    # Huge box removes PBC/image interactions: interior-interior energy only.
    return float(ka_energy(x[None], s[None], 100.0)[0] / x.shape[0])


def clash_count(x, cutoff):
    if x.shape[0] < 2:
        return 0
    d = torch.cdist(x, x) + torch.eye(x.shape[0], device=x.device, dtype=x.dtype) * 1e9
    return int((d.min(1).values < cutoff).sum())


@torch.no_grad()
def teacher_forced_position_sample(model, x, s, R, clash_distance):
    """One draw per true prefix; report clashes only against that observed prefix.

    Combining these independently conditioned draws into one energy is not a
    valid joint sample, so this diagnostic deliberately does not do that.
    """
    empty_x, empty_s = x.new_empty(0, 3), s.new_empty(0)
    anchors, h, frame = model._contexts(x, s, empty_x, empty_s, R)
    anchor_y, _ = ball_unsquash(anchors, R)
    u, _ = model.flow.sample(h + model.sp_out_emb(s))
    y = anchor_y + torch.einsum("naj,na->nj", frame, u)
    sampled, _ = ball_squash(y, R)
    clashes = total = 0
    for j in range(x.shape[0]):
        if j:
            clashes += int(torch.cdist(sampled[j:j + 1], x[:j]).min() < clash_distance)
            total += 1
    return sampled, clashes, total


@torch.no_grad()
def evaluate(model, pairs, n_samples, clash_distance):
    model.eval()
    empty_x = pairs[0]["x"].new_empty(0, 3)
    empty_s = pairs[0]["s"].new_empty(0)
    nll = []
    true_u, fr_u = [], []
    tf_clash = tf_trials = fr_clash = fr_particles = 0
    inside = total = 0
    assignment_particle, assignment_whole = [], []
    logq_err = []
    for pair in pairs:
        x, s, R = pair["x"], pair["s"], pair["R"]
        nll.append(float(-model.log_prob_pair(x, s, empty_x, empty_s, R, preordered=True) / pair["n"]))
        true_u.append(cluster_energy(x, s))
        for _ in range(n_samples):
            _xtf, n_tf_clash, n_tf_trials = teacher_forced_position_sample(
                model, x, s, R, clash_distance)
            xfr, sfr, lq = model.sample_pair(empty_x, empty_s, pair["n_A"], pair["n_B"], R,
                                             return_logq=True)
            lq2 = model.log_prob_pair(xfr, sfr, empty_x, empty_s, R, preordered=True)
            logq_err.append(float((lq - lq2).abs()))
            fr_u.append(cluster_energy(xfr, sfr))
            tf_clash += n_tf_clash; tf_trials += n_tf_trials
            fr_clash += clash_count(xfr, clash_distance); fr_particles += xfr.shape[0]
            inside += int((xfr.norm(dim=-1) < R).sum()); total += xfr.shape[0]
            recovered, whole = assignment_recovery(xfr, R)
            assignment_particle.append(recovered); assignment_whole.append(float(whole))
    model.train()
    return {
        "held_nll": statistics.mean(nll),
        "true_u_median": statistics.median(true_u),
        "fr_u_median": statistics.median(fr_u),
        "tf_prefix_clash_frac": tf_clash / max(tf_trials, 1),
        "fr_clash_particle_frac": fr_clash / max(fr_particles, 1),
        "inside_frac": inside / max(total, 1),
        "assignment_particle_frac": statistics.mean(assignment_particle),
        "assignment_config_frac": statistics.mean(assignment_whole),
        "max_logq_error": max(logq_err, default=0.0),
    }


def print_metrics(step, metrics, elapsed):
    print(
        f"step {step:6d} held_nll={metrics['held_nll']:+.3f} "
        f"U(true/FR)={metrics['true_u_median']:+.2f}/{metrics['fr_u_median']:+.2f} "
        f"clash(TF-prefix/FR)={100*metrics['tf_prefix_clash_frac']:.1f}%/{100*metrics['fr_clash_particle_frac']:.1f}% "
        f"inside={100*metrics['inside_frac']:.1f}% "
        f"assign(particle/config)={100*metrics['assignment_particle_frac']:.1f}%/{100*metrics['assignment_config_frac']:.1f}% "
        f"logq_err={metrics['max_logq_error']:.2e} "
        f"({elapsed:.0f}s)", flush=True)


def main():
    args = parse_args()
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    gen = torch.Generator(device=device).manual_seed(args.seed + 1)
    data = torch.load(args.dataset, map_location=device, weights_only=False)
    X, S, L = data["x"].float(), data["s"].long(), float(data["L"])
    split = min(args.train_frames, X.shape[0] - 1)
    train = build_pool(X, S, L, range(split), args.centers_per_frame, args.radii, gen)
    held = build_pool(X, S, L, range(split, X.shape[0]), 1, args.radii, gen)
    if not train or not held:
        raise RuntimeError(f"empty split: train={len(train)} held={len(held)}")
    print(f"free-cluster train={len(train)} held={len(held)} radii={tuple(args.radii)} device={device}",
          flush=True)

    model = KA3DScaffoldAR().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    start, series, best = 0, [], math.inf
    if args.resume and args.artifact.exists():
        state = torch.load(args.artifact, map_location=device, weights_only=False)
        model.load_state_dict(state["state_dict"])
        opt.load_state_dict(state["optimizer"])
        start = int(state["step"]) + 1
        series = state.get("series", [])
        best = float(state.get("best_held_nll", math.inf))
        print(f"resumed step={start} best_held_nll={best:+.3f}", flush=True)

    empty_x = X.new_empty(0, 3); empty_s = S.new_empty(0)
    args.artifact.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    for step in range(start, args.steps + 1):
        ids = torch.randint(len(train), (args.batch,), generator=gen, device=device)
        loss = X.new_zeros(())
        for raw_i in ids:
            pair = train[int(raw_i)]
            x, s = pair["x"], pair["s"]
            if args.rotation_aug:
                x = x @ random_rotation(gen, device, x.dtype).T
                x, s, _ = label_to_scaffold(x, s, pair["R"])
            loss = loss - model.log_prob_pair(x, s, empty_x, empty_s, pair["R"], preordered=True) / pair["n"]
        loss = loss / args.batch
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()

        if step % args.eval_every == 0 or step == args.steps:
            metrics = evaluate(model, held[:args.eval_n], args.eval_samples, args.clash_distance)
            metrics["step"] = step; metrics["train_loss"] = float(loss.detach())
            series.append(metrics)
            print_metrics(step, metrics, time.time() - t0)
            state = {
                "state_dict": model.state_dict(), "optimizer": opt.state_dict(), "step": step,
                "series": series, "best_held_nll": min(best, metrics["held_nll"]),
                "config": vars(args), "dataset": str(args.dataset),
            }
            torch.save(state, args.artifact)
            if metrics["held_nll"] < best:
                best = metrics["held_nll"]
                torch.save(state, args.artifact.with_name(args.artifact.stem + "_best.pt"))

    final = series[-1]
    gates = {
        "exact_logq": final["max_logq_error"] < 1e-3,
        "inside_ball": final["inside_frac"] >= 0.99,
        "assignment_recovery": final["assignment_particle_frac"] >= 0.80,
        "tf_clean": final["tf_prefix_clash_frac"] <= 0.02,
        "fr_clean": final["fr_clash_particle_frac"] <= 0.05,
    }
    print(f"FINAL GATES {gates} -> {'PASS' if all(gates.values()) else 'FAIL'}", flush=True)


if __name__ == "__main__":
    main()
