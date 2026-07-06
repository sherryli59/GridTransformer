"""Stage-B GB0: mine collective relaxation EVENTS from the PT-ladder trajectories (pt_ladder_N{N}.pt).

Ladder layout (ka_reference.main_ladder): per rung, configs indexed t*n_rep + b within each seed block
(t = snapshot time, 8 sweeps apart; b = replica). Adjacent-time same-replica pairs are 8 MC-sweeps of dynamics —
EXCEPT when a PT exchange replaced the config (every 10 sweeps, cold acc ~0.5): those pairs differ globally and
must be REJECTED, not read as physics (the data subtlety this module exists to handle).

Classification of a pair (min-image displacements d_i):
  exchange : frac(d_i > d_mobile) > frac_exchange   (whole-config replacement)
  event    : max d_i >= d_event                      (cage-break); mobile set = {i: d_i > d_mobile}
  quiet    : everything else (vibration only)

Spec: docs/superpowers/specs/2026-07-06-stage-b-collective-move-design.md (GB0 gate).
"""
from __future__ import annotations
import os, torch

ART = os.path.join(os.path.dirname(__file__), "artifacts")
D_MOBILE = 0.35          # > vibration (~0.1-0.15), < hop (~ nn-dist 0.95)
D_EVENT = 0.6            # cage-break scale
FRAC_EXCHANGE = 0.25     # more than a quarter of particles mobile => config replacement, not dynamics
R_CLUSTER = 1.4          # connectivity radius for single-cluster check (~1.5 first-shell)


def pair_index(n_seeds=2, nc=500, n_rep=8):
    """Adjacent-time same-replica pair indices into a per-rung config stack laid out as
    [seed0: t0 r0..r{n_rep-1}, t1 ..., ..., seed1: ...]. Never pairs across the seed boundary.
    Returns (ia, ib) LongTensors."""
    ia, ib = [], []
    block = nc * n_rep
    for s in range(n_seeds):
        for t in range(nc - 1):
            for b in range(n_rep):
                ia.append(s * block + t * n_rep + b)
                ib.append(s * block + (t + 1) * n_rep + b)
    return torch.tensor(ia, dtype=torch.long), torch.tensor(ib, dtype=torch.long)


def _min_image(d, L):
    return d - L * torch.round(d / L)


def classify_pair(xa, xb, L, d_mobile=D_MOBILE, d_event=D_EVENT, frac_exchange=FRAC_EXCHANGE):
    """xa, xb [N,2] -> dict(kind, mobile, k, extent, single_cluster). Min-image throughout."""
    d = _min_image(xb - xa, L).norm(dim=-1)                       # [N]
    mobile = (d > d_mobile).nonzero().squeeze(-1)
    if mobile.numel() > frac_exchange * d.shape[0]:
        return {"kind": "exchange", "mobile": mobile, "k": int(mobile.numel()),
                "extent": float("nan"), "single_cluster": False}
    if float(d.max()) < d_event:
        return {"kind": "quiet", "mobile": torch.empty(0, dtype=torch.long), "k": 0,
                "extent": 0.0, "single_cluster": True}
    # localized event: extent + connectivity of the mobile set (at time a)
    pos = xa[mobile]                                              # [k,2]
    dd = _min_image(pos[:, None, :] - pos[None, :, :], L).norm(dim=-1)   # [k,k]
    extent = float(dd.max()) if mobile.numel() > 1 else 0.0
    # single cluster? union of balls radius R_CLUSTER: BFS on the adjacency
    k = mobile.numel()
    adj = dd < R_CLUSTER
    seen = torch.zeros(k, dtype=torch.bool); frontier = [0]; seen[0] = True
    while frontier:
        i = frontier.pop()
        for j in adj[i].nonzero().squeeze(-1).tolist():
            if not seen[j]:
                seen[j] = True; frontier.append(j)
    return {"kind": "event", "mobile": mobile, "k": int(k), "extent": extent,
            "single_cluster": bool(seen.all())}


def mine_ladder(fname, rungs=None, max_pairs=None, d_mobile=D_MOBILE, d_event=D_EVENT):
    """Mine events per rung from a ladder artifact. Returns {rung: stats}, stats = dict(n_pairs, n_exchange,
    n_quiet, n_event, events=[{k, extent, single_cluster, species(A,B), rung, beta, ia, ib}, ...])."""
    lad = torch.load(os.path.join(ART, fname), map_location="cpu", weights_only=False)
    L, betas, s = lad["L"], lad["betas"], lad["s"].long()
    n_rep = 8
    out = {}
    for l in (rungs if rungs is not None else range(len(betas))):
        cfg = lad["configs_per_rung"][l]                          # [n, N, 2]
        n = cfg.shape[0]
        nc = n // (2 * n_rep)                                     # per-seed snapshot count
        ia, ib = pair_index(n_seeds=2, nc=nc, n_rep=n_rep)
        if max_pairs is not None:
            ia, ib = ia[:max_pairs], ib[:max_pairs]
        stats = {"n_pairs": len(ia), "n_exchange": 0, "n_quiet": 0, "n_event": 0, "events": [],
                 "beta": betas[l]}
        for a, b in zip(ia.tolist(), ib.tolist()):
            r = classify_pair(cfg[a], cfg[b], L, d_mobile=d_mobile, d_event=d_event)
            if r["kind"] == "exchange":
                stats["n_exchange"] += 1
            elif r["kind"] == "quiet":
                stats["n_quiet"] += 1
            else:
                stats["n_event"] += 1
                sp = s[r["mobile"]]
                stats["events"].append({"k": r["k"], "extent": r["extent"],
                                        "single_cluster": r["single_cluster"],
                                        "nA": int((sp == 0).sum()), "nB": int((sp == 1).sum()),
                                        "rung": l, "beta": betas[l], "ia": a, "ib": b})
        out[l] = stats
    return out


def report(fname="pt_ladder_N100.pt", max_pairs=None):
    """GB0 gate report: per-rung counts + cold-rung event morphology."""
    stats = mine_ladder(fname, max_pairs=max_pairs)
    tot_cold = 0
    for l, st in stats.items():
        ev = st["events"]
        ks = torch.tensor([e["k"] for e in ev], dtype=torch.float) if ev else torch.zeros(0)
        loc = sum(1 for e in ev if e["single_cluster"] and e["k"] <= 10)
        if l <= 1:
            tot_cold += loc
        print(f"rung {l} (beta {st['beta']:.2f}): pairs {st['n_pairs']}  exch {st['n_exchange']}  "
              f"quiet {st['n_quiet']}  events {st['n_event']}  localized(k<=10,1-cluster) {loc}  "
              f"k median {ks.median().item() if len(ks) else 0:.0f} max {ks.max().item() if len(ks) else 0:.0f}",
              flush=True)
    print(f"GB0 GATE: cold-rung (beta 2.0+1.81) localized events = {tot_cold}  "
          f"-> {'GO (>=200)' if tot_cold >= 200 else 'FALLBACK: swap-MC harvest above T*'}", flush=True)
    return stats


if __name__ == "__main__":
    import sys
    report(fname=sys.argv[1] if len(sys.argv) > 1 else "pt_ladder_N100.pt")
