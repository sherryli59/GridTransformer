"""Kernel-design decider: on REAL bank events, compare block-selection mechanisms for reversibility.
 (a) slot-anchored (current kernel): occupants of cluster_slots stay identical -> measured abort ~93% live.
 (b) particle-anchored kNN: block S = seed particle + 7 nearest (in xa). Reverse-selectable if in xb some
     particle i in S has kNN-block(i; xb) = S. Fold q_sel counts into Hastings; ratio=0 iff count=0.
Report: fraction of events reverse-selectable under (b) + the forward/reverse seed-count distribution."""
import torch, os
from liquid_coupling_flow.ka_cluster_flow import ART

bank = torch.load(os.path.join(ART, "ka_event_bank_N100_300k.pt"), map_location="cpu", weights_only=False)
L = (100 / 1.2) ** 0.5
K = 8

def knn_block(x, i, K):
    d = x - x[i]; d = d - L * torch.round(d / L)
    dist = d.norm(dim=-1); dist[i] = -1.0            # seed first
    return set(dist.topk(K, largest=False).indices.tolist())

n_ev, n_rev_ok, fwd_counts, rev_counts = 0, 0, [], []
for b, grp in bank["harvest"].items():
    for ev in grp["events"]:
        if not (ev["single_cluster"] and ev["k"] <= 8):
            continue
        xa, xb, mob = ev["xa"], ev["xb"], ev["mobile"]
        # forward block: seed = max-displacement mobile particle; S = its kNN-8 block in xa; require mobile coverage
        d = xb - xa; d = (d - L * torch.round(d / L)).norm(dim=-1)
        S = None
        for i in sorted(mob.tolist(), key=lambda j: -float(d[j])):
            cand = knn_block(xa, i, K)
            if set(mob.tolist()) <= cand:
                S = cand; break
        if S is None:
            continue
        n_ev += 1
        fwd = sum(1 for i in S if knn_block(xa, i, K) == S)
        rev = sum(1 for i in S if knn_block(xb, i, K) == S)
        fwd_counts.append(fwd); rev_counts.append(rev)
        n_rev_ok += (rev > 0)
print(f"events considered {n_ev}; particle-kNN REVERSE-SELECTABLE: {n_rev_ok} ({100*n_rev_ok/max(n_ev,1):.0f}%)")
import statistics as st
print(f"fwd seed-count median {st.median(fwd_counts):.0f}  rev median {st.median(rev_counts):.0f}  "
      f"(counts enter the Hastings ratio as q_sel — nonzero rev = viable)")
