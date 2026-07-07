"""Region-anchored kernel viability: block = occupants of ball(c, R) with c a FIXED spatial anchor
(scaffold site near the event). Symmetric rule both directions; abort iff occupant set changes.
Measure on real events: P(occupancy preserved) and P(movers all inside) vs R. Also block-size stats."""
import torch, os
from liquid_coupling_flow.ka_cluster_flow import ART

bank = torch.load(os.path.join(ART, "ka_event_bank_N100_300k.pt"), map_location="cpu", weights_only=False)
L = (100 / 1.2) ** 0.5

def occ(x, c, R):
    d = x - c; d = d - L * torch.round(d / L)
    return set((d.norm(dim=-1) < R).nonzero().squeeze(-1).tolist())

for R in (1.6, 2.0, 2.4, 2.8):
    n, cov, stab, both, sizes = 0, 0, 0, 0, []
    for b, grp in bank["harvest"].items():
        for ev in grp["events"]:
            if not (ev["single_cluster"] and ev["k"] <= 8):
                continue
            xa, xb, mob = ev["xa"], ev["xb"], ev["mobile"]
            dm = xa[mob] - xa[mob].mean(0, keepdim=True)
            c = xa[mob].mean(0)                                  # anchor ~ mobile centroid (proxy for nearest scaffold site)
            n += 1
            Sa, Sb = occ(xa, c, R), occ(xb, c, R)
            covered = set(mob.tolist()) <= Sa
            stable = Sa == Sb
            cov += covered; stab += stable; both += (covered and stable)
            if covered and stable: sizes.append(len(Sa))
    import statistics as st
    med = st.median(sizes) if sizes else 0
    print(f"R={R}: events {n}  movers-covered {100*cov/n:.0f}%  occupancy-stable {100*stab/n:.0f}%  "
          f"BOTH {100*both/n:.0f}%  usable-block-size median {med} max {max(sizes) if sizes else 0}")
