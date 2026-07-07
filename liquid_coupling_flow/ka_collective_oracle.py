"""ORACLE TEST — price the collective channel before paying for a v2 kernel.

Question: can REAL mined rearrangement events, transplanted onto a stuck SMC population and accepted on energy
alone, push past the ~-3.14/-3.17 depth floor toward the true -3.276? This deliberately IGNORES exactness
(no Hastings; a DIAGNOSTIC, never a sampler): it upper-bounds what ANY exact learned collective kernel built
on this move class could contribute. If even the oracle can't move the floor, v2 is worthless at any ceiling
(the kinetics conclusion); if it can, the prize is measured.

Arms (same stuck start, matched schedule of [attempt block + displacement relaxation]):
  real      — transplant mined (pattern, displacement) pairs: greedy species-aware matching of the event's
              mobile geometry onto the current config (random rotation/reflection + random anchor), then
              apply the event's displacements to the matched particles.
  scrambled — same matching, same displacement MAGNITUDES, directions independently randomized: controls for
              "coordinated kicks" vs real rearrangement structure.
  disp      — displacement-only at the same wall (the null).

Run: python -m liquid_coupling_flow.ka_collective_oracle {real|scrambled|disp}
"""
from __future__ import annotations
import os, sys, time, math, torch
from liquid_coupling_flow.ka_cluster_flow import ART
from liquid_coupling_flow.ka_energy import ka_energy
from liquid_coupling_flow.ka_local_smc import _disp_sweeps

DEV = "cuda" if torch.cuda.is_available() else "cpu"
TOL = 0.35                      # per-particle pattern-match tolerance
BETA = 2.0


def build_library(bank="ka_event_bank_N100.pt", device=DEV):
    """Localized events -> (r_pattern [k,2] centroid-relative, disp [k,2], species [k]) lists."""
    b = torch.load(os.path.join(ART, bank), map_location=device, weights_only=False)
    ref = torch.load(os.path.join(ART, "ka_reference_N100.pt"), map_location=device, weights_only=False)
    s_ref = ref["s"].long().to(device)
    L = (100 / 1.2) ** 0.5
    lib = []
    for beta, grp in b["harvest"].items():
        for ev in grp["events"]:
            if not (ev["single_cluster"] and ev["k"] <= 8):
                continue
            xa, xb, mob = ev["xa"].to(device), ev["xb"].to(device), ev["mobile"].to(device)
            pa = xa[mob]                                          # [k,2]
            c = pa[0:1] + _mi(pa - pa[0:1], L).mean(0, keepdim=True)   # min-image centroid (anchor at member 0)
            r = _mi(pa - c, L)                                    # [k,2] centroid-relative pattern
            d = _mi(xb[mob] - pa, L)                              # [k,2] displacements
            lib.append((r, d, s_ref[mob]))
    return lib, L


def _mi(d, L):
    return d - L * torch.round(d / L)


def _rot(v, th, refl=False):
    c, s = math.cos(th), math.sin(th)
    R = torch.tensor([[c, -s], [s, c]], device=v.device, dtype=v.dtype)
    if refl:
        R = R @ torch.tensor([[1.0, 0.0], [0.0, -1.0]], device=v.device, dtype=v.dtype)
    return v @ R.T


@torch.no_grad()
def oracle_attempt(pos, s_row, lib, L, mode, gen=None):
    """One transplant attempt on all B rows (shared event + orientation, per-row anchor).
    Returns (pos_new, fired_frac, accept_frac)."""
    B, N, _ = pos.shape; dev = pos.device
    r, d, sp = lib[int(torch.randint(0, len(lib), (1,), device=dev, generator=gen).item())]
    k = r.shape[0]
    th = float(torch.rand(1, device=dev, generator=gen).item()) * 2 * math.pi
    refl = bool(torch.randint(0, 2, (1,), device=dev, generator=gen).item())
    r = _rot(r, th, refl); d = _rot(d, th, refl)
    if mode == "scrambled":                                       # randomize directions, keep magnitudes
        ang = torch.rand(k, 1, device=dev, generator=gen) * 2 * math.pi
        mag = d.norm(dim=-1, keepdim=True)
        d = torch.cat([torch.cos(ang), torch.sin(ang)], -1) * mag
    anchor = torch.rand(B, 1, 2, device=dev, generator=gen) * L
    tgt = torch.remainder(anchor + r[None], L)                    # [B,k,2] pattern sites
    taken = torch.zeros(B, N, dtype=torch.bool, device=dev)
    idxs, ok = [], torch.ones(B, dtype=torch.bool, device=dev)
    for j in range(k):
        dist = _mi(tgt[:, j:j + 1] - pos, L).norm(dim=-1)         # [B,N]
        dist = dist.masked_fill(taken, 1e9).masked_fill(s_row[None].expand(B, N) != sp[j], 1e9)
        best = dist.argmin(1)                                     # [B]
        bd = dist.gather(1, best[:, None]).squeeze(1)
        ok = ok & (bd < TOL)
        taken[torch.arange(B, device=dev), best] = True
        idxs.append(best)
    idx = torch.stack(idxs, 1)                                    # [B,k]
    prop = pos.clone()
    rows = torch.arange(B, device=dev)[:, None]
    prop[rows, idx] = torch.remainder(prop[rows, idx] + d[None], L)
    dU = ka_energy(prop, s_row, L) - ka_energy(pos, s_row, L)
    u = torch.rand(B, device=dev, generator=gen)
    accept = ok & (torch.log(u) < -BETA * dU)
    pos = torch.where(accept[:, None, None], prop, pos)
    return pos, ok.float().mean().item(), accept.float().mean().item()


def run(mode, blocks=40, attempts_per_block=50, n_disp=20,
        stuck="smc_pilot_arm0_a2stack_N100.pt"):
    d = torch.load(os.path.join(ART, stuck), map_location=DEV, weights_only=False)
    pos, s = d["pos"].to(DEV), d["s"].to(DEV).long()
    s_row = s[0]
    lib, L = build_library()
    gen = torch.Generator(device=DEV).manual_seed(0)
    N = pos.shape[1]
    u0 = (ka_energy(pos, s_row, L) / N).mean().item()
    print(f"[oracle {mode}] stuck start <U>/N {u0:.4f}  library {len(lib)} events  B {pos.shape[0]}", flush=True)
    hist = [(0, u0)]; t0 = time.time(); fired, acc = [], []
    for blk in range(1, blocks + 1):
        if mode in ("real", "scrambled"):
            for _ in range(attempts_per_block):
                pos, f, a = oracle_attempt(pos, s_row, lib, L, mode, gen=gen)
                fired.append(f); acc.append(a)
        pos = _disp_sweeps(pos, s_row, L, kT=1.0 / BETA, n=n_disp)
        u = (ka_energy(pos, s_row, L) / N).mean().item()
        hist.append((blk, u))
        if blk % 5 == 0 or blk == 1:
            fr = sum(fired) / max(len(fired), 1); ar = sum(acc) / max(len(acc), 1)
            print(f"  block {blk:3d}: <U>/N {u:.4f}  fired {fr:.3f}  accept {ar:.4f}  "
                  f"({(time.time()-t0)/blk:.1f}s/block)", flush=True)
    print(f"[oracle {mode}] final <U>/N {hist[-1][1]:.4f}  (start {u0:.4f}, true eq -3.276)", flush=True)
    torch.save({"hist": hist, "mode": mode, "fired": fired, "acc": acc},
               os.path.join(ART, f"oracle_{mode}_N100.pt"))


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else "real")
