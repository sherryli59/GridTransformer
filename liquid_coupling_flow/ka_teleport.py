"""TELEPORT-SWAP kernel — the distilled core of variable-membership blocks (spec
2026-07-07-teleport-and-alchemical-design.md), built entirely from A2.

Move: two slots a, b with NON-OVERLAPPING cages exchange occupants — each arriving particle is placed by A2's
beta-conditioned frozen-cage conditional in the other's neighborhood. This is the CORRECT use of the frozen-cage
density (each destination conditional models exactly "the occupant of this slot given everyone else", and after
the swap the arrival IS the occupant). It is swap-and-breathe's Hastings structure at LONG RANGE: nonlocal
membership change (the move class no local kernel expresses) + a long-range species teleport when species differ.

Exactness:
- selection: uniform slot pair, deterministic symmetric overlap-abort (cages exclude both slots => all four
  densities are computed on the SAME frozen environment in both directions);
- alpha = min(1, e^{-beta dU} * [q(xa_old|cage_a,sp_a) q(xb_old|cage_b,sp_b)] / [q(xa_new|cage_b,sp_a) q(xb_new|cage_a,sp_b)]);
- dU exact via the two-mover cluster energy marginal (canonical ka_energy differencing).

Both collective-move walls vanish by construction: selection is geometric/symmetric; k=1 per placement (no
compounding). The open question is PHYSICS (pocket availability) — priced by price().
"""
from __future__ import annotations
import os, sys, time, torch
from liquid_coupling_flow import ka_cluster as KC
from liquid_coupling_flow.ka_cluster_csmc import cluster_energy_total
from liquid_coupling_flow.ka_cluster_flow import slot_order, _scaffold, ART

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def cages_overlap(a, b, sc, L, n_ctx):
    """True if slot a is in slot b's cage or vice versa (or a==b). Deterministic in (a,b) => symmetric abort."""
    if a == b:
        return True
    ctx_a = KC.frame_ctx_slots(torch.tensor([a], device=sc.device), sc, L, n_ctx)
    ctx_b = KC.frame_ctx_slots(torch.tensor([b], device=sc.device), sc, L, n_ctx)
    return bool((ctx_a == b).any() or (ctx_b == a).any())


def pair_logq(HB, pos, s, a, b, xq_a_dest_b, xq_b_dest_a, sc, L, beta):
    """log q(xq_a | cage_b, sp_a) + log q(xq_b | cage_a, sp_b): the two-placement density with each query
    scored at its DESTINATION slot under the MOVER's species. Cages exclude both movers (overlap pre-filtered),
    so this is well-defined on the frozen rest-state regardless of where the movers currently sit."""
    B = pos.shape[0]
    s_a_at_b = s.clone(); s_a_at_b[:, b] = s[:, a]                   # species of the arrival at slot b
    s_b_at_a = s.clone(); s_b_at_a[:, a] = s[:, b]
    lq_a = HB.log_q_site(pos, s_a_at_b, b, xq_a_dest_b, sc, L, beta)
    lq_b = HB.log_q_site(pos, s_b_at_a, a, xq_b_dest_a, sc, L, beta)
    return lq_a + lq_b


@torch.no_grad()
def teleport_swap_move(HB, pos, s, a, b, sc, L, beta=2.0, gen=None):
    """One teleport-swap between slots a and b (pos/s slot-ordered; caller pre-filters overlap or accepts the
    inline abort). Returns (pos_new, s_new, info)."""
    B, N, _ = pos.shape; dev = pos.device
    beta_t = beta if torch.is_tensor(beta) else torch.full((B,), float(beta), device=dev)
    if cages_overlap(a, b, sc, L, HB.n_ctx):
        return pos, s, {"accept": 0.0, "abort": True, "accept_mask": torch.zeros(B, dtype=torch.bool, device=dev)}
    # forward placements: particle-from-a into b's cage (species sp_a), particle-from-b into a's cage
    s_a_at_b = s.clone(); s_a_at_b[:, b] = s[:, a]
    s_b_at_a = s.clone(); s_b_at_a[:, a] = s[:, b]
    x_a_new, lq_f_a = HB.sample_site(pos, s_a_at_b, b, sc, L, beta_t)     # [B,2] new position near b
    x_b_new, lq_f_b = HB.sample_site(pos, s_b_at_a, a, sc, L, beta_t)
    # reverse densities: the old positions scored at their own (source) slots under their own species
    lq_r = pair_logq(HB, pos, s, b, a, pos[:, b], pos[:, a], sc, L, beta_t)   # q(xb_old|cage_b,sp_b)+q(xa_old|cage_a,sp_a)
    lq_f = lq_f_a + lq_f_b
    # exact two-mover energy marginal (species swapped in the new state)
    cl = torch.tensor([a, b], device=dev)
    s_new = s.clone(); s_new[:, a] = s[:, b]; s_new[:, b] = s[:, a]
    xC_old = pos[:, cl]
    xC_new = torch.stack([x_b_new, x_a_new], 1)                      # slot a <- particle-from-b, slot b <- from-a
    U_old = cluster_energy_total(xC_old, pos, s, cl, L)
    U_new = cluster_energy_total(xC_new, pos, s_new, cl, L)
    log_alpha = -beta_t * (U_new - U_old) + lq_r - lq_f
    u = torch.rand(B, device=dev, generator=gen)
    accept = torch.log(u) < log_alpha
    pos_out = pos.clone(); s_out = s.clone()
    pos_out[:, cl] = torch.where(accept[:, None, None], xC_new, xC_old)
    s_out[:, cl] = torch.where(accept[:, None], s_new[:, cl], s[:, cl])
    return pos_out, s_out, {"accept": accept.float().mean().item(), "abort": False, "accept_mask": accept,
                            "dU_mean": float((U_new - U_old).mean())}


@torch.no_grad()
def teleport_sweep(HB, pos, s, sc, L, geo, beta=2.0, n_moves=None, gen=None):
    """pi_beta-invariant teleport sweep (lab-frame state: re-slot-order per move, scatter back both pos+species)."""
    B, N, _ = pos.shape; dev = pos.device
    n_moves = N if n_moves is None else n_moves
    accs, aborts = [], 0
    for _ in range(n_moves):
        ab = torch.randint(0, N, (2,), device=dev, generator=gen)
        a, b = int(ab[0]), int(ab[1])
        order = geo._curve_order(pos, N)
        pos_o = torch.gather(pos, 1, order[..., None].expand(-1, -1, 2))
        s_o = torch.gather(s, 1, order)
        pos_n, s_n, info = teleport_swap_move(HB, pos_o, s_o, a, b, sc, L, beta=beta, gen=gen)
        if info["abort"]:
            aborts += 1; continue
        idx = order[:, [a, b]]
        pos = pos.clone(); s = s.clone()
        pos[torch.arange(B, device=dev)[:, None], idx] = pos_n[:, [a, b]]
        s[torch.arange(B, device=dev)[:, None], idx] = s_n[:, [a, b]]
        accs.append(info["accept"])
    return pos, s, {"accept": sum(accs) / max(len(accs), 1), "abort_frac": aborts / n_moves}


def price(T=5, B=128):
    """THE PRICING GATE: teleport acceptance at (i) equilibrium cold rung, (ii) the stuck A2-stack population,
    (iii) mid-anneal beta=1.2 rung. Verdict: >0.5% anywhere interesting => wire into the stack; ~0 => the
    variable-membership direction closes at the pricing gate."""
    from liquid_coupling_flow.ka_heatbath import _load
    sc, L, geo = _scaffold(100, DEV)
    HB = _load(torch.load(os.path.join(ART, "ka_heatbath_N100.pt"), map_location=DEV, weights_only=False), DEV)
    lad = torch.load(os.path.join(ART, "pt_ladder_N100.pt"), map_location=DEV, weights_only=False)
    gen = torch.Generator(device=DEV).manual_seed(0)
    regimes = {"equilibrium(b=2)": (lad["configs_per_rung"][0][:B], lad["s"], 2.0),
               "mid-anneal(b=1.2)": (lad["configs_per_rung"][5][:B], lad["s"], 1.2)}
    stuck = os.path.join(ART, "smc_pilot_arm0_a2stack_N100.pt")
    if os.path.exists(stuck):
        d = torch.load(stuck, map_location=DEV, weights_only=False)
        regimes["SMC-stuck(b=2)"] = (d["pos"][:B], d["s"][0], 2.0)
    out = {}
    for name, (x, s0, beta) in regimes.items():
        pos, s = slot_order(x.to(DEV), s0.to(DEV).long(), geo, 100)
        accs, t0 = [], time.time()
        for _ in range(T):
            pos, s, info = teleport_sweep(HB, pos, s, sc, L, geo, beta=beta, gen=gen)
            accs.append(info["accept"])
        out[name] = sum(accs) / len(accs)
        print(f"TELEPORT {name}: accept {out[name]:.5f}  abort {info['abort_frac']:.2f}  "
              f"({(time.time()-t0)/T:.1f}s/sweep)", flush=True)
    verdict = "WIRE INTO STACK" if max(out.values()) > 0.005 else "CLOSES at pricing gate"
    print(f"TELEPORT PRICE: {out} -> {verdict}", flush=True)
    torch.save(out, os.path.join(ART, "teleport_price_N100.pt"))


if __name__ == "__main__":
    price()
