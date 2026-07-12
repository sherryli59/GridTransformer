"""G_PTS(R) on the N=4096 big box (T=0.5, rho=1.1486, L=15.278) — PER-CAVITY PARALLEL TEMPERING (v3).

WHY PT: single-site heat-bath (any init: uniform, alien-equilibrium, beta-ramp burn) leaves a +0.4-1.3/part
collectively-strained tail that decays too slowly for the estimator's ~0.1/part bar (measured: gpts_bigbox
runs 1-3; the strain is clash-free = wrong packing, needs collective relaxation). This is exactly why
Berthier-Charbonneau-Yaida 2016 used PT over temperature per cavity; PT-for-references is also this repo's
own standard (2D ka_reference.py). The learned-sampler science comes AFTER this reference curve exists.

Protocol per (R, cavity): TWO independent PT stacks (stack A seeded from the DATA interior, stack B from an
ALIEN equilibrated config's interior at the same center), each NRUNG geometric beta rungs 0.8 -> 2.0.
Each sweep = per-rung oracle heat-bath (per-rung beta) + adjacent-rung Metropolis exchange (alternating
parity). Cold-rung (beta=2) configs collected over the second half of the run.

Estimator = paper core overlap qc between COLD samples of stack A x stack B (independent by construction
when converged). Convergence certificate: (1) cold-rung energy gap |U_A - U_B|/n < 0.15; (2) crossAB ~
within-stack qc. Non-converged flagged, never silently pooled. Full configs/traces/exchange rates saved."""
import time
from pathlib import Path
import torch

from liquid_coupling_flow.ka3d_cavity_ar import _mic
from liquid_coupling_flow.ka3d_cavity_carve import carve
from liquid_coupling_flow.ka_energy import ka_energy, SIGMA, EPS, RCUT_FACTOR

DEV = "cuda"; T = 0.5; BETA = 1.0 / T; RCTX = 2.5; BIGL = 100.0
BETA_HOT = 0.4                               # hot rung ~ near-ideal liquid: guarantees renewal; the ladder
                                             # descent then performs repeated annealing attempts (round trips)
NCAND = 192; TEL_FRAC = 0.5; SIGLOC = 0.2
RGRID = [1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
NRUNGS = {1.5: 12, 2.0: 12, 2.5: 12, 3.0: 16, 3.5: 16, 4.0: 16}   # more rungs at big n (exchange overlap ~ 1/sqrt(n))
SWEEPS = {1.5: 200, 2.0: 200, 2.5: 200, 3.0: 240, 3.5: 240, 4.0: 280}
COLD_EVERY = 15                              # collect cold-rung sample every this many sweeps, 2nd half only
NCAV = 6
OUT = Path("liquid_coupling_flow/artifacts/gpts_bigbox_pt_N4096.pt")
T_SIG = torch.tensor(SIGMA, device=DEV); T_EPS = torch.tensor(EPS, device=DEV)


def make_betas(nrung):
    return BETA_HOT * (BETA / BETA_HOT) ** (torch.arange(nrung, device=DEV) / (nrung - 1))   # hot->cold


def row_E(cand, s_i, ox, os_):
    sig = T_SIG[s_i[:, None], os_]; eps = T_EPS[s_i[:, None], os_]
    rc2 = (RCUT_FACTOR * sig) ** 2
    r2 = (cand[:, :, None, :] - ox[:, None, :, :]).square().sum(-1)
    inv6 = (sig[:, None] ** 2 / r2) ** 3
    e = 4 * eps[:, None] * (inv6 ** 2 - inv6)
    src6 = (1.0 / RCUT_FACTOR) ** 6
    e = torch.where(r2 < rc2[:, None], e - 4 * eps[:, None] * (src6 ** 2 - src6), torch.zeros_like(e))
    return e.sum(-1)


def full_E(x_int, s_int, bnd, sb):
    Mn = x_int.shape[0]
    allx = torch.cat([x_int, bnd[None].expand(Mn, bnd.shape[0], 3)], 1)
    alls = torch.cat([s_int, sb[None].expand(Mn, sb.shape[0])], 1)
    return ka_energy(allx, alls.long(), BIGL)


def heatbath_sweep(x_int, s_int, bnd, sb, Rr, gen, betas):
    """One sweep with PER-CHAIN beta vector [M]."""
    Mn, Nn = x_int.shape[0], x_int.shape[1]
    n_tel = int(TEL_FRAC * NCAND)
    for i in torch.randperm(Nn, generator=gen, device=DEV).tolist():
        oth = [j for j in range(Nn) if j != i]
        ox = torch.cat([x_int[:, oth], bnd[None].expand(Mn, bnd.shape[0], 3)], 1)
        os_ = torch.cat([s_int[:, oth], sb[None].expand(Mn, sb.shape[0])], 1)
        u = torch.randn(Mn, n_tel, 3, generator=gen, device=DEV)
        u = u / u.norm(dim=-1, keepdim=True) * (torch.rand(Mn, n_tel, 1, generator=gen, device=DEV) ** (1 / 3)) * Rr
        loc = x_int[:, i:i + 1] + SIGLOC * torch.randn(Mn, NCAND - n_tel, 3, generator=gen, device=DEV)
        cand = torch.cat([u, loc, x_int[:, i:i + 1]], 1)
        logits = -betas[:, None] * row_E(cand, s_int[:, i], ox, os_)
        pick = torch.multinomial(torch.softmax(logits, 1), 1, generator=gen).squeeze(1)
        x_int = x_int.clone(); x_int[:, i] = cand[torch.arange(Mn, device=DEV), pick]
    return x_int


def pt_exchange(x, s_int, bnd, sb, gen, parity, betas, pair_acc, pair_try):
    """Adjacent-rung Metropolis exchange for one PT stack x [nrung,n,3]; alternating parity; per-pair stats."""
    U = full_E(x, s_int, bnd, sb)                                                     # [nrung]
    nrung = x.shape[0]
    for k in range(parity, nrung - 1, 2):
        dlog = (betas[k] - betas[k + 1]) * (U[k] - U[k + 1])
        pair_try[k] += 1
        if torch.rand((), generator=gen, device=DEV).log() < dlog:
            xk = x[k].clone(); x[k] = x[k + 1]; x[k + 1] = xk
            Uk = U[k].clone(); U[k] = U[k + 1]; U[k + 1] = Uk
            pair_acc[k] += 1
    return x


def core_qc(X, sX, Y, sY, gen, rc=0.5, b=0.2, n_mc=3000):
    qX = torch.zeros(X.shape[0], device=DEV)
    for sp in (0, 1):
        xm, ym = (sX == sp), (sY == sp)
        if int(xm.sum()) > 0 and int(ym.sum()) > 0:
            qX[xm] = torch.exp(-torch.cdist(X[xm], Y[ym]).min(1).values ** 2 / (2 * b * b))
    u = torch.randn(n_mc, 3, generator=gen, device=DEV); u = u / u.norm(dim=-1, keepdim=True)
    pts = u * (torch.rand(n_mc, 1, generator=gen, device=DEV) ** (1.0 / 3.0)) * rc
    return float(qX[torch.cdist(pts, X).argmin(1)].mean())


def alien_init(X, S, ci, c, Rr, L, n, gen, n_chain_ids=16):
    """Interior positions from a different underlying chain's config at the same center; count-matched."""
    for cj in range(X.shape[0]):
        if cj % n_chain_ids == ci % n_chain_ids:
            continue
        pa = carve(X[cj], S[cj], c, Rr, L)
        xa = _mic(pa["x_in"], c, L); na = xa.shape[0]
        if na < max(4, n // 2):
            continue
        if na >= n:
            return xa[torch.argsort(xa.norm(dim=-1))[:n]]
        u = torch.randn(n - na, 3, generator=gen, device=DEV)
        pad = u / u.norm(dim=-1, keepdim=True) * (torch.rand(n - na, 1, generator=gen, device=DEV) ** (1 / 3)) * Rr
        return torch.cat([xa, pad], 0)
    raise RuntimeError("no alien config found")


def main():
    t0 = time.time()
    d = torch.load("liquid_coupling_flow/artifacts/ka3d_dataset_N4096_T0.5_rho1.15.pt",
                   map_location=DEV, weights_only=False)
    X, S, L = d["x"].to(DEV).float(), d["s"].to(DEV).long(), float(d["L"])
    gen = torch.Generator(device=DEV).manual_seed(0)
    results = []
    for Rr in RGRID:
        for cav in range(NCAV):
            ci = (cav * len(RGRID) + RGRID.index(Rr)) % X.shape[0]
            c = torch.rand(3, generator=gen, device=DEV) * L
            p = carve(X[ci], S[ci], c, Rr, L)
            if p["n_in"] < 8:
                continue
            x_in = _mic(p["x_in"], c, L); s_in = p["s_in"]
            xout = _mic(p["x_out"], c, L); bm = xout.norm(dim=-1) < (Rr + RCTX)
            bnd, sb = xout[bm], p["s_out"][bm]
            n = x_in.shape[0]
            nrung = NRUNGS[Rr]; betas = make_betas(nrung)
            xA = x_in[None].expand(nrung, n, 3).contiguous()                          # stack A: data-seeded
            xB = alien_init(X, S, ci, c, Rr, L, n, gen)[None].expand(nrung, n, 3).contiguous()  # stack B
            s_int = s_in[None].expand(nrung, n).contiguous()
            e_ref = float(full_E(x_in[None], s_in[None], bnd, sb)[0])
            coldA, coldB = [], []
            pair_acc = [0] * nrung; pair_try = [0] * nrung
            traces = {"sweep": [], "UcoldA": [], "UcoldB": []}
            nsw = SWEEPS[Rr]
            for sw in range(1, nsw + 1):
                xA = heatbath_sweep(xA, s_int, bnd, sb, Rr, gen, betas)
                xB = heatbath_sweep(xB, s_int, bnd, sb, Rr, gen, betas)
                xA = pt_exchange(xA, s_int, bnd, sb, gen, sw % 2, betas, pair_acc, pair_try)
                xB = pt_exchange(xB, s_int, bnd, sb, gen, sw % 2, betas, pair_acc, pair_try)
                if sw % COLD_EVERY == 0:
                    eA = float((full_E(xA[-1:], s_int[:1], bnd, sb)[0] - e_ref) / n)
                    eB = float((full_E(xB[-1:], s_int[:1], bnd, sb)[0] - e_ref) / n)
                    traces["sweep"].append(sw); traces["UcoldA"].append(eA); traces["UcoldB"].append(eB)
                    if sw > nsw // 2:
                        coldA.append(xA[-1].clone()); coldB.append(xB[-1].clone())
            turnA = [core_qc(coldA[i], s_in, coldA[i + 1], s_in, gen) for i in range(len(coldA) - 1)]
            crossAB = [core_qc(a, s_in, b_, s_in, gen) for a in coldA for b_ in coldB]
            withinA = [core_qc(coldA[i], s_in, coldA[j], s_in, gen)
                       for i in range(len(coldA)) for j in range(i + 1, len(coldA))]
            withinB = [core_qc(coldB[i], s_in, coldB[j], s_in, gen)
                       for i in range(len(coldB)) for j in range(i + 1, len(coldB))]
            eA = full_E(torch.stack(coldA), s_in[None].expand(len(coldA), n), bnd, sb).mean()
            eB = full_E(torch.stack(coldB), s_in[None].expand(len(coldB), n), bnd, sb).mean()
            gap = float((eB - eA) / n)
            conv = abs(gap) < 0.15
            qm = sum(crossAB) / len(crossAB)
            rec = {"R": Rr, "cav": cav, "ref_ci": ci, "n_in": n, "center": c.cpu(),
                   "qc_crossAB": crossAB, "qc_withinA": withinA, "qc_withinB": withinB,
                   "gapBA": gap, "converged": conv, "pair_acc": pair_acc, "pair_try": pair_try, "turnoverA": turnA,
                   "traces": traces, "coldA": torch.stack(coldA).cpu(), "coldB": torch.stack(coldB).cpu(),
                   "s_int": s_in.cpu(), "bnd": bnd.cpu(), "sb": sb.cpu(), "x_data": x_in.cpu(), "e_ref": e_ref}
            results.append(rec)
            torch.save({"results": results, "L": L, "T": T, "rho": float(d["rho"]), "beta_hot": BETA_HOT,
                        "protocol": "per-cavity PT (2 stacks: data/alien seeds), cold-rung crossAB qc"}, OUT)
            wA = sum(withinA) / max(1, len(withinA)); wB = sum(withinB) / max(1, len(withinB))
            exr = sum(pair_acc) / max(1, sum(pair_try))
            exr_min = min((a / t) for a, t in zip(pair_acc[:-1], pair_try[:-1]) if t > 0)
            print(f"[gptsPT] R={Rr} cav={cav} n={n:4d}  qc_AB={qm:.3f} (wA {wA:.3f} wB {wB:.3f})  "
                  f"gapBA={gap:+.3f} exch={exr:.2f}/min{exr_min:.2f} turnA={sum(turnA)/max(1,len(turnA)):.2f} "
                  f"{'OK' if conv else 'NOT-CONV'} ({(time.time()-t0)/60:.0f} min)", flush=True)
    print("\n[gptsPT] G_PTS(R) (crossAB median over converged cavities):", flush=True)
    for Rr in RGRID:
        qs = sorted(sum(r["qc_crossAB"]) / len(r["qc_crossAB"]) for r in results if r["R"] == Rr and r["converged"])
        nc = sum(1 for r in results if r["R"] == Rr and not r["converged"])
        print(f"  R={Rr}:  G_PTS={qs[len(qs)//2]:.3f} ({len(qs)} conv, {nc} flagged)" if qs
              else f"  R={Rr}:  NO converged cavities ({nc} flagged)", flush=True)
    print(f"[gptsPT] DONE -> {OUT}  ({(time.time()-t0)/60:.0f} min)", flush=True)


if __name__ == "__main__":
    main()
