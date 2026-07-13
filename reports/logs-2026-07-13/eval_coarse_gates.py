"""Task 6 Step 2: evaluation gates for the coarse-cell-then-fine-bin head (KA3DScaffoldEBMCoarse)
against the sequential per-axis baseline (KA3DScaffoldEBMBatched). Four gates, all on HELD chains
(ka3d_dataset_N4096_T0.5_rho1.15.pt, first 16 configs), K=8, R=2.0, pos_temp=0.4:

  G-NLL:     mean held-cavity NLL/n via log_prob_pair.                PASS if coarse <= -2.636.
  G-CLASH:   sampled-block clash% (<0.9 sigma, index self-excl).      PASS if coarse < 15%.
  G-DEFICIT: per-cavity median block-MTM sf-sr (exact tempered MH).   PASS if coarse median > baseline median.
  G-MASK:    coarse arm, min_sep=0.85 on BOTH sample_block_b and
             block_log_prob_b: (a) clash% strictly below unmasked
             coarse; (b) fp32 round-trip on trained-model samples
             (median<1e-3 and match(<1e-3) frac>0.9, baseline-style).

Frame/cavity/block construction is copied VERBATIM from reports/logs-2026-07-12/probe_cavity_dependence.py
(probe()) and reports/logs-2026-07-13/test_grow_mobile.py (clash_rate); the fp32 round-trip bar is copied
from reports/logs-2026-07-13/test_coarse_exact.py's float32 soft-gate (median<1e-3 and frac>0.9), which is
the BINDING measurement that script deferred to this one. NOTE: _mic(x, center, L) -- third arg is the
BOX LENGTH L, not R (past bug per CLAUDE.md context)."""
import argparse
import statistics as st
import sys
import time
import traceback

import torch

from liquid_coupling_flow.ka3d_ebm_batched import KA3DScaffoldEBMBatched
from liquid_coupling_flow.ka3d_coarse_head import KA3DScaffoldEBMCoarse
from liquid_coupling_flow.ka3d_scaffold_ar import label_to_scaffold, fixed_ball_scaffold
from liquid_coupling_flow.ka3d_cavity_ar import _mic
from liquid_coupling_flow.ka3d_cavity_carve import carve
from liquid_coupling_flow.ka_energy import ka_energy, SIGMA

ART = "liquid_coupling_flow/artifacts"
RCTX = 2.5
K = 8
R = 2.0
PT = 0.4
BETA = 2.0
BIGL = 100.0
CLASH_CUT = 0.9
MIN_SEP = 0.85
NLL_BASELINE_TARGET = -2.636         # ebm3ax_rho115 baseline held NLL/n (brief target for the coarse arm)
CLASH_PASS_THRESH = 15.0             # %
MASK_MATCH_BAR = 0.9                 # fp32 round-trip: fraction of samples with |score-sample| < 1e-3


def mkgen(seed, dev):
    return torch.Generator(device=dev).manual_seed(seed)


def clamp_ball(x, R, eps=1e-5):
    """Radially clamp any point with |x|>=R into the OPEN ball (mirrors train_ebm3d_bigbox.py)."""
    n = x.norm(dim=-1, keepdim=True)
    return x * (R * (1.0 - eps) / n.clamp_min(1e-12)).clamp(max=1.0)


def load_model(cls, ckpt, dev):
    m = cls(cat_bins=128, cat_range=2.5).to(dev)
    ck = torch.load(ckpt, map_location=dev, weights_only=False)
    miss, unexp = m.load_state_dict(ck["state_dict"], strict=False)
    m.eval()
    m.use_frame = False
    step = ck.get("step", "?")
    print(f"loaded {ckpt} (step {step}): {len(miss)} missing / {len(unexp)} unexpected keys (strict=False)",
          flush=True)
    return m


def build_cavity(x_cfg, s_cfg, c, L, R=R, RCTX=RCTX, K=K):
    """Verbatim carve/_mic/label_to_scaffold recipe (probe_cavity_dependence.py:probe()), plus the
    clamp_ball safety wrap train_ebm3d_bigbox.py's evaluate() uses before label_to_scaffold."""
    p = carve(x_cfg, s_cfg, c, R, L)
    if p["n_in"] < K + 6:
        return None
    xin = clamp_ball(_mic(p["x_in"], c, L), R)
    xo, so, _ = label_to_scaffold(xin, p["s_in"], R)
    xout = _mic(p["x_out"], c, L)
    bm = xout.norm(dim=-1) < (R + RCTX)
    bnd, sb = xout[bm], p["s_out"][bm]
    assert bnd.shape[0] < 600
    return {"xo": xo, "so": so, "bnd": bnd, "sb": sb, "n": xo.shape[0]}


def gather_cavities(X, S, L, n_need, gen, dev, max_cfg=16):
    """One cavity per config, first `max_cfg` configs of the held dataset (mirrors test_grow_mobile.py's
    `for ci in range(16)` loop). Random center per config via `gen`."""
    rows = []
    for ci in range(min(max_cfg, X.shape[0])):
        if len(rows) >= n_need:
            break
        c = torch.rand(3, generator=gen, device=dev) * L
        cav = build_cavity(X[ci], S[ci], c, L)
        if cav is None:
            continue
        cav["ci"] = ci
        rows.append(cav)
    return rows


def pick_block(xo, n, gen, dev):
    anch = fixed_ball_scaffold(n, R, dev)
    seed = int(torch.randint(n, (), generator=gen, device=dev))
    blk = torch.zeros(n, dtype=torch.bool, device=dev)
    blk[(anch - anch[seed]).norm(dim=-1).topk(K, largest=False).indices] = True
    return blk


def clash_rate(mob_x, mob_s, nbr_x, nbr_s, sig, subset=None):
    """Index-based self-exclusion clash metric (test_grow_mobile.py:clash_rate, verbatim)."""
    Mn, Km = mob_x.shape[0], mob_x.shape[1]
    hits = 0
    tot = 0
    for mm in range(Mn):
        d = torch.cdist(mob_x[mm], nbr_x[mm])
        d[torch.arange(Km), torch.arange(Km)] = float("inf")
        mn = (d / sig[mob_s[mm][:, None], nbr_s[mm][None, :]]).min(1).values
        sel = mn if subset is None else mn[subset]
        hits += int((sel < CLASH_CUT).sum())
        tot += sel.numel()
    return hits, tot


# ---------------------------------------------------------------------------
# Gates
# ---------------------------------------------------------------------------

@torch.no_grad()
def gate_nll(model, cavities):
    """G-NLL: mean held-cavity NLL/n via log_prob_pair (mirrors train_ebm3d_bigbox.py:evaluate())."""
    nlls = [float(-model.log_prob_pair(c["xo"], c["so"], c["bnd"], c["sb"], R, preordered=True) / c["n"])
            for c in cavities]
    return st.mean(nlls), nlls


@torch.no_grad()
def gate_clash(model, cavities, gen, dev, sig, M, min_sep, pos_temp=PT):
    """% placed block particles with min sigma-gap < 0.9 (index self-excl), sample_block_b(M draws)."""
    hits = tot = 0
    for cav in cavities:
        xo, so, bnd, sb, n = cav["xo"], cav["so"], cav["bnd"], cav["sb"], cav["n"]
        blk = pick_block(xo, n, gen, dev)
        xo_b = xo[None].expand(M, n, 3).contiguous()
        so_b = so[None].expand(M, n).contiguous()
        xo_g, so_g, _ = model.sample_block_b(xo_b, so_b, blk, bnd, sb, R, gen=gen, pos_temp=pos_temp,
                                              min_sep=min_sep)
        mob_x, mob_s = xo_g[:, blk], so_g[:, blk]
        cage_x = torch.cat([bnd[None].expand(M, -1, -1), xo_g[:, ~blk]], 1)
        cage_s = torch.cat([sb[None].expand(M, -1), so_g[:, ~blk]], 1)
        nbr_x = torch.cat([mob_x, cage_x], 1)
        nbr_s = torch.cat([mob_s, cage_s], 1)
        h, t = clash_rate(mob_x, mob_s, nbr_x, nbr_s, sig)
        hits += h
        tot += t
    return 100 * hits / tot, hits, tot


@torch.no_grad()
def gate_deficit(model, cavities, pos_temp, nrep, ntrial, gen, dev):
    """Per-cavity median MTM sf-sr (exact tempered MH arithmetic, probe_cavity_dependence.py:probe())."""
    defs = []
    for cav in cavities:
        xo, so, bnd, sb, n = cav["xo"], cav["so"], cav["bnd"], cav["sb"], cav["n"]
        blk = pick_block(xo, n, gen, dev)
        lq_data = float(model.block_log_prob_b(xo[None], so[None], blk, bnd, sb, R, pos_temp=pos_temp)[0])
        Ux = float(ka_energy(
            torch.cat([xo[None, blk], torch.cat([bnd, xo[~blk]], 0)[None]], 1).double(),
            torch.cat([so[None, blk], torch.cat([sb, so[~blk]], 0)[None]], 1).long(), BIGL)[0])
        u0 = -BETA * Ux - lq_data
        sfsr = []
        for _ in range(nrep):
            Xr = xo[None].expand(ntrial, n, 3).contiguous()
            Sr = so[None].expand(ntrial, n).contiguous()
            Xp, Sp, lqf = model.sample_block_b(Xr, Sr, blk, bnd, sb, R, gen=gen, pos_temp=pos_temp)
            Uy = ka_energy(
                torch.cat([Xp[:, blk], torch.cat([bnd, xo[~blk]], 0)[None].expand(ntrial, -1, -1)], 1).double(),
                torch.cat([Sp[:, blk], torch.cat([sb, so[~blk]], 0)[None].expand(ntrial, -1)], 1).long(),
                BIGL).float()
            up = -BETA * Uy - lqf
            Jm = int(torch.multinomial(torch.softmax(up, 0), 1, generator=gen))
            sf = torch.logsumexp(up, 0)
            lr = up.clone()
            lr[Jm] = u0
            sr = torch.logsumexp(lr, 0)
            sfsr.append(float(sf - sr))
        defs.append(st.median(sfsr))
    return st.median(defs), defs


@torch.no_grad()
def gate_mask_roundtrip(model, cavities, gen, dev, pos_temp, min_sep, M):
    """fp32 round-trip on the ACTUAL trained-model samples (test_coarse_exact.py's float32 soft gate,
    reused against the trained checkpoint -- the BINDING measurement deferred from Task 4)."""
    diffs = []
    for cav in cavities:
        xo, so, bnd, sb, n = cav["xo"], cav["so"], cav["bnd"], cav["sb"], cav["n"]
        blk = pick_block(xo, n, gen, dev)
        xo_b = xo[None].expand(M, n, 3).contiguous()
        so_b = so[None].expand(M, n).contiguous()
        xs, ss, lq_sample = model.sample_block_b(xo_b, so_b, blk, bnd, sb, R, gen=gen, pos_temp=pos_temp,
                                                   min_sep=min_sep)
        lq_score = model.block_log_prob_b(xs, ss, blk, bnd, sb, R, pos_temp=pos_temp, min_sep=min_sep)
        diffs.append((lq_score - lq_sample).abs())
    diff = torch.cat(diffs)
    median = float(diff.median())
    frac = float((diff < 1e-3).float().mean())
    return median, frac, diff


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=f"{ART}/ka3d_cavity_coarse_best.pt")
    ap.add_argument("--baseline-ckpt", default=f"{ART}/ka3d_cavity_ebm3ax_rho115_best.pt")
    ap.add_argument("--dataset", default=f"{ART}/ka3d_dataset_N4096_T0.5_rho1.15.pt")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--ncav-nll", type=int, default=16)
    ap.add_argument("--ncav-clash", type=int, default=12)
    ap.add_argument("--ncav-def", type=int, default=8)
    ap.add_argument("--nrep", type=int, default=4, help="MTM reps per cavity for G-DEFICIT")
    ap.add_argument("--ntrial", type=int, default=16, help="MTM trial count N per rep")
    ap.add_argument("--M", type=int, default=8, help="parallel samples per cavity for G-CLASH/G-MASK")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="reports/logs-2026-07-13/eval_coarse_gates.pt")
    a = ap.parse_args()
    dev = a.device
    sig = torch.tensor(SIGMA, device=dev)

    results = {"args": vars(a)}
    t0 = time.time()
    try:
        coarse = load_model(KA3DScaffoldEBMCoarse, a.ckpt, dev)
        baseline = load_model(KA3DScaffoldEBMBatched, a.baseline_ckpt, dev)

        D = torch.load(a.dataset, map_location=dev, weights_only=False)
        X, S, L = D["x"].to(dev).float(), D["s"].to(dev).long(), float(D["L"])
        print(f"held dataset: {a.dataset} X={tuple(X.shape)} L={L}", flush=True)

        # ---- G-NLL ----
        print("\n=== G-NLL ===", flush=True)
        cavs_nll = gather_cavities(X, S, L, a.ncav_nll, mkgen(a.seed + 1, dev), dev)
        nll_c, nll_c_list = gate_nll(coarse, cavs_nll)
        nll_b, nll_b_list = gate_nll(baseline, cavs_nll)
        nll_pass = nll_c <= NLL_BASELINE_TARGET
        print(f"  n_cavities={len(cavs_nll)}  coarse NLL/n={nll_c:+.4f}  baseline NLL/n={nll_b:+.4f}  "
              f"target<={NLL_BASELINE_TARGET}  {'PASS' if nll_pass else 'FAIL'}", flush=True)
        results.update(nll_coarse=nll_c, nll_baseline=nll_b, nll_coarse_list=nll_c_list,
                        nll_baseline_list=nll_b_list, nll_pass=nll_pass, nll_ncav=len(cavs_nll))
        torch.save(results, a.out)

        # ---- G-CLASH ----
        print("\n=== G-CLASH ===", flush=True)
        cavs_clash = gather_cavities(X, S, L, a.ncav_clash, mkgen(a.seed + 2, dev), dev)
        clash_c, hc, tc = gate_clash(coarse, cavs_clash, mkgen(a.seed + 100, dev), dev, sig, a.M, min_sep=None)
        clash_b, hb, tb = gate_clash(baseline, cavs_clash, mkgen(a.seed + 100, dev), dev, sig, a.M, min_sep=None)
        clash_pass = clash_c < CLASH_PASS_THRESH
        print(f"  n_cavities={len(cavs_clash)} M={a.M}  coarse clash={clash_c:.1f}% ({hc}/{tc})  "
              f"baseline clash={clash_b:.1f}% ({hb}/{tb})  target<{CLASH_PASS_THRESH}%  "
              f"{'PASS' if clash_pass else 'FAIL'}", flush=True)
        results.update(clash_coarse=clash_c, clash_baseline=clash_b, clash_pass=clash_pass,
                        clash_ncav=len(cavs_clash))
        torch.save(results, a.out)

        # ---- G-DEFICIT ----
        print("\n=== G-DEFICIT ===", flush=True)
        cavs_def = gather_cavities(X, S, L, a.ncav_def, mkgen(a.seed + 3, dev), dev)
        def_c, def_c_list = gate_deficit(coarse, cavs_def, PT, a.nrep, a.ntrial, mkgen(a.seed + 200, dev), dev)
        def_b, def_b_list = gate_deficit(baseline, cavs_def, PT, a.nrep, a.ntrial, mkgen(a.seed + 200, dev), dev)
        def_pass = def_c > def_b
        print(f"  n_cavities={len(cavs_def)} nrep={a.nrep} ntrial={a.ntrial}  coarse median sf-sr={def_c:+.1f}  "
              f"baseline median sf-sr={def_b:+.1f}  {'PASS' if def_pass else 'FAIL'} (coarse>baseline)", flush=True)
        print(f"  coarse per-cavity medians: {[f'{v:+.1f}' for v in def_c_list]}", flush=True)
        print(f"  baseline per-cavity medians: {[f'{v:+.1f}' for v in def_b_list]}", flush=True)
        results.update(deficit_coarse=def_c, deficit_baseline=def_b, deficit_coarse_list=def_c_list,
                        deficit_baseline_list=def_b_list, deficit_pass=def_pass, deficit_ncav=len(cavs_def))
        torch.save(results, a.out)

        # ---- G-MASK ----
        print("\n=== G-MASK (coarse arm, min_sep=0.85) ===", flush=True)
        clash_masked, hm, tm = gate_clash(coarse, cavs_clash, mkgen(a.seed + 300, dev), dev, sig, a.M,
                                           min_sep=MIN_SEP)
        mask_clash_pass = clash_masked < clash_c
        print(f"  masked clash={clash_masked:.1f}% ({hm}/{tm})  vs unmasked coarse={clash_c:.1f}%  "
              f"{'PASS' if mask_clash_pass else 'FAIL'} (strictly below)", flush=True)
        median_rt, frac_rt, diff_rt = gate_mask_roundtrip(coarse, cavs_clash, mkgen(a.seed + 400, dev), dev,
                                                            PT, MIN_SEP, a.M)
        mask_fp32_pass = median_rt < 1e-3 and frac_rt > MASK_MATCH_BAR
        print(f"  fp32 round-trip: median|score-sample|={median_rt:.2e}  match(<1e-3)={100*frac_rt:.1f}%  "
              f"bar: median<1e-3 and frac>{100*MASK_MATCH_BAR:.0f}%  {'PASS' if mask_fp32_pass else 'FAIL'}",
              flush=True)
        if not mask_fp32_pass:
            print("  deployment must score in float64", flush=True)
        mask_pass = mask_clash_pass and mask_fp32_pass
        results.update(mask_clash=clash_masked, mask_clash_pass=mask_clash_pass, mask_rt_median=median_rt,
                        mask_rt_frac=frac_rt, mask_fp32_pass=mask_fp32_pass, mask_pass=mask_pass)
        torch.save(results, a.out)

        # ---- summary table ----
        print("\n=== SUMMARY ===", flush=True)
        rows = [
            ("G-NLL", f"{nll_c:+.4f}", f"{nll_b:+.4f}", f"<= {NLL_BASELINE_TARGET}", nll_pass),
            ("G-CLASH", f"{clash_c:.1f}%", f"{clash_b:.1f}%", f"< {CLASH_PASS_THRESH}%", clash_pass),
            ("G-DEFICIT", f"{def_c:+.1f}", f"{def_b:+.1f}", "coarse > baseline", def_pass),
            ("G-MASK", f"masked={clash_masked:.1f}% rt_frac={100*frac_rt:.0f}%", f"unmasked={clash_c:.1f}%",
             "masked<unmasked & rt frac>90%", mask_pass),
        ]
        hdr = f"{'gate':<10} {'coarse':<32} {'baseline/ref':<20} {'threshold':<28} {'result'}"
        print(hdr, flush=True)
        print("-" * len(hdr), flush=True)
        for name, cv, bv, th, ok in rows:
            print(f"{name:<10} {cv:<32} {bv:<20} {th:<28} {'PASS' if ok else 'FAIL'}", flush=True)
        results["summary_pass"] = {"G-NLL": nll_pass, "G-CLASH": clash_pass, "G-DEFICIT": def_pass,
                                    "G-MASK": mask_pass}
        results["elapsed_s"] = time.time() - t0
        torch.save(results, a.out)
        print(f"\nsaved -> {a.out}  ({time.time()-t0:.0f}s)", flush=True)
    except Exception as e:
        print(f"\nEXCEPTION: {e}", flush=True)
        traceback.print_exc()
        results["exception"] = repr(e)
        results["elapsed_s"] = time.time() - t0
        torch.save(results, a.out)
        print(f"saved PARTIAL results -> {a.out}", flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
