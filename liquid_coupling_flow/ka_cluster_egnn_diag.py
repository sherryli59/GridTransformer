"""Bottleneck diagnostics for the EGNN cluster flow (vs proposal-A/B baseline numbers).
Subcommands: split | nocage | cagesens | losst | traj | overfit
Usage: python -m liquid_coupling_flow.ka_cluster_egnn_diag <subcommand> [ckpt-name]
Every subcommand prints numbers; figures are saved under liquid_coupling_flow/artifacts/ (full path printed)."""
from __future__ import annotations
import os, sys, torch
from liquid_coupling_flow import ka_cluster as KC, ka_cluster_egnn as E
from liquid_coupling_flow.ka_cluster_flow import slot_order, _scaffold
from liquid_coupling_flow.ka_gridformer import _wrap_pm

DEV = "cuda" if torch.cuda.is_available() else "cpu"
ART = E.ART


def _setup(B=128, ck_name="ka_cluster_egnn_N100.pt"):
    sc, L, geo = _scaffold(100, DEV)
    ref = torch.load(os.path.join(ART, "ka_reference_N100.pt"), map_location=DEV, weights_only=False)
    pos0, sso = slot_order(ref["x"][:B].to(DEV), ref["s"].to(DEV).long(), geo, 100)
    ck = torch.load(os.path.join(ART, ck_name), map_location=DEV, weights_only=False)
    return sc, L, geo, pos0, sso, ck, E.load_flow(ck, DEV)


def _minr_split(xC, pos0, cl, L, k):
    """(intra min-r [B*k], cage min-r [B*k]) for a resampled cluster given the TRUE surroundings."""
    di = _wrap_pm(xC[:, :, None] - xC[:, None], L).norm(dim=-1) + torch.eye(k, device=DEV)[None] * 1e3
    mask = torch.ones(pos0.shape[1], dtype=torch.bool, device=DEV); mask[cl] = False
    dc = _wrap_pm(xC[:, :, None] - pos0[:, mask][:, None], L).norm(dim=-1)
    return di.min(-1).values.reshape(-1), dc.min(-1).values.reshape(-1)


@torch.no_grad()
def split(ck_name="ka_cluster_egnn_N100.pt", n_steps=8, B=128):
    """Intra vs cage clash split on single-cluster resamples given TRUE surroundings (+ parity among intra)."""
    sc, L, geo, pos0, sso, ck, P = _setup(B, ck_name); k = ck["k"]; P.n_steps = n_steps
    print(f"ckpt {ck_name} step {ck['step']} loss {ck['loss_last']:.3f} n_steps {n_steps}", flush=True)
    iu = torch.triu(torch.ones(k, k, device=DEV), 1).bool()
    par = torch.arange(k, device=DEV) % 2; same = par[:, None] == par[None, :]
    ii, cc, ti, tc, sp_cl, cr_cl = [], [], [], [], 0, 0
    for seed in range(0, 100, 5):
        cl = KC.cluster_slots(seed, sc, k, L)
        xC = P.sample_fast(pos0, sso, cl, sc, L)
        a, b = _minr_split(xC, pos0, cl, L, k); ii.append(a); cc.append(b)
        a, b = _minr_split(pos0[:, cl], pos0, cl, L, k); ti.append(a); tc.append(b)
        di = _wrap_pm(xC[:, :, None] - xC[:, None], L).norm(dim=-1)
        clash = (di < 0.7) & iu[None]
        sp_cl += (clash & same[None]).sum().item(); cr_cl += (clash & ~same[None]).sum().item()
    ii, cc, ti, tc = (torch.cat(v) for v in (ii, cc, ti, tc))
    for nm, gen, tru in (("intra", ii, ti), ("cage ", cc, tc)):
        print(f"{nm}: clash<0.7 {(gen<0.7).float().mean()*100:5.1f}%  mean {gen.mean():.2f}   "
              f"(TRUE {(tru<0.7).float().mean()*100:.1f}% / {tru.mean():.2f})", flush=True)
    ov = torch.minimum(ii, cc)
    print(f"overall: clash {(ov<0.7).float().mean()*100:.1f}%   [EGNN-15k 52 / A-sharp 44 / B 55 / data 0]")
    tot = max(sp_cl + cr_cl, 1)
    print(f"parity among intra clashes: same {100*sp_cl/tot:.0f}% (43% by construction)")


@torch.no_grad()
def nocage(B=128):
    """Intra-ONLY capability test: flow trained with n_cage=0; base anchored at the TRUE cluster centroid
    (position leak is fine — this tests exclusion-carving, not placement). n_steps sweep guards against
    integrator artifacts."""
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    sc, L, geo, pos0, sso, ck, P = _setup(B, "ka_cluster_egnn_nocage_N100.pt"); k = ck["k"]
    iu = torch.triu(torch.ones(k, k, device=DEV), 1).bool()
    pair_g, pair_t, spp = [], [], []
    for ns in (8, 32, 64):
        P.n_steps = ns; gen, tru = [], []
        for seed in range(0, 100, 5):
            cl = KC.cluster_slots(seed, sc, k, L)
            xC = P.sample_fast(pos0, sso, cl, sc, L)
            dg = _wrap_pm(xC[:, :, None] - xC[:, None], L).norm(dim=-1)
            dt = _wrap_pm(pos0[:, cl][:, :, None] - pos0[:, cl][:, None], L).norm(dim=-1)
            gen.append((dg + torch.eye(k, device=DEV)[None] * 1e3).min(-1).values.reshape(-1))
            tru.append((dt + torch.eye(k, device=DEV)[None] * 1e3).min(-1).values.reshape(-1))
            if ns == 32:
                sp = sso[:, cl]; ps = sp[:, :, None] + sp[:, None, :]            # 0=AA 1=AB 2=BB
                pair_g.append(dg[:, iu].reshape(-1)); pair_t.append(dt[:, iu].reshape(-1))
                spp.append(ps[:, iu].reshape(-1))
        gen, tru = torch.cat(gen), torch.cat(tru)
        print(f"n_steps {ns:2d}: intra-clash {(gen<0.7).float().mean()*100:5.1f}%  mean-minr {gen.mean():.2f}   "
              f"(TRUE {(tru<0.7).float().mean()*100:.1f}% / {tru.mean():.2f})  [with-cage EGNN: 24% / 0.91]", flush=True)
    pg, pt, ss = torch.cat(pair_g), torch.cat(pair_t), torch.cat(spp)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for a, (v, nm) in zip(axes, ((0, "AA"), (1, "AB"), (2, "BB"))):
        a.hist(pt[ss == v].cpu().numpy(), bins=40, range=(0, 3), density=True, alpha=0.6, label="data")
        a.hist(pg[ss == v].cpu().numpy(), bins=40, range=(0, 3), density=True, alpha=0.6, label="nocage flow")
        a.set_title(f"intra pair dist {nm}"); a.legend()
    out = os.path.join(ART, "ka_egnn_nocage_intra.png"); fig.tight_layout(); fig.savefig(out, dpi=120)
    print("saved", out, flush=True)


@torch.no_grad()
def cagesens(B=64):
    """Does the velocity field listen to the cage? Perturb the cage particle NEAREST the cage centroid
    radially outward by 0.15 vs the FARTHEST (control); compare the strongest cluster-velocity response."""
    sc, L, geo, pos0, sso, ck, P = _setup(B); k = ck["k"]
    torch.manual_seed(0)
    for seed in (5, 40, 75):
        cl = KC.cluster_slots(seed, sc, k, L)
        cloud, sp, c = E.build_cloud(pos0, sso, cl, sc, L, ck["n_cage"])
        x1 = cloud[:, :k]; z = E.sample_base(c, k, P.sigma_b)
        dcage = _wrap_pm(cloud[:, k:] - c[:, None], L).norm(dim=-1)              # [B,n_cage]
        for t0 in (0.5, 1.0):
            xt = torch.remainder(z + t0 * _wrap_pm(x1 - z, L), L)                # on-path state
            base = torch.cat([xt, cloud[:, k:]], 1)
            tv = torch.full((B,), t0, device=DEV)
            v0 = P.ce.egnn.forward(tv, base, sp)[:, :k]
            row = f"seed {seed} t={t0}: |v| {v0.norm(dim=-1).mean():.3f}"
            for tag, pick in (("nearest", dcage.argmin(-1)), ("farthest", dcage.argmax(-1))):
                pert = base.clone(); j = pick + k
                dir_ = _wrap_pm(pert[torch.arange(B), j] - c, L)
                dir_ = dir_ / dir_.norm(dim=-1, keepdim=True).clamp_min(1e-6)
                pert[torch.arange(B), j] = torch.remainder(pert[torch.arange(B), j] + 0.15 * dir_, L)
                dv = (P.ce.egnn.forward(tv, pert, sp)[:, :k] - v0).norm(dim=-1).max(-1).values
                row += f"  d|v|({tag}) {dv.mean():.4f}"
            print(row, flush=True)


@torch.no_grad()
def losst(reps=60, batch=64):
    """Held-out FM squared error binned by t (rows) x distance-to-nearest-cage at x_t (cols): WHERE is the error?"""
    from liquid_coupling_flow.ka_gridformer_train import augment
    sc, L, geo = _scaffold(100, DEV)
    ref = torch.load(os.path.join(ART, "ka_reference_N100.pt"), map_location=DEV, weights_only=False)
    data, s = ref["x"].to(DEV), ref["s"].to(DEV).long()
    ck = torch.load(os.path.join(ART, "ka_cluster_egnn_N100.pt"), map_location=DEV, weights_only=False)
    P = E.load_flow(ck, DEV); k, n_cage = ck["k"], ck["n_cage"]
    tg = torch.linspace(0.05, 0.95, 10, device=DEV)
    err = torch.zeros(10, 3); cnt = torch.zeros(10, 3)
    dedges = torch.tensor([0.0, 1.0, 1.5, 100.0], device=DEV)
    torch.manual_seed(1)
    for rep in range(reps):
        idx = torch.randint(0, data.shape[0], (batch,), device=DEV)
        pos, s_ord = slot_order(augment(data[idx], L), s, geo, 100)
        seed = int(torch.randint(0, 100, (1,)).item()); cl = KC.cluster_slots(seed, sc, k, L)
        cloud, sp, c = E.build_cloud(pos, s_ord, cl, sc, L, n_cage); x1 = cloud[:, :k]
        z = E.sample_base(c, k, P.sigma_b)
        perm = E.ot_assign(z, x1, sp[:, :k], L)
        target = _wrap_pm(torch.gather(x1, 1, perm[..., None].expand(-1, -1, 2)) - z, L)
        for ti, t0 in enumerate(tg):
            xt = torch.remainder(z + t0 * target, L)
            v = P.ce.egnn.forward(torch.full((batch,), float(t0), device=DEV),
                                  torch.cat([xt, cloud[:, k:]], 1), sp)[:, :k]
            e2 = ((v - target) ** 2).sum(-1)                                     # [B,k]
            dc = _wrap_pm(xt[:, :, None] - cloud[:, None, k:], L).norm(dim=-1).min(-1).values
            db = torch.bucketize(dc, dedges) - 1                                 # 0,1,2
            for b in range(3):
                m = db == b
                err[ti, b] += e2[m].sum().item(); cnt[ti, b] += int(m.sum())
    tab = err / cnt.clamp(min=1)
    print("FM sq-err by t (rows) x dist-to-nearest-cage [<1.0 | 1.0-1.5 | >1.5]:")
    for ti, t0 in enumerate(tg):
        print(f"  t={float(t0):.2f}  " + "  ".join(f"{tab[ti, b]:7.3f}" for b in range(3)), flush=True)


@torch.no_grad()
def traj(n_steps=32, B=128):
    """Clash timeline along the sampling ODE: when does the cage clash fail to resolve?"""
    import collections
    sc, L, geo, pos0, sso, ck, P = _setup(B); k = ck["k"]
    agg = collections.defaultdict(lambda: [0.0, 0.0, 0])
    for seed in range(0, 100, 10):
        cl = KC.cluster_slots(seed, sc, k, L)
        cloud, sp, c = E.build_cloud(pos0, sso, cl, sc, L, ck["n_cage"])
        cage = cloud[:, k:]; clx = E.sample_base(c, k, P.sigma_b); dt = 1.0 / n_steps
        mask = torch.ones(100, dtype=torch.bool, device=DEV); mask[cl] = False

        def vo(x, t):
            return P.ce.egnn.forward(torch.tensor(float(t), device=DEV), torch.cat([x, cage], 1), sp)[:, :k]

        for i in range(n_steps + 1):
            di = _wrap_pm(clx[:, :, None] - clx[:, None], L).norm(dim=-1) + torch.eye(k, device=DEV)[None] * 1e3
            dc = _wrap_pm(clx[:, :, None] - pos0[:, mask][:, None], L).norm(dim=-1)
            a = agg[round(i * dt, 3)]
            a[0] += float((di.min(-1).values < 0.7).float().mean() * 100)
            a[1] += float((dc.min(-1).values < 0.7).float().mean() * 100)
            a[2] += 1
            if i == n_steps:
                break
            t = i * dt
            v1 = vo(clx, t); v2 = vo(torch.remainder(clx + 0.5 * dt * v1, L), t + 0.5 * dt)
            v3 = vo(torch.remainder(clx + 0.5 * dt * v2, L), t + 0.5 * dt)
            v4 = vo(torch.remainder(clx + dt * v3, L), t + dt)
            clx = torch.remainder(clx + (dt / 6.0) * (v1 + 2 * v2 + 2 * v3 + v4), L)
    print("   t    intra%  cage%")
    for t in sorted(agg):
        a, b, n = agg[t]
        print(f"  {t:.2f}  {a / n:6.1f}  {b / n:6.1f}", flush=True)


@torch.no_grad()
def overfit():
    """Eval the Task-5 overfit probe: resample the FIXED cluster on the 4 memorized configs (32 draws each).
    Can the flow drive clash -> ~0 on data it has memorized? (capacity discriminator, cf. coupling-flow arm)"""
    sc, L, geo = _scaffold(100, DEV)
    ref = torch.load(os.path.join(ART, "ka_reference_N100.pt"), map_location=DEV, weights_only=False)
    pos0, sso = slot_order(ref["x"][:4].to(DEV), ref["s"].to(DEV).long(), geo, 100)
    ck = torch.load(os.path.join(ART, "ka_cluster_egnn_overfit_N100.pt"), map_location=DEV, weights_only=False)
    P = E.load_flow(ck, DEV); P.n_steps = 32; k = ck["k"]
    cl = KC.cluster_slots(5, sc, k, L)
    pos_r = pos0.repeat_interleave(32, 0); sso_r = sso.repeat_interleave(32, 0)
    xC = P.sample_fast(pos_r, sso_r, cl, sc, L)
    a, b = _minr_split(xC, pos_r, cl, L, k)
    x1 = pos_r[:, cl]
    perm = E.ot_assign(xC, x1, sso_r[:, cl], L)
    rms = _wrap_pm(xC - torch.gather(x1, 1, perm[..., None].expand(-1, -1, 2)), L).norm(dim=-1).mean()
    print(f"OVERFIT probe (4 configs, seed-5 cluster, 32 draws each):")
    print(f"  intra clash {(a<0.7).float().mean()*100:.1f}%  cage clash {(b<0.7).float().mean()*100:.1f}%  "
          f"overall {(torch.minimum(a,b)<0.7).float().mean()*100:.1f}%  |sample-true(OT)| {rms:.3f}", flush=True)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "split"
    kw = {}
    if cmd == "split" and len(sys.argv) > 2:
        kw["ck_name"] = sys.argv[2]
    {"split": split, "nocage": nocage, "cagesens": cagesens, "losst": losst,
     "traj": traj, "overfit": overfit}[cmd](**kw)
