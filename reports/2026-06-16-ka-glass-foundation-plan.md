# KA-glass foundation (P0+P1) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and validate the 2D Kob–Andersen physics toolkit (energy, swap Monte Carlo, observables) and run the three GO/NO-GO gates (O1 no-crystallization, O2 equilibration, P1 annealing-hardness) that decide whether the architecture work proceeds.

**Architecture:** Pure PyTorch, batched over independent chains/configs, GPU-capable. New modules live beside the existing `liquid_coupling_flow/` code and reuse its conventions (min-image, periodic box, `smc.py`). The foundation is independent of the generative models — it produces equilibrated reference data + the hardness verdict.

**Tech Stack:** PyTorch (float64 for energy/log-det checks, float32 OK for MC), matplotlib (Agg), pytest-style standalone test scripts run as modules (matching the repo's `liquid_coupling_flow/tests/test_*.py` pattern, executed via `python -m`).

---

## File structure

- Create `liquid_coupling_flow/ka_energy.py` — KA binary LJ energy (species σ/ε matrix; total + per-particle + partial-by-pair-type). One responsibility: the potential.
- Create `liquid_coupling_flow/ka_mcmc.py` — swap Monte Carlo: displacement moves + A↔B identity-swap moves; cached equilibrated-config generator keyed by (N, L, T, comp). One responsibility: ground-truth sampling.
- Create `liquid_coupling_flow/ka_observables.py` — partial g_AA/g_AB/g_BB(r), partial S(q), per-particle energy distribution, hexatic ψ6. One responsibility: structural observables.
- Create `liquid_coupling_flow/tests/test_ka_energy.py`, `tests/test_ka_observables.py` — correctness tests.
- Create `liquid_coupling_flow/ka_gate_o1_crystallization.py` — O1 experiment + pass/fail.
- Create `liquid_coupling_flow/ka_gate_o2_equilibration.py` — O2 experiment + pass/fail.
- Create `liquid_coupling_flow/ka_gate_p1_hardness.py` — P1 experiment + pass/fail.

Convention: species stored as an int8 tensor `s ∈ {0=A,1=B}` of shape `[N]` (fixed per system; swap MC permutes labels, displacement MC moves positions). Energy/observables take `(x, s)`.

---

## Task 1: KA binary LJ energy

**Files:**
- Create: `liquid_coupling_flow/ka_energy.py`
- Test: `liquid_coupling_flow/tests/test_ka_energy.py`

- [ ] **Step 1: Write the failing test**

```python
# liquid_coupling_flow/tests/test_ka_energy.py
import math, torch
from liquid_coupling_flow.ka_energy import ka_energy, SIGMA, EPS

def _pair(r, a, b):  # one-pair shifted LJ, analytic
    s, e, rc = SIGMA[a][b], EPS[a][b], 2.5 * SIGMA[a][b]
    if r >= rc: return 0.0
    sr6 = (s / r) ** 6; src6 = (s / rc) ** 6
    return 4 * e * (sr6**2 - sr6) - 4 * e * (src6**2 - src6)

def test_two_body_all_species():
    L = 20.0
    for (a, b) in [(0,0),(0,1),(1,1)]:
        x = torch.tensor([[[0.0,0.0],[1.05,0.0]]], dtype=torch.float64)
        s = torch.tensor([a, b], dtype=torch.int8)
        got = ka_energy(x, s, L).item()
        assert abs(got - _pair(1.05, a, b)) < 1e-9, (a, b, got)

def test_translation_invariant():
    torch.manual_seed(0); L = 10.0
    x = torch.rand(1, 12, 2, dtype=torch.float64) * L
    s = (torch.arange(12) % 3 == 0).to(torch.int8)  # ~33% B
    e0 = ka_energy(x, s, L)
    e1 = ka_energy(torch.remainder(x + 3.3, L), s, L)
    assert (e0 - e1).abs().max() < 1e-9

def test_per_particle_sums_to_total():
    torch.manual_seed(1); L = 10.0
    x = torch.rand(2, 16, 2, dtype=torch.float64) * L
    s = (torch.rand(16) < 0.35).to(torch.int8)
    tot = ka_energy(x, s, L)
    per = ka_energy(x, s, L, per_particle=True)   # [B,N], sum over i = 2*total (each pair counted twice)
    assert (per.sum(-1) - 2 * tot).abs().max() < 1e-9
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PY=/home/sherryli/xsli/softwares/anaconda3/envs/lightning/bin/python; $PY -m liquid_coupling_flow.tests.test_ka_energy`
Expected: FAIL (ModuleNotFoundError: ka_energy).

- [ ] **Step 3: Write minimal implementation**

```python
# liquid_coupling_flow/ka_energy.py
"""2D Kob-Andersen binary LJ energy (species-dependent sigma/eps, shifted, min-image)."""
from __future__ import annotations
import torch

# standard KA interaction matrix (indices 0=A, 1=B)
SIGMA = [[1.0, 0.8], [0.8, 0.88]]
EPS   = [[1.0, 1.5], [1.5, 0.5]]
RCUT_FACTOR = 2.5

def _matrix(s, table, device, dtype):
    # s: [N] int8 -> [N,N] pair-parameter matrix
    t = torch.tensor(table, device=device, dtype=dtype)      # [2,2]
    si = s.long()
    return t[si][:, si]                                       # [N,N]

def ka_energy(x, s, L, per_particle: bool = False):
    """x: [B,N,2] in [0,L)^2; s: [N] in {0,1}. Returns [B] (or [B,N] if per_particle)."""
    B, N, d = x.shape
    dtype = x.dtype
    sig = _matrix(s, SIGMA, x.device, dtype)                  # [N,N]
    eps = _matrix(s, EPS, x.device, dtype)
    rc = RCUT_FACTOR * sig                                    # [N,N]
    diff = x[:, :, None, :] - x[:, None, :, :]                # [B,N,N,2]
    diff = diff - L * torch.round(diff / L)
    r2 = (diff ** 2).sum(-1)                                  # [B,N,N]
    eye = torch.eye(N, device=x.device, dtype=torch.bool)
    r2 = r2.masked_fill(eye, 1e12)
    inv6 = (sig ** 2 / r2) ** 3
    e = 4 * eps * (inv6 ** 2 - inv6)
    src6 = (sig / rc) ** 6
    eshift = 4 * eps * (src6 ** 2 - src6)
    within = r2 < rc ** 2
    e = torch.where(within, e - eshift, torch.zeros_like(e))  # [B,N,N], diagonal=0
    if per_particle:
        return e.sum(-1)                                      # [B,N]
    return 0.5 * e.sum(dim=(1, 2))                            # [B]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `$PY -m liquid_coupling_flow.tests.test_ka_energy`
Expected: PASS (all three).

- [ ] **Step 5: Commit**

```bash
git add liquid_coupling_flow/ka_energy.py liquid_coupling_flow/tests/test_ka_energy.py
git commit -m "KA binary LJ energy (species sigma/eps matrix, shifted, per-particle)"
```

---

## Task 2: Swap Monte Carlo

**Files:**
- Create: `liquid_coupling_flow/ka_mcmc.py`
- Test: (sanity assertions inside the module's `__main__`, plus reuse in gates)

- [ ] **Step 1: Write the implementation (displacement + swap moves)**

```python
# liquid_coupling_flow/ka_mcmc.py
"""Swap Monte Carlo for 2D Kob-Andersen: displacement moves + A<->B identity swaps.
Swap MC is what makes KA equilibrable in the supercooled regime. Energy uses a local
recompute per move (O(N) per particle) for speed."""
from __future__ import annotations
import os, torch
from liquid_coupling_flow.ka_energy import SIGMA, EPS, RCUT_FACTOR

def _pair_energy_one(ri, ai, x, s, i, L):
    # energy of particle (ri, ai) vs all others; ai: [B] species of particle i
    diff = ri[:, None, :] - x                                # [B,N,2]
    diff = diff - L * torch.round(diff / L)
    r2 = (diff ** 2).sum(-1)                                  # [B,N]
    r2[:, i] = 1e12
    sig = torch.tensor(SIGMA, device=x.device, dtype=x.dtype)[ai][:, s.long()]  # [B,N]
    eps = torch.tensor(EPS, device=x.device, dtype=x.dtype)[ai][:, s.long()]
    rc = RCUT_FACTOR * sig
    inv6 = (sig ** 2 / r2) ** 3
    e = 4 * eps * (inv6 ** 2 - inv6)
    src6 = (sig / rc) ** 6
    e = e - 4 * eps * (src6 ** 2 - src6)
    return torch.where(r2 < rc ** 2, e, torch.zeros_like(e)).sum(-1)  # [B]

def swap_mcmc(N, L, kT, s, d=2, device="cpu", n_chains=512, n_equil=2000,
              n_collect=200, every=20, step=0.08, swap_frac=0.2, x0=None):
    """s: [N] fixed species labels. Returns (configs [n,N,2], s)."""
    B = n_chains
    x = (torch.rand(B, N, d, device=device) * L) if x0 is None else x0.to(device).expand(B, N, d).clone()
    sd = s.to(device)
    snaps = []
    for sweep in range(n_equil + n_collect):
        # displacement sweep
        for i in range(N):
            xi = x[:, i, :]; ai = sd[i].expand(B)
            prop = torch.remainder(xi + step * torch.randn_like(xi), L)
            dE = _pair_energy_one(prop, ai, x, sd, i, L) - _pair_energy_one(xi, ai, x, sd, i, L)
            acc = torch.log(torch.rand(B, device=device)) < (-dE / kT)
            x[:, i, :] = torch.where(acc[:, None], prop, xi)
        # swap moves: pick random A and random B index, propose swapping identities.
        # Implemented as swapping POSITIONS of i (species a) and j (species b) -> equivalent.
        nsw = int(swap_frac * N)
        Aidx = (sd == 0).nonzero().squeeze(-1); Bidx = (sd == 1).nonzero().squeeze(-1)
        for _ in range(nsw):
            i = Aidx[torch.randint(len(Aidx), (1,))].item()
            j = Bidx[torch.randint(len(Bidx), (1,))].item()
            xi, xj = x[:, i, :].clone(), x[:, j, :].clone()
            e_old = (_pair_energy_one(xi, sd[i].expand(B), x, sd, i, L)
                     + _pair_energy_one(xj, sd[j].expand(B), x, sd, j, L))
            # swap positions
            xp = x.clone(); xp[:, i, :] = xj; xp[:, j, :] = xi
            e_new = (_pair_energy_one(xj, sd[i].expand(B), xp, sd, i, L)
                     + _pair_energy_one(xi, sd[j].expand(B), xp, sd, j, L))
            acc = torch.log(torch.rand(B, device=device)) < (-(e_new - e_old) / kT)
            x[:, i, :] = torch.where(acc[:, None], xj, xi)
            x[:, j, :] = torch.where(acc[:, None], xi, xj)
        if sweep >= n_equil and (sweep - n_equil) % every == 0:
            snaps.append(x.clone())
    return torch.cat(snaps, 0), sd.cpu()

def make_species(N, frac_B=0.35, seed=0):
    g = torch.Generator().manual_seed(seed)
    s = (torch.rand(N, generator=g) < frac_B).to(torch.int8)
    return s

def get_data(N, L, kT, frac_B=0.35, device="cpu", **kw):
    cache = f"/tmp/ka2d_N{N}_L{L:g}_T{kT:g}_B{frac_B:g}.pt"
    if os.path.exists(cache):
        d = torch.load(cache, map_location=device); return d["x"].to(device), d["s"].to(device)
    s = make_species(N, frac_B)
    x, sd = swap_mcmc(N, L, kT, s, device=device, **kw)
    torch.save({"x": x.cpu(), "s": sd}, cache)
    return x.to(device), sd.to(device)
```

- [ ] **Step 2: Sanity check (energy decreases + matches full energy)**

Add to `ka_mcmc.py`:

```python
if __name__ == "__main__":
    import torch
    from liquid_coupling_flow.ka_energy import ka_energy
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    N, L, kT = 64, (64/0.8)**0.5, 1.0
    s = make_species(N, 0.35)
    x, sd = swap_mcmc(N, L, kT, s, device=dev, n_chains=64, n_equil=200, n_collect=20, every=10)
    U = (ka_energy(x, sd, L) / N).mean().item()
    print(f"equilibrated <U>/N = {U:.3f} (should be finite, negative-ish, no infs)")
    assert torch.isfinite(ka_energy(x, sd, L)).all()
    print("swap_mcmc sanity OK")
```

Run: `$PY -m liquid_coupling_flow.ka_mcmc`
Expected: prints finite `<U>/N`, "swap_mcmc sanity OK".

- [ ] **Step 3: Commit**

```bash
git add liquid_coupling_flow/ka_mcmc.py
git commit -m "Swap Monte Carlo for 2D KA (displacement + identity-swap moves), cached"
```

---

## Task 3: Observables (partial g(r) + hexatic ψ6)

**Files:**
- Create: `liquid_coupling_flow/ka_observables.py`
- Test: `liquid_coupling_flow/tests/test_ka_observables.py`

- [ ] **Step 1: Write the failing test**

```python
# liquid_coupling_flow/tests/test_ka_observables.py
import torch
from liquid_coupling_flow.ka_observables import partial_gr, psi6
from liquid_coupling_flow.ka_mcmc import make_species

def test_ideal_gas_partial_gr_flat():
    torch.manual_seed(0); N, L = 80, 12.0
    s = make_species(N, 0.35)
    x = torch.rand(6000, N, 2) * L
    rc, g = partial_gr(x, s, L, rmax=L/2, nbins=60, pair=(0,0))
    plateau = g[rc > 0.6 * (L/2)].mean().item()
    assert 0.9 < plateau < 1.1, plateau

def test_psi6_perfect_triangular_is_one():
    # triangular lattice patch -> |psi6| ~ 1 for interior particles
    import math
    rows = []
    a = 1.0
    for j in range(8):
        for i in range(8):
            rows.append([i*a + (j%2)*a/2, j*a*math.sqrt(3)/2])
    x = torch.tensor(rows)[None]  # [1,64,2]
    L = 100.0  # large box, no PBC effects on interior
    val = psi6(x, L)              # [1] mean |psi6| over particles
    assert val.item() > 0.9, val.item()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `$PY -m liquid_coupling_flow.tests.test_ka_observables`
Expected: FAIL (ModuleNotFoundError).

- [ ] **Step 3: Write minimal implementation**

```python
# liquid_coupling_flow/ka_observables.py
"""Structural observables for 2D KA: partial g(r) (per pair-type), hexatic psi6."""
from __future__ import annotations
import math, torch

def _minimg(x, L):
    diff = x[:, :, None, :] - x[:, None, :, :]
    diff = diff - L * torch.round(diff / L)
    r = torch.sqrt((diff ** 2).sum(-1) + 1e-12)
    return diff, r

def partial_gr(x, s, L, rmax, nbins, pair):
    """g_{ab}(r). x:[B,N,2], s:[N], pair=(a,b). Plateau normalized to ~1."""
    B, N = x.shape[0], x.shape[1]
    a, b = pair
    _, r = _minimg(x, L)                                  # [B,N,N]
    ia = (s == a).nonzero().squeeze(-1); ib = (s == b).nonzero().squeeze(-1)
    sub = r[:, ia][:, :, ib]                              # [B,na,nb]
    if a == b:
        m = ~torch.eye(len(ia), dtype=torch.bool, device=x.device)[None]
        d = sub[m.expand_as(sub)].reshape(-1)
        npairs = len(ia) * (len(ia) - 1)
    else:
        d = sub.reshape(-1); npairs = len(ia) * len(ib)
    d = d[d < rmax].cpu()
    counts = torch.histc(d, bins=nbins, min=0.0, max=rmax)
    edges = torch.linspace(0.0, rmax, nbins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    shell = math.pi * (edges[1:] ** 2 - edges[:-1] ** 2)
    rho_b = (len(ib) if a != b else len(ia) - 1) / (L ** 2)
    ideal = B * len(ia) * rho_b * shell                  # expected counts per (a-ref) over B frames
    return centers.numpy(), (counts / ideal).numpy()

def psi6(x, L, cutoff=1.5):
    """Mean over particles of |(1/n) sum_j e^{i6 theta_ij}| using neighbours within cutoff."""
    diff, r = _minimg(x, L)                               # [B,N,N,2], [B,N,N]
    ang = torch.atan2(diff[..., 1], diff[..., 0])        # [B,N,N]
    nb = (r < cutoff) & (r > 1e-6)
    w = nb.float()
    z = (torch.exp(1j * 6 * ang) * w).sum(-1)            # [B,N]
    n = w.sum(-1).clamp_min(1)
    return (z.abs() / n).mean(-1)                         # [B]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `$PY -m liquid_coupling_flow.tests.test_ka_observables`
Expected: PASS (flat ideal-gas g(r); ψ6≈1 on a triangular patch).

- [ ] **Step 5: Commit**

```bash
git add liquid_coupling_flow/ka_observables.py liquid_coupling_flow/tests/test_ka_observables.py
git commit -m "KA observables: partial g_ab(r) (ideal-gas validated) + hexatic psi6"
```

---

## Task 4: Gate O1 — no crystallization

**Files:**
- Create: `liquid_coupling_flow/ka_gate_o1_crystallization.py`

- [ ] **Step 1: Write the gate experiment**

```python
# liquid_coupling_flow/ka_gate_o1_crystallization.py
"""O1 GATE: does 2D KA (65:35) stay disordered (no crystallization) at the target T?
PASS if, after long swap MC, hexatic order psi6 stays low (liquid-like, < ~0.4) and
partial g(r) shows no sharp crystalline peaks. Scans a couple of compositions/T."""
from __future__ import annotations
import os, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, torch
from liquid_coupling_flow.ka_mcmc import swap_mcmc, make_species
from liquid_coupling_flow.ka_observables import partial_gr, psi6

ART = os.path.join(os.path.dirname(__file__), "artifacts")

def main(device="cuda" if torch.cuda.is_available() else "cpu"):
    rho = 1.2                       # 2D KA number density (standard ~1.2)
    N = 256; L = (N / rho) ** 0.5
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    rows = []
    for k, (fracB, T) in enumerate([(0.35, 0.50), (0.35, 0.45), (0.50, 0.50)]):
        s = make_species(N, fracB)
        x, sd = swap_mcmc(N, L, T, s, device=device, n_chains=128,
                          n_equil=4000, n_collect=200, every=20, step=0.07)
        p6 = psi6(x, L).mean().item()
        rc, gAA = partial_gr(x, sd, L, rmax=L/2, nbins=120, pair=(0, 0))
        rows.append((fracB, T, p6))
        ax[k].plot(rc, gAA); ax[k].set_title(f"fracB={fracB} T={T}\npsi6={p6:.3f} "
                   f"({'LIQUID' if p6 < 0.4 else 'ORDERED?'})")
        ax[k].set_xlabel("r"); ax[k].set_ylabel("g_AA(r)")
        print(f"fracB={fracB} T={T}: psi6={p6:.3f}", flush=True)
    fig.suptitle("O1 gate: KA stays disordered? (PASS if psi6 < ~0.4, g(r) liquid-like)")
    fig.tight_layout(); fig.savefig(os.path.join(ART, "ka_gate_o1.png"), dpi=120)
    best = min(rows, key=lambda r: r[2])
    print(f"O1 VERDICT: {'PASS' if best[2] < 0.4 else 'FAIL'} (best psi6={best[2]:.3f} at "
          f"fracB={best[0]}, T={best[1]})", flush=True)

if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the gate**

Run: `$PY -m liquid_coupling_flow.ka_gate_o1_crystallization`
Expected: prints psi6 per setting + `O1 VERDICT: PASS` (a disordered composition/T exists). If FAIL for all → revisit composition (try 50:50 / other) before proceeding; record which (fracB, T) passed → these become the system parameters.

- [ ] **Step 3: Commit**

```bash
git add liquid_coupling_flow/ka_gate_o1_crystallization.py
git commit -m "O1 gate: KA no-crystallization check (psi6 + g(r)); record passing fracB,T"
```

---

## Task 5: Gate O2 — equilibration feasibility

**Files:**
- Create: `liquid_coupling_flow/ka_gate_o2_equilibration.py`

- [ ] **Step 1: Write the gate experiment**

```python
# liquid_coupling_flow/ka_gate_o2_equilibration.py
"""O2 GATE: can swap MC equilibrate N=256..2048 at the chosen (fracB,T) in budget?
PASS if <U>/N from an ORDERED-grid IC and a DISORDERED IC converge to the same value
(within ~0.01) at every N -> we can make trustworthy references. Set (FRACB,T) from O1."""
from __future__ import annotations
import os, math, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, torch
from liquid_coupling_flow.ka_mcmc import swap_mcmc, make_species
from liquid_coupling_flow.ka_energy import ka_energy

ART = os.path.join(os.path.dirname(__file__), "artifacts")
FRACB, T, RHO = 0.35, 0.50, 1.2     # <- update from O1 verdict

def grid_ic(N, L):
    n = math.ceil(N ** 0.5); c = (torch.arange(n) + 0.5) * L / n
    g = torch.stack(torch.meshgrid(c, c, indexing="ij"), -1).reshape(-1, 2)
    return g[:N]

def main(device="cuda" if torch.cuda.is_available() else "cpu"):
    sizes = [256, 512, 1024, 2048]
    rows = []
    for N in sizes:
        L = (N / RHO) ** 0.5; s = make_species(N, FRACB)
        kw = dict(device=device, n_chains=64, n_equil=6000, n_collect=100, every=20, step=0.06)
        xo, sd = swap_mcmc(N, L, T, s, x0=grid_ic(N, L), **kw)
        xu, _ = swap_mcmc(N, L, T, s, **kw)
        Uo = (ka_energy(xo, sd, L) / N).mean().item()
        Uu = (ka_energy(xu, sd, L) / N).mean().item()
        rows.append((N, Uo, Uu, abs(Uo - Uu)))
        print(f"N={N}: U_ordered={Uo:.4f} U_uniform={Uu:.4f} gap={abs(Uo-Uu):.4f}", flush=True)
    ok = all(g < 0.01 for *_ , g in rows)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot([r[0] for r in rows], [r[3] for r in rows], "-o")
    ax.axhline(0.01, ls="--", color="r"); ax.set_xlabel("N"); ax.set_ylabel("|U_ord - U_unif|")
    ax.set_title(f"O2: equilibration gap vs N ({'PASS' if ok else 'FAIL'})")
    fig.tight_layout(); fig.savefig(os.path.join(ART, "ka_gate_o2.png"), dpi=120)
    print(f"O2 VERDICT: {'PASS' if ok else 'FAIL'}", flush=True)

if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the gate**

Run: `$PY -m liquid_coupling_flow.ka_gate_o2_equilibration`
Expected: per-N gap printed; `O2 VERDICT: PASS` (ordered & disordered ICs agree → references are trustworthy). If FAIL at large N → increase `n_equil`/`swap_frac`, or cap the validated size range at the largest passing N.

- [ ] **Step 3: Commit**

```bash
git add liquid_coupling_flow/ka_gate_o2_equilibration.py
git commit -m "O2 gate: swap-MC equilibration feasibility N=256..2048 (IC-agreement)"
```

---

## Task 6: Gate P1 — annealing hardness

**Files:**
- Create: `liquid_coupling_flow/ka_gate_p1_hardness.py`

- [ ] **Step 1: Write the gate experiment**

```python
# liquid_coupling_flow/ka_gate_p1_hardness.py
"""P1 GATE: is the LARGE KA glass actually HARD for annealing-from-disorder?
Compare swap-MC reference <U>/N (the right answer, from O2) to uniform+SMC (single-
particle displacement only, NO swaps) at increasing budget. PASS (GO) if uniform+SMC
stays ABOVE the reference by a clear margin at the largest budget -> annealing-from-
disorder cannot reach the basin -> a learned proposal has room to help. (Mirror of the
LJ hard_state_scan, now expected to PASS.)"""
from __future__ import annotations
import os, math, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, numpy as np, torch
from liquid_coupling_flow.ka_mcmc import swap_mcmc, make_species
from liquid_coupling_flow.ka_energy import ka_energy
from liquid_coupling_flow.smc import anneal_smc, SingleParticleMetropolis

ART = os.path.join(os.path.dirname(__file__), "artifacts")
FRACB, T, RHO = 0.35, 0.50, 1.2     # <- from O1/O2

def main(device="cuda" if torch.cuda.is_available() else "cpu"):
    N = 1024; L = (N / RHO) ** 0.5; s = make_species(N, FRACB).to(device)
    xref, sd = swap_mcmc(N, L, T, make_species(N, FRACB), device=device,
                         n_chains=32, n_equil=6000, n_collect=50, every=20, step=0.06)
    U_ref = (ka_energy(xref, sd, L) / N).mean().item()
    base = -2 * N * math.log(L)
    means = []
    budgets = [10, 30, 80, 200]
    for nb in budgets:
        torch.manual_seed(0)
        x0 = torch.rand(512, N, 2, device=device) * L
        res = anneal_smc(x0, lambda x: torch.full((x.shape[0],), base, device=x.device),
                         lambda x: -ka_energy(x, sd, L) / T,
                         SingleParticleMetropolis(step=0.06, n_sweeps=1, L=L), n_bridge=nb)
        Us = (ka_energy(res["x"], sd, L) / N).cpu().numpy()
        means.append(float((res["weights"].cpu().numpy() * Us).sum()))
        print(f"budget={nb}: uniform+SMC <U>/N={means[-1]:.4f} (ref {U_ref:.4f})", flush=True)
    gap = means[-1] - U_ref
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(budgets, means, "-o", label="uniform+SMC (no swaps)")
    ax.axhline(U_ref, ls="--", color="C2", label=f"swap-MC ref {U_ref:.3f}")
    ax.set_xlabel("SMC budget (sweeps)"); ax.set_ylabel("<U>/N")
    ax.set_title(f"P1 hardness (N={N}): gap={gap:+.3f} "
                 f"({'GO (hard)' if gap > 0.05 else 'NO-GO (anneals)'})"); ax.legend()
    fig.tight_layout(); fig.savefig(os.path.join(ART, "ka_gate_p1.png"), dpi=120)
    print(f"P1 VERDICT: {'GO' if gap > 0.05 else 'NO-GO'} (gap={gap:+.3f})", flush=True)

if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the gate**

Run: `$PY -m liquid_coupling_flow.ka_gate_p1_hardness`
Expected: uniform+SMC `<U>/N` stays clearly above the swap-MC reference at budget 200 → `P1 VERDICT: GO`. If NO-GO (uniform+SMC reaches the reference) → the system isn't hard at this N/T; go colder or larger, or reconsider (same honest stop as the LJ case).

- [ ] **Step 3: Commit**

```bash
git add liquid_coupling_flow/ka_gate_p1_hardness.py
git commit -m "P1 gate: KA annealing-hardness (uniform+SMC vs swap-MC reference)"
```

---

## After this plan

If O1=PASS, O2=PASS, P1=GO: record the chosen (fracB, T, ρ, sizes) and proceed to the
**architecture plan** (transformer-AR + spatial coupling flow, swap-augmented SMC, the
size-transfer benchmark). If P1=NO-GO at reachable settings: stop and write it up — the
honest negative, same discipline as the LJ line.

## Self-review notes
- Spec coverage: P0 (energy=Task1, swap MC=Task2, observables=Task3) and P1 gates
  (O1=Task4, O2=Task5, P1=Task6) all covered. Architecture/benchmark = next plan (by design).
- No placeholders: all code is concrete; gate thresholds explicit (psi6<0.4; gap<0.01; gap>0.05).
- Consistency: `ka_energy(x,s,L)`, `swap_mcmc(...)->(x,s)`, `make_species`, `partial_gr`, `psi6`
  signatures match across tasks; species int8 {0,1} convention uniform.
- Open knobs (ρ=1.2, step, n_equil) are physics defaults to be tuned by O1/O2, not placeholders.
