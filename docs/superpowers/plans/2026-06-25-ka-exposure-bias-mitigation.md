# KA Exposure-Bias Mitigation (Soft Labels + Scheduled Sampling) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Mitigate exposure bias in the causal `KALocalFrameModel` AR generator via (primary) geometric soft labels + stochastic target sampling and (secondary) GapDiff self-conditioned scheduled sampling, measured in-distribution at N=100.

**Architecture:** Both features are **training-only** and live in **new subclass files**; the validated base model (`ka_localframe.py`) — and therefore the inference path `sample()`/`log_prob()` — is **never edited**, so the model's exactness gate (`sample(return_logq=True) == log_prob`) is preserved by construction. A shared position+species NLL helper is reused by both the soft-label loss and the scheduled-sampling loss so the two compose for the combined arm.

**Tech Stack:** Python, PyTorch, pytest. Existing modules: `liquid_coupling_flow/ka_localframe.py` (base model), `ka_energy.py`, `ka_observables.py` (`partial_gr`), `ka_gridformer_train.py` (`augment`, `overlap_frac`).

## Global Constraints

- Target model: `liquid_coupling_flow.ka_localframe.KALocalFrameModel` (causal, geometry-invariant). Defaults `rho=1.2, n_bins=192, knn=16, arc_range=3.0`; `bin_w = 2*arc_range/n_bins`; `arc_scale(N) = N**(1/6)`; `d = 2`.
- **Inference invariant:** never edit `ka_localframe.py`. All new code is in new files (`ka_exposure_lf.py`, `ka_softlabel.py`, `ka_sched.py`, `ka_exposure_campaign.py`) and `liquid_coupling_flow/tests/`.
- **No-op at trivial config:** soft loss with `soft=False, stochastic=False` must equal the base one-hot NLL; scheduled loss with `p_keep=1.0` must equal the base NLL — both bit-exact (atol 1e-4).
- Regime/verdict: **in-distribution N=100, T\*=0.5**. Co-primary win-gate: absolute free-run `g_BB` peak↑ + spurious `g(r<0.88)`↓ + FR clash level↓ (mean and closure `j>80`), **and** the FR→TF drift gap. Reference data `g_BB` peak ≈ 2.40.
- Checkpoints in `liquid_coupling_flow/artifacts/`: baseline `ka_localframe_N100_20k.pt`; failed-noise null `ka_localframe_ss_N100.pt`; reference `ka_reference_N100.pt`.
- Tests: `liquid_coupling_flow/tests/test_*.py`, plain pytest functions + a `__main__` runner, CPU, artifact-free (random-weight small model, synthetic configs).
- Species labels are NOT softened (discrete/non-geometric); only the `(a,b)` position head is.

---

### Task 1: Local-frame exposure diagnostic + baseline measurement

**Files:**
- Create: `liquid_coupling_flow/ka_exposure_lf.py`
- Test: `liquid_coupling_flow/tests/test_exposure_lf.py`

**Interfaces:**
- Produces:
  - `clash_by_index(cand, prefix, L, thr=0.7) -> np.ndarray[N]` — per-index fraction of particle `j` within `thr` of any predecessor `0..j-1` of `prefix`.
  - `tf_fr_by_index(m, N, device, B=512) -> dict` with keys `tf[N], fr[N], gbb_tf=(peak,spur), gbb_fr=(peak,spur), gbb_data=(peak,spur)`.
  - `main(device)` — loads baseline + noise-null checkpoints, prints early/mid/late clash, saves a figure.

- [ ] **Step 1: Write the failing test**

```python
# liquid_coupling_flow/tests/test_exposure_lf.py
import numpy as np, torch
from liquid_coupling_flow.ka_exposure_lf import clash_by_index


def test_clash_by_index_known_geometry():
    # 3 particles on a line at x=0,0.5,5.0 in a big box: j=1 clashes with j=0 (d=0.5<0.7),
    # j=2 clashes with neither (min dist 4.5). One batch element.
    L = 50.0
    pos = torch.tensor([[[0.0, 0.0], [0.5, 0.0], [5.0, 0.0]]])
    out = clash_by_index(pos, pos, L, thr=0.7)
    assert out.shape == (3,)
    assert out[0] == 0.0          # j=0 has no predecessor
    assert out[1] == 1.0          # 0.5 < 0.7
    assert out[2] == 0.0          # 4.5 > 0.7


def test_clash_by_index_pbc_wrap():
    # j=1 at 49.9, predecessor at 0.0; min-image distance is 0.1 (<0.7) across the boundary.
    L = 50.0
    pos = torch.tensor([[[0.0, 0.0], [49.9, 0.0]]])
    out = clash_by_index(pos, pos, L, thr=0.7)
    assert out[1] == 1.0


if __name__ == "__main__":
    test_clash_by_index_known_geometry()
    test_clash_by_index_pbc_wrap()
    print("EXPOSURE-LF TESTS PASSED")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest liquid_coupling_flow/tests/test_exposure_lf.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'liquid_coupling_flow.ka_exposure_lf'`.

- [ ] **Step 3: Write the diagnostic module**

```python
# liquid_coupling_flow/ka_exposure_lf.py
"""Per-index TEACHER-FORCED vs FREE-RUN exposure-bias diagnostic for the causal local-frame model.
TF: place every particle j from the TRUE prefix (model heads on _local context); clash j vs the TRUE
0..j-1. FR: the model's own AR rollout (sample); clash j vs its OWN 0..j-1. The TF->FR gap, growing with
curve index j (worst at closure j>80), IS the exposure bias. Co-reports g_BB peak + spurious(<0.88)."""
from __future__ import annotations
import os, torch, torch.nn.functional as F, numpy as np, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from liquid_coupling_flow.ka_localframe import KALocalFrameModel
from liquid_coupling_flow.ka_observables import partial_gr

ART = os.path.join(os.path.dirname(__file__), "artifacts")


def clash_by_index(cand, prefix, L, thr=0.7):
    """cand [B,N,2] (particle j), prefix [B,N,2] (the 0..j-1 to check j against). Returns clash[N]."""
    N = cand.shape[1]; out = np.zeros(N)
    for j in range(1, N):
        df = cand[:, j:j + 1] - prefix[:, :j]; df = df - L * torch.round(df / L)
        out[j] = float(((df ** 2).sum(-1).min(1).values.sqrt() < thr).float().mean())
    return out


def _gbb(pos, sp, L):
    rc, g = partial_gr(pos, sp, L, min(L / 2, 4.0), 100, (1, 1))
    return float(g.max()), float(g[rc < 0.88].mean())


@torch.no_grad()
def _tf_place(m, xo, so, L, N, device):
    """Teacher-forced placement of every particle j from the TRUE prefix context."""
    sc = m.geo._scaffold(N, device); arc = m._arc_scale(N); B = xo.shape[0]
    context, origin = m._local(xo, so, sc, L, N)
    ba = torch.multinomial(F.softmax(m.head_a(context).reshape(-1, m.n_bins), -1), 1).reshape(B, N)
    bb = torch.multinomial(F.softmax(m.head_b(context + m.bin_a_emb(ba)).reshape(-1, m.n_bins), -1), 1).reshape(B, N)
    a = m._bin_center(ba) + (torch.rand(B, N, device=device) - 0.5) * m.bin_w
    bc = m._bin_center(bb) + (torch.rand(B, N, device=device) - 0.5) * m.bin_w
    return torch.remainder(origin + torch.stack([a, bc], -1) * arc, L)


@torch.no_grad()
def tf_fr_by_index(m, N, device, B=512):
    ref = torch.load(os.path.join(ART, f"ka_reference_N{N}.pt"), map_location=device, weights_only=False)
    s, L, data = ref["s"].to(device).long(), ref["L"], ref["x"].to(device)[:B]
    nB = int((s == 1).sum()) if s.dim() == 1 else int((s[0] == 1).sum())
    order = m.geo._curve_order(data, N)
    xo = torch.gather(data, 1, order[..., None].expand(-1, -1, 2))
    so = torch.gather(s.expand(B, N).clone() if s.dim() == 1 else s[:B], 1, order)
    tf = clash_by_index(_tf_place(m, xo, so, L, N, device), xo, L)                    # TF: vs TRUE prefix
    xg, sg = m.sample(B, N, n_B=nB, device=device)
    og = m.geo._curve_order(xg, N)
    xgo = torch.gather(xg, 1, og[..., None].expand(-1, -1, 2)); sgo = torch.gather(sg, 1, og)
    fr = clash_by_index(xgo, xgo, L)                                                  # FR: vs OWN prefix
    sd = s.expand(B, N) if s.dim() == 1 else s[:B]
    return {"tf": tf, "fr": fr, "L": L,
            "gbb_tf": _gbb(_tf_place(m, xo, so, L, N, device), so, L),
            "gbb_fr": _gbb(xg, sg, L), "gbb_data": _gbb(data, sd, L)}


def _load(ckpt, device):
    ck = torch.load(os.path.join(ART, ckpt), map_location=device, weights_only=False)
    m = KALocalFrameModel(rho=ck["rho"], n_bins=ck["n_bins"], knn=ck["knn"],
                          head_mode=ck.get("head_mode", "factorized")).to(device)
    m.load_state_dict(ck["state_dict"], strict=False); m.eval()
    return m


@torch.no_grad()
def main(N=100, device="cuda" if torch.cuda.is_available() else "cpu"):
    arms = [("baseline (one-hot)", "ka_localframe_N100_20k.pt", "C0"),
            ("noise-null (failed)", "ka_localframe_ss_N100.pt", "C7")]
    fig, ax = plt.subplots(1, 1, figsize=(8, 5)); rows = []
    for tag, ckpt, col in arms:
        if not os.path.exists(os.path.join(ART, ckpt)):
            print(f"SKIP {tag}: {ckpt} missing", flush=True); continue
        d = tf_fr_by_index(_load(ckpt, device), N, device); jj = np.arange(1, N)
        early = d["fr"][1:N // 3].mean(); mid = d["fr"][N // 3:2 * N // 3].mean(); late = d["fr"][2 * N // 3:].mean()
        print(f"{tag:22s}: FR clash early {early:.3f} mid {mid:.3f} late(closure) {late:.3f} | "
              f"TF mean {d['tf'][1:].mean():.3f} FR mean {d['fr'][1:].mean():.3f} gap {d['fr'][1:].mean()-d['tf'][1:].mean():.3f} | "
              f"g_BB peak TF {d['gbb_tf'][0]:.2f} FR {d['gbb_fr'][0]:.2f} data {d['gbb_data'][0]:.2f} | "
              f"spur FR {d['gbb_fr'][1]:.3f} data {d['gbb_data'][1]:.3f}", flush=True)
        ax.plot(jj, d["tf"][1:], col, ls="--", lw=1.5, label=f"{tag} TF")
        ax.plot(jj, d["fr"][1:], col, lw=2, label=f"{tag} FR")
        rows.append((tag, d))
    ax.set_xlabel("curve index j"); ax.set_ylabel("fresh-clash rate"); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    ax.set_title(f"Exposure bias (local-frame): TF vs FR clash by index, N={N}")
    out = os.path.join(ART, f"ka_exposure_lf_N{N}.png"); fig.tight_layout(); fig.savefig(out, dpi=120)
    print(f"saved {out}", flush=True)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest liquid_coupling_flow/tests/test_exposure_lf.py -q`
Expected: PASS (2 passed).

- [ ] **Step 5: Run the baseline measurement (records the numbers every later arm must beat)**

Run: `python -m liquid_coupling_flow.ka_exposure_lf`
Expected: two printed rows (baseline + noise-null) with `FR clash early/mid/late`, `TF/FR gap`, `g_BB peak/spur`; a saved figure `artifacts/ka_exposure_lf_N100.png`. Record the **baseline** FR clash mean, closure mean, FR g_BB peak, and FR spurious — these are the gate references. Confirm the **noise-null** does not beat baseline (the documented failure).

- [ ] **Step 6: Commit**

```bash
git add liquid_coupling_flow/ka_exposure_lf.py liquid_coupling_flow/tests/test_exposure_lf.py
git commit -m "feat(ka): local-frame TF-vs-FR exposure diagnostic + baseline measurement"
```

---

### Task 2: Geometric soft-label objective (Feature B core, TDD)

**Files:**
- Create: `liquid_coupling_flow/ka_softlabel.py`
- Test: `liquid_coupling_flow/tests/test_softlabel.py`

**Interfaces:**
- Produces:
  - `soft_target(centers, target, tau) -> Tensor[..., n_bins]` — normalized `softmax(-(centers - target)^2 / tau)`.
  - `pos_species_nll(m, context, origin, xo, so, L, N, *, sigma_bins, soft, stochastic, canonical, gen) -> Tensor[B]` — per-config position+species NLL; reused by Task 4/6.
  - `KALocalFrameSoft(KALocalFrameModel)` with `train_loss(x, s, *, sigma_bins, soft, stochastic, canonical=None, gen=None) -> Tensor[B]`.
- Consumes: `KALocalFrameModel._local, .head_a, .head_b, .bin_a_emb, .head_species, ._bin, ._bin_center, .bin_w, .n_bins, ._arc_scale, .geo` (all from Task-0 base, unmodified).

- [ ] **Step 1: Write the failing tests**

```python
# liquid_coupling_flow/tests/test_softlabel.py
import math, torch
from liquid_coupling_flow.ka_localframe import KALocalFrameModel
from liquid_coupling_flow.ka_softlabel import soft_target, KALocalFrameSoft


def _tiny(seed=0):
    torch.manual_seed(seed)
    m = KALocalFrameSoft(rho=1.2, n_bins=192, knn=16); m.eval()
    N, B = 12, 2
    x = torch.rand(B, N, 2) * m._Lof(N)
    s = (torch.rand(B, N) < 0.35).long()
    return m, x, s, N, B


def test_soft_target_normalized_and_peaks_at_containing_bin():
    m, *_ = _tiny()
    centers = m._bin_center(torch.arange(m.n_bins))
    tgt = torch.tensor([-1.234, 0.0, 2.0])
    P = soft_target(centers, tgt, tau=(0.7 * m.bin_w) ** 2)
    assert torch.allclose(P.sum(-1), torch.ones(3), atol=1e-5)          # normalized
    assert torch.equal(P.argmax(-1), m._bin(tgt))                       # argmax == containing bin


def test_trivial_config_parity_with_base_log_prob():
    m, x, s, N, B = _tiny(1)
    d = 2; vol = d * N * math.log(m.bin_w); jac = d * N * math.log(m._arc_scale(N))
    loss = m.train_loss(x, s, sigma_bins=1.0, soft=False, stochastic=False)   # NLL per config
    base = m.log_prob(x, s)                                                   # logp per config
    assert torch.allclose(loss, -(base + vol + jac), atol=1e-4)


def test_inference_exactness_gate_preserved():
    m, _, _, N, B = _tiny(2)
    pos, sp, logq = m.sample(B, N, n_B=4, device="cpu", return_logq=True)
    assert torch.allclose(logq, m.log_prob(pos, sp), atol=1e-4)


def test_soft_loss_is_finite_and_differentiable():
    m, x, s, N, B = _tiny(3)
    for soft, stoch in [(True, False), (False, True), (True, True)]:
        g = torch.Generator().manual_seed(7)
        loss = m.train_loss(x, s, sigma_bins=1.0, soft=soft, stochastic=stoch, gen=g).mean()
        assert torch.isfinite(loss)
        loss.backward(); assert m.head_a.weight.grad is not None
        m.zero_grad()


if __name__ == "__main__":
    test_soft_target_normalized_and_peaks_at_containing_bin()
    test_trivial_config_parity_with_base_log_prob()
    test_inference_exactness_gate_preserved()
    test_soft_loss_is_finite_and_differentiable()
    print("SOFTLABEL TESTS PASSED")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest liquid_coupling_flow/tests/test_softlabel.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'liquid_coupling_flow.ka_softlabel'`.

- [ ] **Step 3: Write the soft-label module**

```python
# liquid_coupling_flow/ka_softlabel.py
"""Feature B: GEOMETRIC SOFT LABELS + stochastic target sampling for the (a,b) head of the local-frame AR
generator. The head bins a continuous coordinate into n_bins fine bins; a one-hot single-sample target
teaches a spiky conditional and treats physically-near bins as equally wrong as far bins. Soft labels =
kernel-density smoothing of the target (P(bin_i) ~ exp(-(c_i-target)^2/tau)); stochastic target sampling
draws the assigned bin from that kernel. TRAINING-ONLY: base log_prob/sample are untouched, so the
exactness gate is preserved. Species labels are NOT softened (discrete)."""
from __future__ import annotations
import os, time, math, torch, torch.nn.functional as F
from liquid_coupling_flow.ka_localframe import KALocalFrameModel, _wrap_pm, KNN

ART = os.path.join(os.path.dirname(__file__), "artifacts")


def soft_target(centers, target, tau):
    """centers [n_bins], target [...] -> normalized soft categorical [..., n_bins]."""
    d2 = (centers.view(*([1] * target.dim()), -1) - target.unsqueeze(-1)) ** 2
    return F.softmax(-d2 / tau, dim=-1)


def _axis_loss(logp, centers, target, tau, soft, stochastic, gen):
    """logp [B,N,n_bins] log-softmax head; returns (axis_nll[B,N], assigned_bin[B,N]).
    `assigned` is the bin used to condition the b-head (sampled if stochastic, else the containing bin).
    The loss is soft cross-entropy if `soft`, else hard CE at `assigned`."""
    bw = centers[1] - centers[0]                                   # == m.bin_w
    P = soft_target(centers, target, tau) if (soft or stochastic) else None
    if stochastic:
        assigned = torch.multinomial(P.reshape(-1, P.shape[-1]), 1, generator=gen).reshape(target.shape)
    else:                                                          # containing bin == m._bin(target) exactly
        assigned = ((target - (centers[0] - bw / 2)) / bw).long().clamp(0, centers.numel() - 1)
    if soft:
        nll = -(P * logp).sum(-1)
    else:
        nll = -logp.gather(-1, assigned.unsqueeze(-1)).squeeze(-1)
    return nll, assigned


def pos_species_nll(m, context, origin, xo, so, L, N, *, sigma_bins, soft, stochastic, canonical, gen):
    """Per-config position+species NLL given a precomputed context/origin (from TRUE or MIXED prefix).
    Reused by the scheduled-sampling arm. Returns [B]."""
    if soft or stochastic:
        assert sigma_bins > 0, "sigma_bins must be > 0 when soft or stochastic labeling is on"
    tau = (sigma_bins * m.bin_w) ** 2
    centers = m._bin_center(torch.arange(m.n_bins, device=context.device))
    ab = _wrap_pm(xo - origin, L) / m._arc_scale(N)                # normalized target (a,b)
    la = F.log_softmax(m.head_a(context), -1)
    nll_a, asg_a = _axis_loss(la, centers, ab[..., 0], tau, soft, stochastic, gen)
    lb = F.log_softmax(m.head_b(context + m.bin_a_emb(asg_a)), -1)
    nll_b, _ = _axis_loss(lb, centers, ab[..., 1], tau, soft, stochastic, gen)
    s_logits = m.head_species(context)
    use_canon = m.canonical if canonical is None else canonical
    if use_canon:
        oh = F.one_hot(so, m.n_species).to(s_logits.dtype)
        rem = oh.sum(1, keepdim=True) - (oh.cumsum(1) - oh)
        s_logits = s_logits.masked_fill(rem <= 0, float("-inf"))
    lp_s = F.log_softmax(s_logits, -1).gather(-1, so[..., None]).squeeze(-1)
    return (nll_a + nll_b - lp_s).sum(1)                           # [B]


class KALocalFrameSoft(KALocalFrameModel):
    def train_loss(self, x, s, *, sigma_bins, soft, stochastic, canonical=None, gen=None):
        B, N = x.shape[0], x.shape[1]; s = s.long()
        s = s.expand(B, N).clone() if s.dim() == 1 else s
        L = self._Lof(N)
        order = self.geo._curve_order(x, N)
        xo = torch.gather(x, 1, order[..., None].expand(-1, -1, 2)); so = torch.gather(s, 1, order)
        sc = self.geo._scaffold(N, x.device)
        context, origin = self._local(xo, so, sc, L, N)
        return pos_species_nll(self, context, origin, xo, so, L, N, sigma_bins=sigma_bins,
                               soft=soft, stochastic=stochastic, canonical=canonical, gen=gen)
```

Note on `_axis_loss`: the containing-bin recomputation uses the uniform bin grid (`bw = centers[1]-centers[0]`, `lo = centers[0]-bw/2`) which reproduces `m._bin` exactly (the test `test_soft_target_normalized_and_peaks_at_containing_bin` and the parity test guard this).

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest liquid_coupling_flow/tests/test_softlabel.py -q`
Expected: PASS (4 passed). If `test_trivial_config_parity` fails, the containing-bin recomputation in `_axis_loss` disagrees with `m._bin`; fix `lo`/`bw` to match `_bin` = `((v+arc_range)/bin_w).long().clamp(...)`.

- [ ] **Step 5: Commit**

```bash
git add liquid_coupling_flow/ka_softlabel.py liquid_coupling_flow/tests/test_softlabel.py
git commit -m "feat(ka): geometric soft-label + stochastic target objective (training-only, parity-tested)"
```

---

### Task 3: Soft-label training + RQ 4-way ablation + τ-sweep

**Files:**
- Modify: `liquid_coupling_flow/ka_softlabel.py` (append `train` + `run_ablation` + `__main__`)

**Interfaces:**
- Consumes: `KALocalFrameSoft.train_loss`, `augment` (`ka_gridformer_train`), `tf_fr_by_index` (`ka_exposure_lf`).
- Produces: checkpoints `artifacts/ka_softlabel_N100_<cell>.pt`; printed diagnostic per cell.

- [ ] **Step 1: Append the training + ablation runner**

```python
# --- append to liquid_coupling_flow/ka_softlabel.py ---

def train(cell, sigma_bins, train_N=100, steps=10000, warm="ka_localframe_N100_20k.pt",
          device="cuda" if torch.cuda.is_available() else "cpu"):
    """cell in {'neither','soft','stochastic','both'}. Warm-start fine-tune (fair vs baseline)."""
    from liquid_coupling_flow.ka_gridformer_train import augment
    soft = cell in ("soft", "both"); stochastic = cell in ("stochastic", "both")
    ref = torch.load(os.path.join(ART, f"ka_reference_N{train_N}.pt"), map_location=device, weights_only=False)
    data, sp, L = ref["x"].to(device), ref["s"].to(device).long(), ref["L"]; N = data.shape[1]
    m = KALocalFrameSoft(rho=1.2, n_bins=192, knn=KNN).to(device)
    ck = torch.load(os.path.join(ART, warm), map_location=device, weights_only=False)
    m.load_state_dict(ck["state_dict"], strict=False); m.train()
    gen = torch.Generator(device=device).manual_seed(0)
    print(f"SOFT-LABEL train cell={cell} sigma_bins={sigma_bins} steps={steps} warm={warm}", flush=True)
    opt = torch.optim.AdamW(m.parameters(), lr=2e-4, weight_decay=1e-4); B, t0 = 128, time.time()
    for step in range(steps):
        idx = torch.randint(0, data.shape[0], (B,), device=device)
        loss = (m.train_loss(augment(data[idx], L), sp, sigma_bins=sigma_bins,
                             soft=soft, stochastic=stochastic, gen=gen) / N).mean()
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 5.0); opt.step()
        if step % 1000 == 0:
            print(f"  step {step:5d} nll/N {loss.item():.3f} {time.time()-t0:.0f}s", flush=True)
    tag = f"ka_softlabel_N{train_N}_{cell}_s{sigma_bins}.pt"
    torch.save({"state_dict": m.state_dict(), "rho": 1.2, "n_bins": 192, "knn": KNN,
                "cell": cell, "sigma_bins": sigma_bins, "step": steps}, os.path.join(ART, tag))
    print(f"saved {tag}", flush=True); return tag


def run_ablation(train_N=100, steps=10000, sigma_bins=1.0,
                 device="cuda" if torch.cuda.is_available() else "cpu"):
    from liquid_coupling_flow.ka_exposure_lf import tf_fr_by_index, _load
    base = tf_fr_by_index(_load("ka_localframe_N100_20k.pt", device), train_N, device)
    base_fr, base_peak, base_spur = base["fr"][1:].mean(), base["gbb_fr"][0], base["gbb_fr"][1]
    print(f"BASELINE: FR clash {base_fr:.3f} g_BB peak {base_peak:.2f} spur {base_spur:.3f} "
          f"(data peak {base['gbb_data'][0]:.2f})", flush=True)
    for cell in ("neither", "soft", "stochastic", "both"):
        sb = 0.0 if cell == "neither" else sigma_bins
        tag = train(cell, sb, train_N=train_N, steps=steps, device=device)
        ck = torch.load(os.path.join(ART, tag), map_location=device, weights_only=False)
        m = KALocalFrameSoft(rho=1.2, n_bins=192, knn=KNN).to(device)
        m.load_state_dict(ck["state_dict"]); m.eval()
        d = tf_fr_by_index(m, train_N, device)
        flag = "  <-- OVER-SMOOTH (FR clash up)" if d["fr"][1:].mean() > base_fr + 1e-3 else ""
        print(f"CELL {cell:10s}: FR clash {d['fr'][1:].mean():.3f} (closure {d['fr'][2*train_N//3:].mean():.3f}) "
              f"g_BB peak {d['gbb_fr'][0]:.2f} spur {d['gbb_fr'][1]:.3f} | gap {d['fr'][1:].mean()-d['tf'][1:].mean():.3f}{flag}",
              flush=True)


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "ablation":
        run_ablation(steps=int(sys.argv[2]) if len(sys.argv) > 2 else 10000,
                     sigma_bins=float(sys.argv[3]) if len(sys.argv) > 3 else 1.0)
    else:
        train(sys.argv[1] if len(sys.argv) > 1 else "both",
              float(sys.argv[2]) if len(sys.argv) > 2 else 1.0)
```

- [ ] **Step 2: Smoke-run the runner (cheap, 200 steps) to verify plumbing**

Run: `python -c "from liquid_coupling_flow.ka_softlabel import train; train('both', 1.0, steps=200)"`
Expected: prints decreasing `nll/N`, saves `artifacts/ka_softlabel_N100_both_s1.0.pt`, no exceptions.

- [ ] **Step 3: Run the full 4-way ablation**

Run: `python -m liquid_coupling_flow.ka_softlabel ablation 10000 1.0`
Expected: a BASELINE line then four CELL lines. **Gate:** `soft`/`both` should lower FR clash and/or raise g_BB peak vs BASELINE without raising spurious; any cell with the `OVER-SMOOTH` flag is not promoted. Per RQ, `soft`-alone may underperform — record honestly.

- [ ] **Step 4: τ-sweep on the best ablation cell**

Run (substitute `<best>` = winning cell, e.g. `both`):
```bash
for sb in 0.5 1.0 2.0 4.0; do python -c "from liquid_coupling_flow.ka_softlabel import train; from liquid_coupling_flow.ka_exposure_lf import tf_fr_by_index; import torch; t=train('<best>', $sb, steps=10000); ck=torch.load('liquid_coupling_flow/artifacts/'+t,weights_only=False); from liquid_coupling_flow.ka_softlabel import KALocalFrameSoft,KNN; m=KALocalFrameSoft(rho=1.2,n_bins=192,knn=KNN).cuda(); m.load_state_dict(ck['state_dict']); m.eval(); d=tf_fr_by_index(m,100,'cuda'); print('sigma_bins=$sb FR',round(float(d['fr'][1:].mean()),3),'gBB',round(d['gbb_fr'][0],2),'spur',round(d['gbb_fr'][1],3))"; done
```
Expected: a clear interior optimum in `sigma_bins` (too small → no smoothing benefit; too large → over-smoothing raises FR clash). Record the winning `(cell, sigma_bins)` as `B*`.

- [ ] **Step 5: Commit**

```bash
git add liquid_coupling_flow/ka_softlabel.py
git commit -m "feat(ka): soft-label training + RQ 4-way ablation + tau-sweep runner"
```

---

### Task 4: Scheduled-sampling objective (Feature A core, TDD)

**Files:**
- Create: `liquid_coupling_flow/ka_sched.py`
- Test: `liquid_coupling_flow/tests/test_sched.py`

**Interfaces:**
- Produces:
  - `arc_pT(progress, r=2.0, floor=0.5) -> float` — GapDiff arc anneal; `arc_pT(0)=1`, decays to `floor`.
  - `KALocalFrameSched(KALocalFrameModel)` with `log_prob_sched(x, s, p_keep, *, sigma_bins=0.0, soft=False, stochastic=False, canonical=None, gen=None) -> Tensor[B]` (training log-prob; NLL = `-log_prob_sched/N`).
- Consumes: `pos_species_nll` (Task 2), base `_local`, heads, `_bin_center`, `bin_w`, `_arc_scale`.

- [ ] **Step 1: Write the failing tests**

```python
# liquid_coupling_flow/tests/test_sched.py
import math, torch
from liquid_coupling_flow.ka_sched import arc_pT, KALocalFrameSched


def _tiny(seed=0):
    torch.manual_seed(seed)
    m = KALocalFrameSched(rho=1.2, n_bins=192, knn=16); m.eval()
    N, B = 12, 2
    x = torch.rand(B, N, 2) * m._Lof(N); s = (torch.rand(B, N) < 0.35).long()
    return m, x, s, N, B


def test_arc_pT_schedule():
    assert abs(arc_pT(0.0) - 1.0) < 1e-9
    assert abs(arc_pT(1.0, r=2.0, floor=0.5) - 0.5) < 1e-9          # floored
    assert abs(arc_pT(0.5, r=2.0, floor=0.0) - math.sqrt(3) / 2) < 1e-6
    assert arc_pT(0.3) >= arc_pT(0.6)                              # monotone non-increasing


def test_p_keep_one_parity_with_base_log_prob():
    m, x, s, N, B = _tiny(1)
    d = 2; vol = d * N * math.log(m.bin_w); jac = d * N * math.log(m._arc_scale(N))
    torch.manual_seed(123)
    lp = m.log_prob_sched(x, s, p_keep=1.0)                        # no self-conditioning
    assert torch.allclose(lp, m.log_prob(x, s) + vol + jac, atol=1e-4)


def test_inference_exactness_gate_preserved():
    m, _, _, N, B = _tiny(2)
    pos, sp, logq = m.sample(B, N, n_B=4, device="cpu", return_logq=True)
    assert torch.allclose(logq, m.log_prob(pos, sp), atol=1e-4)


def test_self_prefix_is_detached():
    m, x, s, N, B = _tiny(3)
    s2 = s.expand(B, N).clone() if s.dim() == 1 else s
    order = m.geo._curve_order(x, N)
    xo = torch.gather(x, 1, order[..., None].expand(-1, -1, 2)); so = torch.gather(s2, 1, order)
    xmix = m._self_prefix(xo, so, m.geo._scaffold(N, x.device), m._Lof(N), N, p_keep=0.0)
    assert not xmix.requires_grad


if __name__ == "__main__":
    test_arc_pT_schedule()
    test_p_keep_one_parity_with_base_log_prob()
    test_inference_exactness_gate_preserved()
    test_self_prefix_is_detached()
    print("SCHED TESTS PASSED")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest liquid_coupling_flow/tests/test_sched.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'liquid_coupling_flow.ka_sched'`.

- [ ] **Step 3: Write the scheduled-sampling module**

```python
# liquid_coupling_flow/ka_sched.py
"""Feature A: GapDiff self-conditioned SCHEDULED SAMPLING for the local-frame AR generator. Distinct from
the failed noise version (ka_localframe_ss): we feed the model's OWN one-step placements (detached) into
the conditioning prefix, not isotropic Gaussian noise. Two-pass detached (no backprop-through-chain):
pass 1 (no_grad) places every particle from the TRUE prefix -> x_hat; build a per-particle Bernoulli(1-p_T)
mix of x_hat/true -> xmix; pass 2 recomputes context/origin from xmix and scores the TRUE target. Arc
anneal p_T(progress)=max(sqrt(r^2-(r*progress)^2)/r, floor). TRAINING-ONLY; base log_prob/sample untouched."""
from __future__ import annotations
import os, time, math, torch, torch.nn.functional as F
from liquid_coupling_flow.ka_localframe import KALocalFrameModel, KNN
from liquid_coupling_flow.ka_softlabel import pos_species_nll

ART = os.path.join(os.path.dirname(__file__), "artifacts")


def arc_pT(progress, r=2.0, floor=0.5):
    return max(math.sqrt(max(r * r - (r * progress) ** 2, 0.0)) / r, floor)


class KALocalFrameSched(KALocalFrameModel):
    @torch.no_grad()
    def _self_prefix(self, xo, so, sc, L, N, p_keep):
        """Pass 1: place every j from the TRUE prefix; per-particle Bernoulli(1-p_keep) mix -> detached xmix."""
        B = xo.shape[0]; arc = self._arc_scale(N)
        context, origin = self._local(xo, so, sc, L, N)
        ba = torch.multinomial(F.softmax(self.head_a(context).reshape(-1, self.n_bins), -1), 1).reshape(B, N)
        bb = torch.multinomial(F.softmax(self.head_b(context + self.bin_a_emb(ba)).reshape(-1, self.n_bins), -1), 1).reshape(B, N)
        a = self._bin_center(ba) + (torch.rand(B, N, device=xo.device) - 0.5) * self.bin_w
        bc = self._bin_center(bb) + (torch.rand(B, N, device=xo.device) - 0.5) * self.bin_w
        x_hat = torch.remainder(origin + torch.stack([a, bc], -1) * arc, L)
        keep = torch.rand(B, N, device=xo.device) < p_keep                  # True -> keep TRUE position
        return torch.where(keep[..., None], xo, x_hat).detach()

    def log_prob_sched(self, x, s, p_keep, *, sigma_bins=0.0, soft=False, stochastic=False,
                       canonical=None, gen=None):
        B, N = x.shape[0], x.shape[1]; s = s.long()
        s = s.expand(B, N).clone() if s.dim() == 1 else s
        L = self._Lof(N); sc = self.geo._scaffold(N, x.device)
        order = self.geo._curve_order(x, N)
        xo = torch.gather(x, 1, order[..., None].expand(-1, -1, 2)); so = torch.gather(s, 1, order)
        xmix = self._self_prefix(xo, so, sc, L, N, p_keep)                  # detached drifted prefix
        context, origin = self._local(xmix, so, sc, L, N)                  # grad path (pass 2)
        nll = pos_species_nll(self, context, origin, xo, so, L, N, sigma_bins=sigma_bins,
                              soft=soft, stochastic=stochastic, canonical=canonical, gen=gen)
        return -nll                                                        # log_prob (variable part)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest liquid_coupling_flow/tests/test_sched.py -q`
Expected: PASS (4 passed). The parity test relies on `p_keep=1.0` ⇒ `keep` all True ⇒ `xmix == xo` ⇒ pass-2 context equals base; `pos_species_nll` with `soft=False,stochastic=False` equals base one-hot NLL (guaranteed by Task 2's parity test).

- [ ] **Step 5: Commit**

```bash
git add liquid_coupling_flow/ka_sched.py liquid_coupling_flow/tests/test_sched.py
git commit -m "feat(ka): GapDiff scheduled-sampling objective (own-placement, two-pass detached, parity-tested)"
```

---

### Task 5: Scheduled-sampling training run

**Files:**
- Modify: `liquid_coupling_flow/ka_sched.py` (append `train` + `__main__`)

**Interfaces:**
- Consumes: `KALocalFrameSched.log_prob_sched`, `arc_pT`, `augment`, `tf_fr_by_index`.
- Produces: checkpoint `artifacts/ka_sched_N100.pt` + printed diagnostic.

- [ ] **Step 1: Append the training runner**

```python
# --- append to liquid_coupling_flow/ka_sched.py ---

def train(train_N=100, steps=10000, r=2.0, floor=0.5, sigma_bins=0.0, soft=False, stochastic=False,
          warm="ka_localframe_N100_20k.pt", device="cuda" if torch.cuda.is_available() else "cpu"):
    from liquid_coupling_flow.ka_gridformer_train import augment
    ref = torch.load(os.path.join(ART, f"ka_reference_N{train_N}.pt"), map_location=device, weights_only=False)
    data, sp, L = ref["x"].to(device), ref["s"].to(device).long(), ref["L"]; N = data.shape[1]
    m = KALocalFrameSched(rho=1.2, n_bins=192, knn=KNN).to(device)
    ck = torch.load(os.path.join(ART, warm), map_location=device, weights_only=False)
    m.load_state_dict(ck["state_dict"], strict=False); m.train()
    gen = torch.Generator(device=device).manual_seed(0)
    print(f"SCHEDULED-SAMPLING train steps={steps} r={r} floor={floor} soft={soft} stoch={stochastic} warm={warm}", flush=True)
    opt = torch.optim.AdamW(m.parameters(), lr=2e-4, weight_decay=1e-4); B, t0 = 128, time.time()
    for step in range(steps):
        p_keep = arc_pT(step / max(1, steps - 1), r=r, floor=floor)
        idx = torch.randint(0, data.shape[0], (B,), device=device)
        loss = (-m.log_prob_sched(augment(data[idx], L), sp, p_keep, sigma_bins=sigma_bins,
                                  soft=soft, stochastic=stochastic, gen=gen) / N).mean()
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 5.0); opt.step()
        if step % 1000 == 0:
            print(f"  step {step:5d} p_keep {p_keep:.3f} nll/N {loss.item():.3f} {time.time()-t0:.0f}s", flush=True)
    torch.save({"state_dict": m.state_dict(), "rho": 1.2, "n_bins": 192, "knn": KNN,
                "r": r, "floor": floor, "step": steps}, os.path.join(ART, "ka_sched_N100.pt"))
    print("saved ka_sched_N100.pt", flush=True)


if __name__ == "__main__":
    import sys
    train(steps=int(sys.argv[1]) if len(sys.argv) > 1 else 10000)
```

- [ ] **Step 2: Smoke-run (200 steps) to verify plumbing**

Run: `python -c "from liquid_coupling_flow.ka_sched import train; train(steps=200)"`
Expected: prints `p_keep` decreasing from ~1.0 toward 0.5, decreasing `nll/N`, saves `artifacts/ka_sched_N100.pt`.

- [ ] **Step 3: Full scheduled-sampling run + evaluate**

Run:
```bash
python -m liquid_coupling_flow.ka_sched 10000
python -c "from liquid_coupling_flow.ka_exposure_lf import tf_fr_by_index; from liquid_coupling_flow.ka_sched import KALocalFrameSched,KNN; import torch; ck=torch.load('liquid_coupling_flow/artifacts/ka_sched_N100.pt',weights_only=False); m=KALocalFrameSched(rho=1.2,n_bins=192,knn=KNN).cuda(); m.load_state_dict(ck['state_dict']); m.eval(); d=tf_fr_by_index(m,100,'cuda'); import numpy as np; print('SCHED FR clash',round(float(d['fr'][1:].mean()),3),'closure',round(float(d['fr'][2*100//3:].mean()),3),'gap',round(float(d['fr'][1:].mean()-d['tf'][1:].mean()),3),'gBB',round(d['gbb_fr'][0],2),'spur',round(d['gbb_fr'][1],3))"
```
Expected: FR→TF **gap** drops vs the Task-1 baseline (the drift-specific win). **Escalation rule (spec §5.2):** if early/mid clash drops but the **closure** (`j>80`) clash survives, that signals the one-step substitution under-represents accumulated drift → escalate to the A2 full-rollout variant (separate follow-up; not in this plan).

- [ ] **Step 4: Commit**

```bash
git add liquid_coupling_flow/ka_sched.py
git commit -m "feat(ka): scheduled-sampling training run + arc-anneal schedule"
```

---

### Task 6: Combine (best-B + A), final figure, results report

**Files:**
- Create: `liquid_coupling_flow/ka_exposure_campaign.py`
- Create: `reports/2026-06-25-ka-exposure-bias-results.md`

**Interfaces:**
- Consumes: `KALocalFrameSched.train` (with `soft=True` + best `sigma_bins`), `tf_fr_by_index`, `_load`, `clash_by_index`.
- Produces: figure `artifacts/ka_exposure_campaign_N100.png`; the results report.

- [ ] **Step 1: Train the combined arm (scheduled prefix + soft-label loss)**

Run (substitute `<best_sb>` from Task 3 τ-sweep, e.g. 1.0):
```bash
python -c "from liquid_coupling_flow.ka_sched import train; train(steps=10000, soft=True, stochastic=True, sigma_bins=<best_sb>)"
```
This reuses `log_prob_sched`'s `soft`/`stochastic` passthrough into `pos_species_nll` — the two features compose in one objective. Rename the checkpoint so it is not overwritten:
```bash
mv liquid_coupling_flow/artifacts/ka_sched_N100.pt liquid_coupling_flow/artifacts/ka_combined_N100.pt
```

- [ ] **Step 2: Write the campaign figure script**

```python
# liquid_coupling_flow/ka_exposure_campaign.py
"""Final campaign figure: TF/FR clash-by-index + g_BB(peak,spurious) for all arms vs the data reference."""
from __future__ import annotations
import os, numpy as np, torch, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from liquid_coupling_flow.ka_exposure_lf import tf_fr_by_index, _load

ART = os.path.join(os.path.dirname(__file__), "artifacts")
ARMS = [("baseline", "ka_localframe_N100_20k.pt", "C0"),
        ("noise-null", "ka_localframe_ss_N100.pt", "C7"),
        ("+B soft(best)", None, "C2"),      # set ckpt name from Task 3 winner, e.g. ka_softlabel_N100_both_s1.0.pt
        ("+A sched", "ka_sched_N100.pt", "C1"),
        ("+A+B", "ka_combined_N100.pt", "C3")]


def main(N=100, soft_ckpt="ka_softlabel_N100_both_s1.0.pt", device="cuda" if torch.cuda.is_available() else "cpu"):
    fig, ax = plt.subplots(1, 2, figsize=(15, 5.2)); jj = np.arange(1, N); peaks = []
    for tag, ckpt, col in ARMS:
        ckpt = soft_ckpt if tag == "+B soft(best)" else ckpt
        if ckpt is None or not os.path.exists(os.path.join(ART, ckpt)):
            print(f"SKIP {tag}: {ckpt} missing", flush=True); continue
        d = tf_fr_by_index(_load(ckpt, device), N, device)
        ax[0].plot(jj, d["fr"][1:], col, lw=2, label=f"{tag} FR (gap {d['fr'][1:].mean()-d['tf'][1:].mean():.3f})")
        peaks.append((tag, d["gbb_fr"][0], d["gbb_fr"][1], d["gbb_data"][0]))
        print(f"{tag:14s}: FR clash {d['fr'][1:].mean():.3f} closure {d['fr'][2*N//3:].mean():.3f} "
              f"g_BB peak {d['gbb_fr'][0]:.2f} spur {d['gbb_fr'][1]:.3f}", flush=True)
    ax[0].set_xlabel("curve index j"); ax[0].set_ylabel("FR clash"); ax[0].legend(fontsize=8); ax[0].grid(alpha=0.3)
    ax[0].set_title(f"Free-run clash by index (N={N})")
    names = [p[0] for p in peaks]; xs = np.arange(len(names))
    ax[1].bar(xs - 0.2, [p[1] for p in peaks], 0.4, label="g_BB peak (FR)")
    ax[1].axhline(peaks[0][3], color="k", ls="--", lw=1, label="data peak")
    ax[1].bar(xs + 0.2, [p[2] for p in peaks], 0.4, label="spurious(<0.88)")
    ax[1].set_xticks(xs); ax[1].set_xticklabels(names, rotation=20, fontsize=8); ax[1].legend(fontsize=8)
    ax[1].set_title("g_BB peak + spurious vs data")
    out = os.path.join(ART, f"ka_exposure_campaign_N{N}.png"); fig.tight_layout(); fig.savefig(out, dpi=120)
    print(f"saved {out}", flush=True)


if __name__ == "__main__":
    import sys
    main(soft_ckpt=sys.argv[1] if len(sys.argv) > 1 else "ka_softlabel_N100_both_s1.0.pt")
```

- [ ] **Step 3: Run the campaign figure**

Run: `python -m liquid_coupling_flow.ka_exposure_campaign <best_soft_ckpt_name>`
Expected: one printed row per arm + saved `artifacts/ka_exposure_campaign_N100.png`.

- [ ] **Step 4: Write the results report (fill with the actual measured numbers)**

Create `reports/2026-06-25-ka-exposure-bias-results.md` with: the baseline vs each-arm table (FR clash mean + closure, FR→TF gap, g_BB peak, spurious), the τ-sweep curve, and the **co-primary verdict** per the spec. Include the **honest-framing** call: explicitly label where the remaining gap is **drift** (addressable here — did A close it?) vs the **residual/half-cage floor** (not addressable by A or B; needs the energy/full-cage corrector, [[full-cage-lever-needs-energy]]). State the **A1→A2 escalation decision** from Task 5 (did closure drift survive?).

- [ ] **Step 5: Commit**

```bash
git add liquid_coupling_flow/ka_exposure_campaign.py reports/2026-06-25-ka-exposure-bias-results.md
git commit -m "feat(ka): combined arm + campaign figure + exposure-bias results report"
```

---

## Self-Review

**Spec coverage:**
- §2 inference invariant → enforced by never editing `ka_localframe.py`; exactness-gate tests in Tasks 2 & 4. ✓
- §3 co-primary win-gate (absolute FR g_BB/clash + FR→TF gap) → Task 1 diagnostic + Tasks 3/5/6 reporting. ✓
- §4 Feature B (soft labels, stochastic sampling, τ in bins, 4-way ablation + τ-sweep, over-smoothing flag) → Tasks 2 & 3. ✓
- §4.1 trivial-config no-op → `test_trivial_config_parity` (Task 2). ✓
- §5 Feature A (own-placement two-pass detached, arc anneal, A2 escalation rule) → Tasks 4 & 5. ✓
- §6 phases / figure for {baseline, noise-null, +B, +A, +A+B} → Task 6. ✓
- §7 checks: exactness gate (T2/T4), trivial-config parity (T2/T4), soft-target normalization (T2), inference-untouched (by construction: no edits to base). ✓
- §8 scope: size-transfer and A2 are explicitly out / staged (Task 5 escalation note). ✓
- §9 deliverables: diagnostic, soft-label training, scheduled training, figure + report — all present. ✓

**Placeholder scan:** `<best>`, `<best_sb>`, `<best_soft_ckpt_name>` are run-time values substituted from the Task-3 ablation winner — each has an explicit source and example value, not an unfilled blank. No "TBD"/"add error handling"/"similar to Task N". ✓

**Type consistency:** `pos_species_nll` signature is identical where consumed (Task 2 `train_loss`, Task 4 `log_prob_sched`). `tf_fr_by_index` returns the same dict keys used in Tasks 1/3/5/6. `arc_pT(progress, r, floor)` consistent. `_load` reused from Task 1. Checkpoint dict keys (`state_dict, rho, n_bins, knn`) match the base loader. ✓

**Note for the implementer (containing-bin convention):** the base `_bin(v) = ((v+arc_range)/bin_w).long().clamp(0, n_bins-1)`. In `_axis_loss`, `lo = centers[0] - bw/2` must equal `-arc_range` and `bw = bin_w`; if `test_trivial_config_parity` (Task 2) fails by a one-bin offset, reconcile `_axis_loss`'s containing-bin formula with `_bin` exactly before proceeding.
