# KA Local-Frame Output-Head Comparison (Phase A) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add three exact, geometry-invariant single-particle output heads (H0 factorized / H1 species→position / H2 joint (s,a)) to the local-frame model and compare them on a k-step partial-rollout yardstick at N=36/100/256.

**Architecture:** Extend `KALocalFrameModel` with a `head_mode` switch that changes only the categorical structure over (species, a-bin) — the `(a,b)->x` coordinate change (and thus exact likelihood) is identical across modes. A shared `_sample_head` method DRYs the per-mode sampling so `sample()` and the new k-step rollout harness use one code path. A new eval module rolls the model out for k free steps from a true prefix and scores structure-vs-k.

**Tech Stack:** PyTorch, the existing `liquid_coupling_flow` package (`ka_localframe.py`, `ka_gridformer.py`, `ka_observables.py`), matplotlib. No pytest suite exists for these modules; validations are runnable assertion scripts executed with `python`, asserting numeric properties (round-trip error, sample↔log_prob self-consistency, exact composition, regression equality).

## Global Constraints

- **Exact likelihood** in every head mode: per-particle `(a,b)->x` Jacobian unchanged; each head is a proper normalized categorical. Self-consistency gate: `logq` accumulated during `sample()` must equal `log_prob()` of that sample (up to atol 1e-3 per particle).
- **Geometry-invariant conditioning**: heads read only `context` from `_local`/`_step` (frame-relative neighbours + species); no curve positional encoding may be added.
- **Canonical species composition** must remain exact (generated configs have exactly `n_B` B-particles).
- **Evaluate at N=36, 100, 256** (trained N=100; held-out 36/256). `rho=1.2`.
- **Backward compatible**: `head_mode="factorized"` reproduces current H0 numbers (N=100 TF-clash ~0.09, FR overlap ~0.27) and loads existing checkpoint `ka_localframe_N100_20k.pt`.
- Scratch dir for temp validation scripts: `/tmp/claude-1002/-mnt-ssd-GridTransformer/2f102e4c-a99a-4c8d-9094-d110ef077179/scratchpad` (referred to as `$SP`).

---

## File Structure

- **Modify** `liquid_coupling_flow/ka_localframe.py`
  - `KALocalFrameModel.__init__`: add `head_mode` arg, `self.d_model`, new head modules (`sp_out_emb`, `head_sa`), persist nothing here.
  - new `KALocalFrameModel._sample_head(self, h, rem)` -> `(sj, ba, bb)` — per-mode sampling (DRY).
  - `KALocalFrameModel.log_prob`: branch the (species, a, b) log-prob by `head_mode`.
  - `KALocalFrameModel.sample`: refactor to call `_sample_head`; add `return_logq` flag.
  - `train()`: add `head_mode`, `knn`, `bf16` knobs; save `head_mode`/`knn` in checkpoints.
- **Create** `liquid_coupling_flow/ka_localframe_kstep.py`
  - `rollout_window(m, pos_true, sp_true, p, k, N, L)` -> generated `(pos, sp)`.
  - `kstep_metrics(...)` -> clash & g_BB(r<0.88) vs k.
  - `heldout_nll(m, N)` -> exact NLL/N.
  - `compare(modes, sizes, ks)` -> driver that trains/loads H0/H1/H2 and emits comparison plots + table.

---

## Task 1: `head_mode` plumbing + DRY sampling + exactness gate (H0 only)

**Files:**
- Modify: `liquid_coupling_flow/ka_localframe.py` (`__init__` ~L69-86, `sample` ~L150-171, `log_prob` ~L130-148)
- Test: `$SP/t1_h0.py`

**Interfaces:**
- Consumes: existing `KALocalFrameModel` (`head_a`, `head_b`, `bin_a_emb`, `head_species`, `sp_emb`, `_local`, `_step`).
- Produces:
  - `KALocalFrameModel(head_mode="factorized"|"species_pos"|"joint_sa", ...)` with `self.d_model:int`, `self.head_mode:str`, `self.sp_out_emb:nn.Embedding`, `self.head_sa:nn.Linear`.
  - `KALocalFrameModel._sample_head(self, h:Tensor[B,d], rem:Tensor[B,n_species]|None) -> (sj:Long[B], ba:Long[B], bb:Long[B])`.
  - `KALocalFrameModel.sample(B,N,n_B,device,return_logq=False) -> (pos,sp)` or `(pos,sp,logq:Tensor[B])`.

- [ ] **Step 1: Write the failing exactness/regression test**

```python
# $SP/t1_h0.py
import torch, math, torch.nn.functional as F
from liquid_coupling_flow.ka_localframe import KALocalFrameModel
dev = "cuda" if torch.cuda.is_available() else "cpu"
m = KALocalFrameModel(rho=1.2, n_bins=192, knn=16, head_mode="factorized").to(dev).eval()
assert hasattr(m, "d_model") and hasattr(m, "_sample_head") and hasattr(m, "sp_out_emb") and hasattr(m, "head_sa")
torch.manual_seed(0)
pos, sp, logq = m.sample(8, 100, n_B=39, device=dev, return_logq=True)
assert int((sp == 1).sum(1).unique().item()) == 39, "canonical composition broken"
lp = m.log_prob(pos, sp)                      # exact density of the drawn sample
assert torch.isfinite(logq).all() and torch.isfinite(lp).all()
err = (logq - lp).abs().max().item() / 100    # per-particle
print(f"H0 sample<->log_prob per-particle |err| {err:.2e}")
assert err < 1e-3, "sample logq != log_prob (exactness self-consistency FAILED)"
print("T1 PASS")
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd /mnt/ssd/GridTransformer && python $SP/t1_h0.py`
Expected: FAIL — `AttributeError` (no `_sample_head`/`sp_out_emb`/`head_sa`) or `sample()` has no `return_logq`.

- [ ] **Step 3: Add `__init__` members**

In `KALocalFrameModel.__init__`, add the `head_mode` arg and after `self.canonical = ...; self.d = 2; ...` line, store/build:

```python
        self.head_mode = head_mode            # "factorized" | "species_pos" | "joint_sa"
        self.d_model = d_model
        self.sp_out_emb = nn.Embedding(n_species, d_model)         # particle's OWN species -> position (H1/H2)
        self.head_sa = nn.Linear(d_model, n_species * n_bins)      # joint (species, a-bin) logits (H2)
```

Update the signature: `def __init__(self, rho=1.2, n_bins=192, d_model=192, n_head=6, n_layer=4, cell_size=0.285, n_species=2, arc_range=3.0, knn=KNN, canonical=True, head_mode="factorized"):`

- [ ] **Step 4: Add the shared `_sample_head` method (factorized branch)**

Add this method to `KALocalFrameModel`:

```python
    def _sample_head(self, h, rem):
        """Per-mode sampling of (species, a-bin, b-bin) from per-particle context h [B,d].
        rem [B,n_species] = remaining species budget (canonical); None disables the mask."""
        B = h.shape[0]
        def mask_species(logits):
            return logits if rem is None else logits.masked_fill(rem <= 0, float("-inf"))
        if self.head_mode == "factorized":
            sj = torch.multinomial(F.softmax(mask_species(self.head_species(h)), -1), 1).squeeze(-1)
            ba = torch.multinomial(F.softmax(self.head_a(h), -1), 1).squeeze(-1)
            bb = torch.multinomial(F.softmax(self.head_b(h + self.bin_a_emb(ba)), -1), 1).squeeze(-1)
        elif self.head_mode == "species_pos":
            sj = torch.multinomial(F.softmax(mask_species(self.head_species(h)), -1), 1).squeeze(-1)
            e = self.sp_out_emb(sj)
            ba = torch.multinomial(F.softmax(self.head_a(h + e), -1), 1).squeeze(-1)
            bb = torch.multinomial(F.softmax(self.head_b(h + e + self.bin_a_emb(ba)), -1), 1).squeeze(-1)
        elif self.head_mode == "joint_sa":
            joint = self.head_sa(h).reshape(B, self.n_species, self.n_bins)
            if rem is not None:
                joint = joint.masked_fill((rem <= 0)[..., None], float("-inf"))
            flat = torch.multinomial(F.softmax(joint.reshape(B, -1), -1), 1).squeeze(-1)
            sj, ba = flat // self.n_bins, flat % self.n_bins
            e = self.sp_out_emb(sj)
            bb = torch.multinomial(F.softmax(self.head_b(h + e + self.bin_a_emb(ba)), -1), 1).squeeze(-1)
        else:
            raise ValueError(self.head_mode)
        return sj, ba, bb
```

- [ ] **Step 5: Refactor `sample` to use `_sample_head` + `return_logq`**

Replace the body of the `for j in range(N):` loop in `sample` and the signature:

```python
    @torch.no_grad()
    def sample(self, B, N, n_B=None, device=None, return_logq=False):
        L = self._Lof(N); sc = self.geo._scaffold(N, device); arc = self._arc_scale(N)
        pos = torch.zeros(B, N, 2, device=device); sp = torch.zeros(B, N, dtype=torch.long, device=device)
        logq = torch.zeros(B, device=device)
        rem = None
        if n_B is not None:
            rem = torch.zeros(B, self.n_species, device=device); rem[:, 0] = N - n_B; rem[:, 1] = n_B
        for j in range(N):
            h, origin = self._step(pos, sp, sc[j], j, L)
            sj, ba, bb = self._sample_head(h, rem)
            if return_logq:
                logq += self._logq_head(h, rem, sj, ba, bb)
            if rem is not None:
                rem[torch.arange(B, device=device), sj] -= 1
            a = self._bin_center(ba) + (torch.rand(B, device=device) - 0.5) * self.bin_w
            b = self._bin_center(bb) + (torch.rand(B, device=device) - 0.5) * self.bin_w
            pos[:, j] = torch.remainder(origin + torch.stack([a, b], -1) * arc, L)
            sp[:, j] = sj
        if return_logq:
            vol = self.d * N * math.log(self.bin_w); jac = self.d * N * math.log(self._arc_scale(N))
            return pos, sp, logq - vol - jac
        return pos, sp
```

Add the matching `_logq_head` helper (mode-aware discrete log-prob of the sampled bins; mirrors `_sample_head`):

```python
    def _logq_head(self, h, rem, sj, bb_a, bb_b):
        B = h.shape[0]
        def msk(lg):
            return lg if rem is None else lg.masked_fill(rem <= 0, float("-inf"))
        if self.head_mode == "factorized":
            ls = F.log_softmax(msk(self.head_species(h)), -1).gather(-1, sj[:, None]).squeeze(-1)
            la = F.log_softmax(self.head_a(h), -1).gather(-1, bb_a[:, None]).squeeze(-1)
            lb = F.log_softmax(self.head_b(h + self.bin_a_emb(bb_a)), -1).gather(-1, bb_b[:, None]).squeeze(-1)
            return ls + la + lb
        if self.head_mode == "species_pos":
            ls = F.log_softmax(msk(self.head_species(h)), -1).gather(-1, sj[:, None]).squeeze(-1)
            e = self.sp_out_emb(sj)
            la = F.log_softmax(self.head_a(h + e), -1).gather(-1, bb_a[:, None]).squeeze(-1)
            lb = F.log_softmax(self.head_b(h + e + self.bin_a_emb(bb_a)), -1).gather(-1, bb_b[:, None]).squeeze(-1)
            return ls + la + lb
        # joint_sa
        joint = self.head_sa(h).reshape(B, self.n_species, self.n_bins)
        if rem is not None:
            joint = joint.masked_fill((rem <= 0)[..., None], float("-inf"))
        lsa = F.log_softmax(joint.reshape(B, -1), -1).reshape(B, self.n_species, self.n_bins)
        lsa = lsa[torch.arange(B), sj, bb_a]
        e = self.sp_out_emb(sj)
        lb = F.log_softmax(self.head_b(h + e + self.bin_a_emb(bb_a)), -1).gather(-1, bb_b[:, None]).squeeze(-1)
        return lsa + lb
```

- [ ] **Step 6: Run to verify it passes**

Run: `cd /mnt/ssd/GridTransformer && python $SP/t1_h0.py`
Expected: PASS — prints `H0 sample<->log_prob per-particle |err| <1e-3` then `T1 PASS`.

- [ ] **Step 7: Commit**

```bash
git add liquid_coupling_flow/ka_localframe.py
git commit -m "feat(localframe): head_mode plumbing + DRY _sample_head + return_logq exactness gate (H0)"
```

---

## Task 2: H1 species→position in `log_prob`

**Files:**
- Modify: `liquid_coupling_flow/ka_localframe.py` (`log_prob` ~L130-148)
- Test: `$SP/t2_h1.py`

**Interfaces:**
- Consumes: `_sample_head`/`_logq_head` (already handle `species_pos`).
- Produces: `log_prob` correct for `head_mode="species_pos"` (matches `_logq_head` densities).

- [ ] **Step 1: Write the failing test**

```python
# $SP/t2_h1.py
import torch
from liquid_coupling_flow.ka_localframe import KALocalFrameModel
dev = "cuda" if torch.cuda.is_available() else "cpu"
m = KALocalFrameModel(rho=1.2, n_bins=192, knn=16, head_mode="species_pos").to(dev).eval()
torch.manual_seed(0)
pos, sp, logq = m.sample(8, 100, n_B=39, device=dev, return_logq=True)
lp = m.log_prob(pos, sp)
err = (logq - lp).abs().max().item() / 100
print(f"H1 sample<->log_prob per-particle |err| {err:.2e}")
assert err < 1e-3, "H1 log_prob != sample logq"
print("T2 PASS")
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd /mnt/ssd/GridTransformer && python $SP/t2_h1.py`
Expected: FAIL — `log_prob` still computes the factorized density, so `|err|` is large (≫1e-3).

- [ ] **Step 3: Restructure `log_prob` to branch by mode**

Replace the head section of `log_prob` (between `ab = ... ; ba, bb = ...` and the `vol/jac` return) with:

```python
        ba, bb = self._bin(ab[..., 0]), self._bin(ab[..., 1])
        # canonical remaining-budget mask (per step j, given true prefix species so)
        oh = F.one_hot(so, self.n_species).to(context.dtype)
        rem = oh.sum(1, keepdim=True) - (oh.cumsum(1) - oh)        # [B,N,n_species]
        use_canon = self.canonical if canonical is None else canonical
        if self.head_mode in ("factorized", "species_pos"):
            s_logits = self.head_species(context)
            if use_canon:
                s_logits = s_logits.masked_fill(rem <= 0, float("-inf"))
            lp_s = F.log_softmax(s_logits, -1).gather(-1, so[..., None]).squeeze(-1)
            e = self.sp_out_emb(so) if self.head_mode == "species_pos" else 0.0
            la = F.log_softmax(self.head_a(context + e), -1)
            lb = F.log_softmax(self.head_b(context + e + self.bin_a_emb(ba)), -1)
            lp_pos = la.gather(-1, ba[..., None]).squeeze(-1) + lb.gather(-1, bb[..., None]).squeeze(-1)
            lp_step = lp_s + lp_pos
        else:  # joint_sa
            joint = self.head_sa(context).reshape(*context.shape[:2], self.n_species, self.n_bins)
            if use_canon:
                joint = joint.masked_fill((rem <= 0)[..., None], float("-inf"))
            lsa = F.log_softmax(joint.reshape(*context.shape[:2], -1), -1).reshape(
                *context.shape[:2], self.n_species, self.n_bins)
            lp_sa = lsa.gather(2, so[..., None, None].expand(-1, -1, 1, self.n_bins)).squeeze(2) \
                       .gather(-1, ba[..., None]).squeeze(-1)
            e = self.sp_out_emb(so)
            lb = F.log_softmax(self.head_b(context + e + self.bin_a_emb(ba)), -1).gather(-1, bb[..., None]).squeeze(-1)
            lp_step = lp_sa + lb
        vol = self.d * N * math.log(self.bin_w); jac = self.d * N * math.log(self._arc_scale(N))
        return lp_step.sum(1) - vol - jac
```

(Note: this single edit also implements Task 3's `joint_sa` branch; Task 3 only adds its test.)

- [ ] **Step 4: Run to verify it passes**

Run: `cd /mnt/ssd/GridTransformer && python $SP/t2_h1.py`
Expected: PASS — `H1 sample<->log_prob per-particle |err| <1e-3`, `T2 PASS`.

- [ ] **Step 5: Commit**

```bash
git add liquid_coupling_flow/ka_localframe.py
git commit -m "feat(localframe): H1 species->position head in log_prob (mode-branched)"
```

---

## Task 3: H2 joint (s,a) exactness + canonical composition

**Files:**
- Modify: none (the `joint_sa` branch landed in Task 2 Step 3)
- Test: `$SP/t3_h2.py`

**Interfaces:**
- Consumes: `log_prob`/`_sample_head`/`_logq_head` `joint_sa` branches.
- Produces: validated H2 (exactness + exact composition + decode invertibility `flat = s*n_bins + a`).

- [ ] **Step 1: Write the failing test**

```python
# $SP/t3_h2.py
import torch
from liquid_coupling_flow.ka_localframe import KALocalFrameModel
dev = "cuda" if torch.cuda.is_available() else "cpu"
m = KALocalFrameModel(rho=1.2, n_bins=192, knn=16, head_mode="joint_sa").to(dev).eval()
torch.manual_seed(0)
pos, sp, logq = m.sample(16, 100, n_B=39, device=dev, return_logq=True)
assert (sp == 1).sum(1).eq(39).all(), "H2 canonical composition not exactly n_B"
lp = m.log_prob(pos, sp)
err = (logq - lp).abs().max().item() / 100
print(f"H2 sample<->log_prob per-particle |err| {err:.2e}; comp OK")
assert err < 1e-3, "H2 log_prob != sample logq"
print("T3 PASS")
```

- [ ] **Step 2: Run to verify it fails (or passes if Task 2 landed the branch)**

Run: `cd /mnt/ssd/GridTransformer && python $SP/t3_h2.py`
Expected: PASS if Task 2 Step 3 is in; if it FAILs on `|err|`, the joint decode `sj, ba = flat // n_bins, flat % n_bins` and `log_prob` gather indices disagree — align both to `flat = sj * n_bins + ba`.

- [ ] **Step 3: (Only if failing) fix decode/gather alignment**

Ensure `_sample_head` joint branch uses `sj, ba = flat // self.n_bins, flat % self.n_bins` and `log_prob` indexes `lsa[..., so, ba]` via the `gather(2, ...).gather(-1, ...)` shown in Task 2. Both encode `flat = species * n_bins + a_bin`.

- [ ] **Step 4: Run to verify it passes**

Run: `cd /mnt/ssd/GridTransformer && python $SP/t3_h2.py`
Expected: PASS — `T3 PASS`.

- [ ] **Step 5: Commit**

```bash
git add liquid_coupling_flow/ka_localframe.py
git commit -m "test(localframe): H2 joint (s,a) exactness + exact composition gate"
```

---

## Task 4: Efficiency knobs in `train()` (KNN, bf16, head_mode, checkpoint fields)

**Files:**
- Modify: `liquid_coupling_flow/ka_localframe.py` (`train` ~L222-250)
- Test: `$SP/t4_train.py`

**Interfaces:**
- Consumes: `KALocalFrameModel`, `augment`.
- Produces: `train(train_N=100, steps=40000, device=..., head_mode="factorized", knn=8, bf16=True)` saving checkpoints to `ka_localframe_{head_mode}_N{train_N}.pt` with keys `state_dict, rho, n_bins, knn, head_mode, train_N, step`.

- [ ] **Step 1: Write the failing test**

```python
# $SP/t4_train.py
import inspect, torch
from liquid_coupling_flow import ka_localframe as M
sig = inspect.signature(M.train).parameters
assert {"head_mode", "knn", "bf16"} <= set(sig), "train() missing new knobs"
M.train(train_N=100, steps=6, head_mode="joint_sa", knn=8, bf16=torch.cuda.is_available())
import os; p = os.path.join(M.ART, "ka_localframe_joint_sa_N100.pt")
ck = torch.load(p, map_location="cpu", weights_only=False)
assert ck["head_mode"] == "joint_sa" and ck["knn"] == 8
print("T4 PASS")
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd /mnt/ssd/GridTransformer && python $SP/t4_train.py`
Expected: FAIL — `AssertionError: train() missing new knobs`.

- [ ] **Step 3: Update `train()` signature, model construction, autocast, and save paths**

```python
def train(train_N=100, steps=40000, device="cuda" if torch.cuda.is_available() else "cpu",
          head_mode="factorized", knn=8, bf16=True):
    import time
    from liquid_coupling_flow.ka_gridformer_train import augment
    ref = torch.load(os.path.join(ART, f"ka_reference_N{train_N}.pt"), map_location=device, weights_only=False)
    data, s, L = ref["x"].to(device), ref["s"].to(device).long(), ref["L"]
    m = KALocalFrameModel(rho=1.2, n_bins=192, knn=knn, head_mode=head_mode).to(device)
    tag = f"{head_mode}_N{train_N}"
    print(f"LOCAL-FRAME train [{head_mode}] @N={train_N}, knn={knn}, bf16={bf16}, "
          f"{sum(p.numel() for p in m.parameters())/1e6:.2f}M params, {steps} steps.", flush=True)
    opt = torch.optim.AdamW(m.parameters(), lr=3e-4, weight_decay=1e-4); B, t0 = 96, time.time()
    use_amp = bf16 and device == "cuda"
    def save(step):
        torch.save({"state_dict": m.state_dict(), "rho": 1.2, "n_bins": 192, "knn": knn,
                    "head_mode": head_mode, "train_N": train_N, "step": step},
                   os.path.join(ART, f"ka_localframe_{tag}.pt"))
    for step in range(steps):
        idx = torch.randint(0, data.shape[0], (B,), device=device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
            loss = -(m.log_prob(augment(data[idx], L), s) / train_N).mean()
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 5.0); opt.step()
        if step % 1000 == 0:
            print(f"  step {step:5d} -logq/N {loss.item():.3f} {time.time()-t0:.0f}s", flush=True)
        if (step + 1) % max(1, steps // 4) == 0:
            m.eval(); line = []
            for Nv in (36, 100, 256):
                try:
                    tf, rn = tf_clash(m, Nv, device); line.append(f"N{Nv} TF {tf:.3f}/rand {rn:.3f}")
                except RuntimeError as e:
                    torch.cuda.empty_cache(); line.append(f"N{Nv} SKIP({type(e).__name__})")
            torch.cuda.empty_cache(); m.train(); print(f"  >> step {step+1}: " + " | ".join(line), flush=True)
            save(step + 1); print(f"  [ckpt {tag} @ {step+1}]", flush=True)
    save(steps); print(f"saved ka_localframe_{tag}.pt", flush=True)
```

Also update the `__main__` block so `train` accepts the mode:
`if len(sys.argv) > 1 and sys.argv[1] == "train": train(steps=int(sys.argv[2]) if len(sys.argv) > 2 else 40000, head_mode=sys.argv[3] if len(sys.argv) > 3 else "factorized")`

- [ ] **Step 4: Run to verify it passes**

Run: `cd /mnt/ssd/GridTransformer && python $SP/t4_train.py`
Expected: PASS — short run prints step logs, writes `ka_localframe_joint_sa_N100.pt`, `T4 PASS`.

- [ ] **Step 5: Commit**

```bash
git add liquid_coupling_flow/ka_localframe.py
git commit -m "feat(localframe): train() head_mode/knn/bf16 knobs + mode-tagged checkpoints"
```

---

## Task 5: k-step partial-rollout eval harness

**Files:**
- Create: `liquid_coupling_flow/ka_localframe_kstep.py`
- Test: `$SP/t5_kstep.py`

**Interfaces:**
- Consumes: `KALocalFrameModel` (`_step`, `_sample_head`, `_bin_center`, `bin_w`, `_arc_scale`, `geo`), `ka_localframe._wrap_pm`.
- Produces:
  - `rollout_window(m, pos_true, sp_true, p, k, N, L) -> (pos, sp)` (atoms `p..p+k-1` free-run from true prefix).
  - `kstep_metrics(m, N, ks, device, B=256) -> dict{k: (clash_frac, gbb_smallr)}`.
  - `heldout_nll(m, N, device, B=256) -> float` (NLL/N on data).

- [ ] **Step 1: Write the failing test (k=1 must reproduce the 1-step diagnostic ballpark)**

```python
# $SP/t5_kstep.py
import torch
from liquid_coupling_flow.ka_localframe import KALocalFrameModel
from liquid_coupling_flow.ka_localframe_kstep import kstep_metrics, heldout_nll
dev = "cuda" if torch.cuda.is_available() else "cpu"
ck = torch.load("liquid_coupling_flow/artifacts/ka_localframe_N100_20k.pt", map_location=dev, weights_only=False)
m = KALocalFrameModel(rho=ck["rho"], n_bins=ck["n_bins"], knn=ck["knn"],
                      head_mode=ck.get("head_mode", "factorized")).to(dev).eval()
m.load_state_dict(ck["state_dict"])
met = kstep_metrics(m, 100, ks=[1, 4, 16], device=dev)
print("kstep:", {k: (round(v[0], 3), round(v[1], 3)) for k, v in met.items()})
assert met[1][0] <= met[16][0] + 1e-6, "clash must be monotone non-decreasing in k"
nll = heldout_nll(m, 100, dev); print("NLL/N", round(nll, 3))
assert met[1][0] < 0.2, "k=1 clash should be small (near the 1-step TF regime)"
print("T5 PASS")
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd /mnt/ssd/GridTransformer && python $SP/t5_kstep.py`
Expected: FAIL — `ModuleNotFoundError: ka_localframe_kstep`.

- [ ] **Step 3: Create the harness**

```python
# liquid_coupling_flow/ka_localframe_kstep.py
"""k-step partial-rollout eval: seed the TRUE prefix to index p, free-run k steps, score the generated
atoms vs data as a function of k (drift dial). Primary yardstick for comparing output-head variants;
generalizes the 1-step teacher-forced diagnostic (k=1) toward full free-run (k=N)."""
from __future__ import annotations
import os, math, torch, torch.nn.functional as F
from liquid_coupling_flow.ka_localframe import KALocalFrameModel, _wrap_pm
ART = os.path.join(os.path.dirname(__file__), "artifacts")


@torch.no_grad()
def rollout_window(m, pos_true, sp_true, p, k, N, L):
    """Copy true config; overwrite atoms p..p+k-1 with the model's free-run (each conditions on the
    drifting prefix pos[:, :j]). Canonical budget seeded from the true prefix counts."""
    B = pos_true.shape[0]; dev = pos_true.device
    pos, sp = pos_true.clone(), sp_true.clone()
    sc = m.geo._scaffold(N, dev); arc = m._arc_scale(N)
    tot = F.one_hot(sp_true, m.n_species).sum(1).float()                  # [B,n_species] total budget
    pre = F.one_hot(sp_true[:, :p], m.n_species).sum(1).float() if p > 0 else torch.zeros_like(tot)
    rem = tot - pre
    for j in range(p, p + k):
        h, origin = m._step(pos, sp, sc[j], j, L)
        sj, ba, bb = m._sample_head(h, rem)
        rem[torch.arange(B, device=dev), sj] -= 1
        a = m._bin_center(ba) + (torch.rand(B, device=dev) - 0.5) * m.bin_w
        b = m._bin_center(bb) + (torch.rand(B, device=dev) - 0.5) * m.bin_w
        pos[:, j] = torch.remainder(origin + torch.stack([a, b], -1) * arc, L)
        sp[:, j] = sj
    return pos, sp


@torch.no_grad()
def kstep_metrics(m, N, ks, device, B=256, n_starts=6, thr=0.7):
    """For each k: clash fraction of generated atoms vs all placed-before, and g_BB(r<0.88) spurious
    B-B contacts among generated atoms. Averaged over n_starts evenly-spaced start indices p>=8."""
    ref = torch.load(os.path.join(ART, f"ka_reference_N{N}.pt"), map_location=device, weights_only=False)
    s, L, data = ref["s"].to(device).long(), ref["L"], ref["x"].to(device)[:B]
    order = m.geo._curve_order(data, N); xo = torch.gather(data, 1, order[..., None].expand(-1, -1, 2))
    sful = s.expand(data.shape[0], N) if s.dim() == 1 else s[:B]
    so = torch.gather(sful[:B], 1, order)
    starts = [int(x) for x in torch.linspace(8, max(9, N - max(ks) - 1), n_starts).long().tolist()]
    out = {}
    for k in ks:
        clash, gbb_num, gbb_den = [], 0.0, 0.0
        for p in starts:
            if p + k > N:
                continue
            pos, sp = rollout_window(m, xo, so, p, k, N, L)
            gen = slice(p, p + k)
            for j in range(p, p + k):                                    # clash vs everything placed before j
                df = _wrap_pm(pos[:, j:j+1] - pos[:, :j], L)
                clash.append(((df ** 2).sum(-1).min(1).values.sqrt() < thr).float().mean())
                # spurious B-B contacts: generated B-j vs earlier B's within 0.88
                isB = sp[:, j] == 1
                if isB.any():
                    dB = _wrap_pm(pos[isB, j:j+1] - pos[isB, :j], L)
                    nb = (sp[isB, :j] == 1)
                    dd = (dB ** 2).sum(-1).sqrt().masked_fill(~nb, 1e3)
                    gbb_num += (dd.min(1).values < 0.88).float().sum().item(); gbb_den += int(isB.sum())
        out[k] = (float(torch.stack(clash).mean()), gbb_num / max(gbb_den, 1))
    return out


@torch.no_grad()
def heldout_nll(m, N, device, B=256):
    ref = torch.load(os.path.join(ART, f"ka_reference_N{N}.pt"), map_location=device, weights_only=False)
    s, data = ref["s"].to(device).long(), ref["x"].to(device)[:B]
    sful = s.expand(data.shape[0], N) if s.dim() == 1 else s[:B]
    return float((-m.log_prob(data, sful[:B]) / N).mean())
```

- [ ] **Step 4: Run to verify it passes**

Run: `cd /mnt/ssd/GridTransformer && python $SP/t5_kstep.py`
Expected: PASS — prints `kstep: {...}`, monotone clash, finite NLL, `T5 PASS`.

- [ ] **Step 5: Commit**

```bash
git add liquid_coupling_flow/ka_localframe_kstep.py
git commit -m "feat(localframe): k-step partial-rollout eval harness + heldout NLL"
```

---

## Task 6: Comparison driver (train/load H0,H1,H2; k-step curves at N=36/100/256)

**Files:**
- Modify: `liquid_coupling_flow/ka_localframe_kstep.py` (add `compare`)
- Test: `$SP/t6_compare.py`

**Interfaces:**
- Consumes: `kstep_metrics`, `heldout_nll`, `KALocalFrameModel`, `ka_localframe.train`.
- Produces: `compare(modes=("factorized","species_pos","joint_sa"), sizes=(36,100,256), ks=(1,2,4,8,16,32), steps=20000, retrain=False)` -> writes `artifacts/ka_localframe_headcmp.png` + prints a table.

- [ ] **Step 1: Write the failing test (smoke: tiny steps, plot exists)**

```python
# $SP/t6_compare.py
import os
from liquid_coupling_flow.ka_localframe_kstep import compare
compare(modes=("factorized",), sizes=(36,), ks=(1, 4), steps=6, retrain=True)
assert os.path.exists("liquid_coupling_flow/artifacts/ka_localframe_headcmp.png")
print("T6 PASS")
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd /mnt/ssd/GridTransformer && python $SP/t6_compare.py`
Expected: FAIL — `ImportError: cannot import name 'compare'`.

- [ ] **Step 3: Add `compare` driver**

```python
# append to liquid_coupling_flow/ka_localframe_kstep.py
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from liquid_coupling_flow import ka_localframe as LF


def _load_or_train(mode, train_N, steps, device, retrain):
    path = os.path.join(ART, f"ka_localframe_{mode}_N{train_N}.pt")
    if retrain or not os.path.exists(path):
        LF.train(train_N=train_N, steps=steps, head_mode=mode, knn=8, bf16=(device == "cuda"))
    ck = torch.load(path, map_location=device, weights_only=False)
    m = KALocalFrameModel(rho=ck["rho"], n_bins=ck["n_bins"], knn=ck["knn"], head_mode=ck["head_mode"]).to(device)
    m.load_state_dict(ck["state_dict"]); m.eval(); return m


@torch.no_grad()
def compare(modes=("factorized", "species_pos", "joint_sa"), sizes=(36, 100, 256),
            ks=(1, 2, 4, 8, 16, 32), steps=20000, retrain=False,
            device="cuda" if torch.cuda.is_available() else "cpu"):
    models = {mode: _load_or_train(mode, 100, steps, device, retrain) for mode in modes}
    fig, ax = plt.subplots(2, len(sizes), figsize=(5 * len(sizes), 9), squeeze=False)
    print(f"\n{'mode':14s} {'N':>4s} {'NLL/N':>8s}  clash@k / gBB@k ...")
    for c, N in enumerate(sizes):
        for mode in modes:
            m = models[mode]
            met = kstep_metrics(m, N, list(ks), device)
            nll = heldout_nll(m, N, device)
            ax[0, c].plot(list(ks), [met[k][0] for k in ks], "o-", label=mode)
            ax[1, c].plot(list(ks), [met[k][1] for k in ks], "o-", label=mode)
            tagN = "train" if N == 100 else "held"
            print(f"{mode:14s} {N:4d} {nll:8.3f}  " + " ".join(f"{met[k][0]:.2f}/{met[k][1]:.2f}" for k in ks)
                  + f"  ({tagN})", flush=True)
        ax[0, c].set_title(f"N={N} clash vs k"); ax[1, c].set_title(f"N={N} g_BB(r<0.88) vs k")
        for r in (0, 1):
            ax[r, c].set_xscale("log", base=2); ax[r, c].set_xlabel("k (free steps)"); ax[r, c].legend(fontsize=8); ax[r, c].grid(alpha=0.3)
    fig.suptitle("Output-head comparison: k-step partial-rollout drift (H0/H1/H2), trained N=100", fontsize=13)
    fig.tight_layout(); out = os.path.join(ART, "ka_localframe_headcmp.png"); fig.savefig(out, dpi=120)
    print(f"saved {out}", flush=True)


if __name__ == "__main__":
    compare()
```

- [ ] **Step 4: Run to verify it passes**

Run: `cd /mnt/ssd/GridTransformer && python $SP/t6_compare.py`
Expected: PASS — writes `ka_localframe_headcmp.png`, `T6 PASS`.

- [ ] **Step 5: Commit**

```bash
git add liquid_coupling_flow/ka_localframe_kstep.py
git commit -m "feat(localframe): head-comparison driver (k-step curves + NLL, H0/H1/H2)"
```

- [ ] **Step 6: Run the real experiment (long; not a unit step)**

Run (background): `cd /mnt/ssd/GridTransformer && python -m liquid_coupling_flow.ka_localframe_kstep > $SP/headcmp.log 2>&1`
Expected: trains H0/H1/H2 (~30 min each at KNN=8+bf16), then `ka_localframe_headcmp.png` + the printed table. Read the verdict: which head's clash/g_BB rises slowest with k, at trained N=100 and held-out N=36/256.

---

## Self-Review

**1. Spec coverage:**
- k-step partial rollout (primary) → Task 5/6. ✓
- Exact held-out NLL (secondary) → `heldout_nll`, Task 5/6. ✓
- H0/H1/H2 heads, H2 = joint (s,a) then b|s,a → Tasks 1-3. ✓
- Exact likelihood per variant → `return_logq` self-consistency gate, Tasks 1-3. ✓
- Geometry-invariant (no curve PE) → heads read only `context`; no PE added. ✓
- Canonical composition exact → Task 3 gate. ✓
- N=36/100/256 → Tasks 5/6. ✓
- KNN=8 + bf16 efficiency → Task 4. ✓
- Backward-compat / H0 regression → Task 1 loads default mode; `tf_clash` unchanged. ✓
- Phase B (block flow) → explicitly out of this plan (own spec). ✓

**2. Placeholder scan:** No TBD/TODO; every code step shows full code; test code is concrete. ✓

**3. Type consistency:** `_sample_head -> (sj,ba,bb)` Long[B]; `_logq_head` same arg order; `sample(...return_logq) -> (pos,sp,logq)`; `log_prob` returns `[B]`; checkpoint keys `head_mode/knn` consistent across Tasks 4-6; joint decode `flat = s*n_bins + a` used identically in `_sample_head`, `_logq_head`, `log_prob`. ✓

**Note on commits:** the standing rule in this repo is *commit only when the user asks*. The commit steps are written into the plan per workflow, but at execution we will confirm before running `git commit`.
