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
    def _self_prefix(self, xo, so, sc, L, N, p_keep, gen=None):
        """Pass 1: place every j from the TRUE prefix; per-particle Bernoulli(1-p_keep) mix -> detached xmix.
        `gen` seeds ALL pass-1 randomness (placement multinomials + jitter + keep mask) so the self-
        conditioning is reproducible run-to-run; without it the trainers' seed gives false determinism."""
        B = xo.shape[0]; arc = self._arc_scale(N); dev = xo.device
        context, origin = self._local(xo, so, sc, L, N)
        ba = torch.multinomial(F.softmax(self.head_a(context).reshape(-1, self.n_bins), -1), 1, generator=gen).reshape(B, N)
        bb = torch.multinomial(F.softmax(self.head_b(context + self.bin_a_emb(ba)).reshape(-1, self.n_bins), -1), 1, generator=gen).reshape(B, N)
        a = self._bin_center(ba) + (torch.rand(B, N, device=dev, generator=gen) - 0.5) * self.bin_w
        bc = self._bin_center(bb) + (torch.rand(B, N, device=dev, generator=gen) - 0.5) * self.bin_w
        x_hat = torch.remainder(origin + torch.stack([a, bc], -1) * arc, L)
        keep = torch.rand(B, N, device=dev, generator=gen) < p_keep         # True -> keep TRUE position
        return torch.where(keep[..., None], xo, x_hat).detach()

    def log_prob_sched(self, x, s, p_keep, *, sigma_bins=0.0, soft=False, stochastic=False,
                       canonical=None, gen=None):
        B, N = x.shape[0], x.shape[1]; s = s.long()
        s = s.expand(B, N).clone() if s.dim() == 1 else s
        L = self._Lof(N); sc = self.geo._scaffold(N, x.device)
        order = self.geo._curve_order(x, N)
        xo = torch.gather(x, 1, order[..., None].expand(-1, -1, 2)); so = torch.gather(s, 1, order)
        xmix = self._self_prefix(xo, so, sc, L, N, p_keep, gen=gen)         # detached drifted prefix
        context, origin = self._local(xmix, so, sc, L, N)                  # grad path (pass 2)
        nll = pos_species_nll(self, context, origin, xo, so, L, N, sigma_bins=sigma_bins,
                              soft=soft, stochastic=stochastic, canonical=canonical, gen=gen)
        return -nll                                                        # log_prob (variable part)


def train(train_N=100, steps=10000, r=2.0, floor=0.5, sigma_bins=0.0, soft=False, stochastic=False,
          warm="ka_localframe_N100_20k.pt", out="ka_sched_N100.pt",
          device="cuda" if torch.cuda.is_available() else "cpu"):
    from liquid_coupling_flow.ka_gridformer_train import augment
    from liquid_coupling_flow.ka_exposure_lf import load_compat
    ref = torch.load(os.path.join(ART, f"ka_reference_N{train_N}.pt"), map_location=device, weights_only=False)
    data, sp, L = ref["x"].to(device), ref["s"].to(device).long(), ref["L"]; N = data.shape[1]
    m = KALocalFrameSched(rho=1.2, n_bins=192, knn=KNN).to(device)
    ck = torch.load(os.path.join(ART, warm), map_location=device, weights_only=False)
    load_compat(m, ck["state_dict"]); m.train()                         # warm-start (allowlists optional heads)
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
                "r": r, "floor": floor, "step": steps}, os.path.join(ART, out))
    print(f"saved {out}", flush=True)


if __name__ == "__main__":
    import sys
    train(steps=int(sys.argv[1]) if len(sys.argv) > 1 else 10000)
