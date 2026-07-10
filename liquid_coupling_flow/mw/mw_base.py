"""Uniform base distribution q0 for the mW annealed-SMC path (Task 5). log q0 is a CONSTANT
(uniform-in-box, i.i.d. per particle), so it must drop out of every mutation acceptance ratio —
LOGQ_CONST=True lets mw_smc.mutation_sweeps certify that reduction BEFORE any neural base exists
(Phase-2 fault isolation)."""
from __future__ import annotations
import math, os, torch


class UniformBase:
    LOGQ_CONST = True

    def __init__(self, N, L):
        self.N, self.L = N, L
        self._lq = -N * 3 * math.log(L)

    def sample(self, B, gen=None):
        return torch.rand(B, self.N, 3, generator=gen,
                          device=gen.device if gen is not None and gen.device.type == "cuda" else "cpu") * self.L

    def log_q(self, x):
        return torch.full((x.shape[0],), self._lq, device=x.device)


class GeneratorBase:
    """mw_smc base backed by a trained autoregressive generator (v10 & kin).

    Wraps any model exposing ``sample(B, N, L, gen, return_logq=True)`` and
    ``log_prob(x, L, preordered=True)`` so mw_smc's smc_run / mutation_sweeps can use it exactly
    like ``UniformBase``:
      * ``sample(B, gen) -> x``           : draw B configs in the model's canonical AR order;
      * ``log_q(x) -> [B]``               : the AR base log-density, evaluated with preordered=True
                                            (the labeled-space exactness convention -- the current
                                            particle labeling IS the AR conditioning order; the SMC
                                            weights and pi_lambda-invariant mutation MH are exact in
                                            that labeled space), in eval mode, no_grad, chunked <=128;
      * ``LOGQ_CONST = False``            : log q0 is NOT a constant, so mutation_sweeps must (and
                                            does) evaluate base.log_q PER site-move -- one causal
                                            forward per move (the cost the base gate measures).
    """

    LOGQ_CONST = False

    def __init__(self, model, N, L):
        self.model = model.eval()
        self.N, self.L = N, L

    def sample(self, B, gen=None):
        self.model.eval()
        with torch.no_grad():
            x, _ = self.model.sample(B, self.N, self.L, gen=gen, return_logq=True)
        return x

    def log_q(self, x, chunk=128):
        self.model.eval()
        outs = []
        with torch.no_grad():
            for i in range(0, x.shape[0], chunk):
                outs.append(self.model.log_prob(x[i:i + chunk], self.L, preordered=True))
        return torch.cat(outs)


# v10-family architecture tags -> load_generator_v10 (identity-v4 pretrain, both fine-tunes).
_V10_TAGS = {
    "v4_exact_plus_toroidal_residual_v10",
    "v10_struct_finetune",
    "v10_struct_finetune2_reinforce",
}


def load_any_generator(path, device):
    """Load a generator checkpoint by dispatching on its ``architecture`` tag; raise on an unknown
    tag rather than guess.  Lazy per-branch imports keep the model stack out of mw_base's import
    graph (mw_smc imports UniformBase, which must stay light)."""
    ck = torch.load(path, map_location=device, weights_only=False)
    arch = ck.get("architecture")
    if arch in _V10_TAGS:
        from liquid_coupling_flow.mw.mw_generator_v10 import load_generator_v10
        return load_generator_v10(path, device=device)
    raise ValueError(
        f"load_any_generator: unknown architecture tag {arch!r} in {os.path.basename(path)}; "
        f"known tags = {sorted(_V10_TAGS)}")
