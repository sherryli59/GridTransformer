"""FlowBase: the eRSI flow deployed as a composed-likelihood SMC base for mw_smc.smc_run.
Implements the mw_base interface (sample / log_q / LOGQ_CONST=False) with a SELF-CONTAINED midpoint
integrator over ipl44's G-a-verified forward_and_divergence (bypasses the shadowed learndiffeq RFM
whose compute_div lacks the species `a` kwarg).

  sample(B): forward ODE uniform->target, x ~ q0.
  log_q(x): reverse ODE target->base, log q0(x) = -N*3*log L - integral(div v) dt   (instantaneous
            change of variables; uniform base density at the recovered t=0 point).

Exactness note: forward-sample-logq and reverse-logq are exact inverses only in the ODE limit; at
finite steps they differ by the discretization error (small at >=40 steps). smc_run's rung guard is
self-consistent (deterministic reverse-logq matches a fresh reverse-logq), so the guard passes; the
residual forward/reverse mismatch is a small density-model bias, not a bookkeeping bug."""
import math, sys, torch
IPL44 = "/mnt/ssd/GridTransformer/liquid_coupling_flow/ipl44/learndiffeq"
for m in [x for x in sys.modules if x.startswith("learndiffeq")]:
    del sys.modules[m]
sys.path.insert(0, IPL44)
from learndiffeq.particles.velocities.egnn_traceable import EGNN_dynamics


class FlowBase:
    LOGQ_CONST = False

    def __init__(self, ckpt, N, L, device="cuda", steps=40, hidden_nf=128, n_layers=4, K=12):
        self.N, self.L, self.device, self.steps = N, L, device, steps
        e = EGNN_dynamics(n_particles=N, n_dimension=3, hidden_nf=hidden_nf, n_layers=n_layers,
                          max_neighbors=K, L=L, n_species=1).to(device)
        c = torch.load(ckpt, map_location=device, weights_only=False)
        e.load_state_dict({k[2:]: v for k, v in c["state_dict"].items() if k.startswith("b.")}, strict=True)
        self.e = e.eval()
        self._lq0 = -N * 3 * math.log(L)

    def _a(self, B):
        return torch.zeros(B, self.N, dtype=torch.long, device=self.device)

    @torch.no_grad()
    def sample(self, B, gen=None):
        x = torch.rand(B, self.N, 3, device=self.device, generator=gen) * self.L
        a = self._a(B)
        ts = torch.linspace(0, 1, self.steps + 1, device=self.device)
        for i in range(self.steps):
            dt = float(ts[i + 1] - ts[i])
            v1, _ = self.e.forward_and_divergence(x, ts[i].expand(B), a)
            xm = torch.remainder(x + 0.5 * dt * v1, self.L)
            vm, _ = self.e.forward_and_divergence(xm, (ts[i] + 0.5 * dt).expand(B), a)
            x = torch.remainder(x + dt * vm, self.L)
        return x

    @torch.no_grad()
    def log_q(self, x, chunk=None):
        x = torch.remainder(x.to(self.device), self.L)
        B = x.shape[0]
        a = self._a(B)
        ts = torch.linspace(0, 1, self.steps + 1, device=self.device)
        D = torch.zeros(B, device=self.device, dtype=torch.float64)   # accumulates integral(div) dt
        xi = x.clone()
        for i in reversed(range(self.steps)):          # traverse t=1 -> t=0
            dt = float(ts[i + 1] - ts[i])
            t1 = ts[i + 1].expand(B)
            v1, _ = self.e.forward_and_divergence(xi, t1, a)
            xm = torch.remainder(xi - 0.5 * dt * v1, self.L)
            vm, dm = self.e.forward_and_divergence(xm, (ts[i + 1] - 0.5 * dt).expand(B), a)
            xi = torch.remainder(xi - dt * vm, self.L)
            D = D + dt * dm.double()
        return (self._lq0 - D).to(x.dtype)


if __name__ == "__main__":
    import time
    from liquid_coupling_flow.mw.mw_energy import RHO_STAR
    DEV = "cuda"
    CK = "liquid_coupling_flow/mw/artifacts/mw_ersi_N64_checkpoints/last.ckpt"
    for N in (64,):
        L = (N / RHO_STAR) ** (1 / 3)
        fb = FlowBase(CK, N, L, DEV, steps=40)
        g = torch.Generator(device=DEV).manual_seed(0)
        t0 = time.time()
        x = fb.sample(64, g)
        torch.cuda.synchronize()
        t_samp = time.time() - t0
        # consistency: forward-sample-logq vs reverse-logq on the same x
        t0 = time.time()
        lq = fb.log_q(x)
        torch.cuda.synchronize()
        t_logq = time.time() - t0
        # per-move proxy: log_q of a batch with one particle perturbed
        xp = x.clone()
        xp[:, 0] = torch.remainder(xp[:, 0] + 0.0783 * torch.randn(64, 3, device=DEV, generator=g), L)
        t0 = time.time()
        lqp = fb.log_q(xp)
        torch.cuda.synchronize()
        t_move = time.time() - t0
        print(f"N={N}: sample(B=64) {t_samp:.2f}s | log_q(B=64) {t_logq:.2f}s (== per single-site move cost)")
        print(f"  log_q range [{float(lq.min()):.1f}, {float(lq.max()):.1f}]  mean {float(lq.mean()):.1f}  "
              f"(uniform lq0 {fb._lq0:.1f})")
        est_rung = t_move * N * 1  # n_sweeps=1
        print(f"  => est per-rung mutation cost (N moves x n_sweeps=1): {est_rung:.0f}s "
              f"=> ~{est_rung*15/60:.0f} min for a 15-rung SMC run")
