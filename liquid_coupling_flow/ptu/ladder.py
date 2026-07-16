"""(T, coord) replica ladder over cavity states with exact Metropolis exchange.
mode='u': identity-bridge (tables from build_tables_u; per-rung identity_sweep enabled).
mode='lam': BCY shrinkage (tables from build_tables_lam; displacement-only, BCY verbatim).
Metrics: TRIPS (lineage bottom->top->bottom), Katzgraber flow f(r), per-rung exchange acc,
dual-init bottom-rung q_c + label-overlap traces. Ported from
reports/logs-2026-07-15/pt_decorrelation_metrics.py (flow/trips bookkeeping) and
bcy_gpts_v2.py (exchange acceptance)."""
import numpy as np
from liquid_coupling_flow.ptu.tables import build_tables_u, build_tables_lam
from liquid_coupling_flow.ptu.kernels import disp_sweep, mobile_U, seed_numba
from liquid_coupling_flow.ptu.identity import identity_sweep


class Ladder:
    def __init__(self, allx0, alls0, n, n_tot, mode, coords, temps, nch=2,
                 exch_mean=10, step=0.3, R=2.0, seed=0):
        assert mode in ("u", "lam")
        self.n, self.n_tot, self.mode = n, n_tot, mode
        self.coords = list(coords); self.temps = list(temps)
        self.NR = len(coords); self.nch = nch
        self.exch_mean = exch_mean; self.step = step; self.R = R
        build = build_tables_u if mode == "u" else build_tables_lam
        self.tabs = [build(c) for c in coords]
        self.betas = [1.0 / t for t in temps]
        # state[stack][rung][chain] -> (allx, alls) copies
        self.X = [[[allx0.copy() for _ in range(nch)] for _ in range(self.NR)] for _ in range(2)]
        self.S = [[[alls0.copy() for _ in range(nch)] for _ in range(self.NR)] for _ in range(2)]
        self.x_ref = allx0[:n].copy(); self.s_ref = alls0[:n].copy()
        self.rng = np.random.default_rng(seed)
        seed_numba(seed)
        # trips/flow bookkeeping per (stack, rung, chain)
        self.seen_top = np.zeros((2, self.NR, nch), dtype=np.bool_)
        self.flow = np.full((2, self.NR, nch), 0.5)
        self.flow_sum = np.zeros(self.NR); self.flow_cnt = np.zeros(self.NR)
        self.trips = 0
        self.exch_acc = np.zeros(self.NR - 1); self.exch_att = np.zeros(self.NR - 1)

    def _sweep_all(self):
        for st in range(2):
            for r in range(self.NR):
                beta = self.betas[r]; tabs = self.tabs[r]
                for ch in range(self.nch):
                    disp_sweep(self.X[st][r][ch], self.S[st][r][ch], self.n, self.n_tot,
                               beta, self.R, self.step, *tabs)
                    if self.mode == "u":
                        identity_sweep(self.X[st][r][ch], self.S[st][r][ch], self.n,
                                       self.n_tot, beta, 4, *tabs)

    def _exchange(self):
        for r in range(self.NR - 1):
            if self.rng.random() >= 1.0 / self.exch_mean:
                continue
            for st in range(2):
                for ch in range(self.nch):
                    xa, sa = self.X[st][r][ch], self.S[st][r][ch]
                    xb, sb = self.X[st][r + 1][ch], self.S[st][r + 1][ch]
                    Uaa = mobile_U(xa, sa, self.n, self.n_tot, *self.tabs[r])
                    Uab = mobile_U(xb, sb, self.n, self.n_tot, *self.tabs[r])
                    Uba = mobile_U(xa, sa, self.n, self.n_tot, *self.tabs[r + 1])
                    Ubb = mobile_U(xb, sb, self.n, self.n_tot, *self.tabs[r + 1])
                    dlog = -self.betas[r] * (Uab - Uaa) - self.betas[r + 1] * (Uba - Ubb)
                    self.exch_att[r] += 1
                    if np.log(self.rng.random()) < dlog:
                        self.exch_acc[r] += 1
                        self.X[st][r][ch], self.X[st][r + 1][ch] = xb, xa
                        self.S[st][r][ch], self.S[st][r + 1][ch] = sb, sa
                        for arr in (self.seen_top, self.flow):
                            tmp = arr[st, r, ch].copy()
                            arr[st, r, ch] = arr[st, r + 1, ch]
                            arr[st, r + 1, ch] = tmp
        # flow/trips accumulate every sweep (dwell-time weighting) -- intentional divergence
        # from pt_decorrelation_metrics.py's attempt-gated version
        self.seen_top[:, self.NR - 1, :] = True
        self.flow[:, self.NR - 1, :] = 0.0
        self.trips += int(self.seen_top[:, 0, :].sum())
        self.seen_top[:, 0, :] = False
        self.flow[:, 0, :] = 1.0
        det = self.flow != 0.5
        self.flow_sum += (self.flow * det).sum(axis=(0, 2))
        self.flow_cnt += det.sum(axis=(0, 2))

    def randomize_stack_B(self, n_sweeps):
        beta = self.betas[-1]; tabs = self.tabs[-1]
        for st_r_ch in [(1, r, ch) for r in range(self.NR) for ch in range(self.nch)]:
            st, r, ch = st_r_ch
            for _ in range(n_sweeps):
                disp_sweep(self.X[st][r][ch], self.S[st][r][ch], self.n, self.n_tot,
                           beta, self.R, self.step, *tabs)
                if self.mode == "u":
                    identity_sweep(self.X[st][r][ch], self.S[st][r][ch], self.n,
                                   self.n_tot, beta, 4, *tabs)

    def run(self, n_sweeps, rec_every, qc_fn):
        qcA, qcB, labA, labB, rec_sw = [], [], [], [], []
        for sw in range(1, n_sweeps + 1):
            self._sweep_all()
            self._exchange()
            if sw % rec_every == 0:
                a = np.mean([qc_fn(self.X[0][0][ch][:self.n], self.x_ref) for ch in range(self.nch)])
                b = np.mean([qc_fn(self.X[1][0][ch][:self.n], self.x_ref) for ch in range(self.nch)])
                la = np.mean([np.mean(self.S[0][0][ch][:self.n] == self.s_ref) for ch in range(self.nch)])
                lb = np.mean([np.mean(self.S[1][0][ch][:self.n] == self.s_ref) for ch in range(self.nch)])
                qcA.append(a); qcB.append(b); labA.append(la); labB.append(lb); rec_sw.append(sw)
        fr = self.flow_sum / np.maximum(self.flow_cnt, 1)
        return {"qcA": qcA, "qcB": qcB, "labA": labA, "labB": labB, "rec_sw": rec_sw,
                "trips": self.trips, "flow": fr,
                "exch_acc": self.exch_acc, "exch_att": self.exch_att}
