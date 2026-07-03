"""Exactness tests for the swap-MH kernel: count preservation, hand-computed proposal ratio, and the decisive
long-run stationarity check against exact enumeration on a tiny system (positions frozen, swaps only)."""
import math, itertools, torch, pytest
from liquid_coupling_flow.ipl44.ipl_swap_smc import swap_attempt, uniform_weight_fn, _swap_log_ratio

torch.manual_seed(0)


def toy_energy(x, s, L=6.0):
    """Independent soft-sphere implementation (min-image, sigma depends on species pair): U=sum (sig/r)^12."""
    B, N, _ = x.shape
    sig = torch.tensor([[1.0, 1.2], [1.2, 1.4]])
    d = x[:, :, None, :] - x[:, None, :, :]
    d = d - L * torch.round(d / L)
    r = d.norm(dim=-1).clamp_min(1e-9)
    sij = sig[s[:, :, None].expand(-1, -1, N), s[:, None, :].expand(-1, N, -1)]
    e = (sij / r) ** 12
    iu = torch.triu_indices(N, N, offset=1)
    return e[:, iu[0], iu[1]].sum(-1)


def sdep_weight_fn(x, s):
    """Nontrivial, s-DEPENDENT weights to stress the reverse-evaluation path."""
    return torch.sigmoid(x[..., 0] - 3.0 + 0.5 * s.float().mean(dim=1, keepdim=True))


def test_count_preserved_and_only_ab_swaps():
    B, N = 32, 8
    x = torch.rand(B, N, 2) * 6.0
    s = torch.zeros(B, N, dtype=torch.long); s[:, :4] = 1
    s = torch.gather(s, 1, torch.argsort(torch.rand(B, N), dim=1))
    U = toy_energy(x, s)
    for _ in range(20):
        s, U, acc = swap_attempt(x, s, U, beta=2.0, energy_fn=toy_energy, weight_fn=sdep_weight_fn)
        assert (s.sum(1) == 4).all()
    # U must track the true energy of the current state
    assert torch.allclose(U, toy_energy(x, s), atol=1e-4)


def test_swap_log_ratio_hand_case():
    """N=4, one config: s=[A,A,B,B]; swap i=0 (A), j=2 (B). Hand-compute g and g'."""
    s = torch.tensor([[0, 0, 1, 1]])
    s_new = torch.tensor([[1, 0, 0, 1]])
    pB_f = torch.tensor([[0.9, 0.1, 0.2, 0.3]])   # forward weights (at state s)
    pB_r = torch.tensor([[0.6, 0.2, 0.7, 0.4]])   # reverse weights (at state s_new)
    i = torch.tensor([0]); j = torch.tensor([2])
    # forward: S_A = pB(0)+pB(1) = 1.0 -> P(i=0)=0.9 ; S_B = (1-pB(2))+(1-pB(3)) = 0.8+0.7=1.5 -> P(j=2)=0.8/1.5
    g = (0.9 / 1.0) * (0.8 / 1.5)
    # reverse (state s_new = [B,A,A,B]): A's = {1,2}: S'_A = pB_r(1)+pB_r(2) = 0.9 -> picks particle 2 (was j): 0.7/0.9
    # B's = {0,3}: S'_B = (1-pB_r(0))+(1-pB_r(3)) = 0.4+0.6 = 1.0 -> picks particle 0 (was i): 0.4/1.0
    gp = (0.7 / 0.9) * (0.4 / 1.0)
    lr = _swap_log_ratio(pB_f, pB_r, s, s_new, i, j)
    assert torch.allclose(lr, torch.tensor([math.log(gp / g)]), atol=1e-5)


@pytest.mark.parametrize("wfn_name", ["uniform", "sdep"])
def test_stationarity_exact_enumeration(wfn_name):
    """Positions frozen, swaps only: chain's empirical distribution over the C(6,3)=20 species states must
    match the exact Boltzmann distribution. Run B=256 parallel chains x 400 attempts, use the second half."""
    torch.manual_seed(1)
    N, nB, beta, L = 6, 3, 1.0, 6.0
    x1 = torch.rand(1, N, 2) * L
    wfn = uniform_weight_fn if wfn_name == "uniform" else sdep_weight_fn
    # exact distribution over labelings
    states = [c for c in itertools.combinations(range(N), nB)]
    svecs = torch.zeros(len(states), N, dtype=torch.long)
    for k, c in enumerate(states):
        svecs[k, list(c)] = 1
    Uex = toy_energy(x1.expand(len(states), -1, -1), svecs, L)
    logp = -beta * Uex; p_exact = torch.softmax(logp, 0)
    # chain
    B = 256
    x = x1.expand(B, -1, -1).contiguous()
    s = svecs[torch.randint(0, len(states), (B,))].clone()
    U = toy_energy(x, s, L)
    key = {tuple(v.tolist()): k for k, v in enumerate(svecs)}
    counts = torch.zeros(len(states))
    for t in range(400):
        s, U, _ = swap_attempt(x, s, U, beta=beta, energy_fn=lambda a, b: toy_energy(a, b, L), weight_fn=wfn)
        if t >= 200:
            for b in range(B):
                counts[key[tuple(s[b].tolist())]] += 1
    emp = counts / counts.sum()
    tv = 0.5 * (emp - p_exact).abs().sum()
    assert tv < 0.05, f"TV(empirical, exact) = {tv:.3f} for {wfn_name} proposer"


def test_position_sweep_preserves_equilibrium():
    """Start FROM reference equilibrium configs at beta=10: 30 sweeps must not drift the energy median
    outside the 5% band (stationarity of the position kernel)."""
    from liquid_coupling_flow.ipl44.ipl_swap_smc import position_sweep
    from liquid_coupling_flow.ipl44.ipl_energy import ipl_energy, ipl_box
    N, L = ipl_box()
    D = "/mnt/ssd/GridTransformer/datasets"
    x = torch.remainder(torch.load(f"{D}/ipl44_T0.1_positions.pt", weights_only=False).float(), float(L))[:64]
    sp = torch.load(f"{D}/ipl44_T0.1_species.pt", weights_only=False).long()[:64]
    o = sp.argsort(-1); sp = torch.gather(sp, 1, o)
    x = torch.gather(x, 1, o.unsqueeze(-1).expand(-1, -1, 2))
    ref_med = ipl_energy(x, sp).median()
    U = ipl_energy(x, sp)
    acc = 0.0
    for _ in range(30):
        x, U, acc = position_sweep(x, sp, U, beta=10.0, L=float(L), step=0.08, energy_fn=ipl_energy)
    assert (U.median() - ref_med).abs() / ref_med.abs() < 0.05
    assert 0.05 < acc < 0.95


def test_run_chain_and_band():
    from liquid_coupling_flow.ipl44.ipl_swap_smc import run_chain, sweeps_to_band
    from liquid_coupling_flow.ipl44.ipl_energy import ipl_energy, ipl_box
    N, L = ipl_box()
    x0 = torch.rand(16, N, 2) * float(L)
    s0 = torch.zeros(16, N, dtype=torch.long); s0[:, 22:] = 1
    out = run_chain(x0, s0, n_sweeps=20, beta=10.0, L=float(L), energy_fn=ipl_energy,
                    weight_fn=uniform_weight_fn, n_swap=4, record_every=5)
    assert len(out["sweep"]) == len(out["U_median"]) == len(out["gbb_peak"])
    assert (out["s"].sum(1) == 22).all()
    # sweeps_to_band: monotone curve hitting the band -> first index's sweep
    curves = {"sweep": [0, 10, 20, 30], "U_median": [30.0, 16.0, 14.8, 14.6], "gbb_peak": [0.5, 2.0, 4.0, 4.2]}
    assert sweeps_to_band(curves, U_ref_med=14.7, gbb_ref_peak=4.3) == 20


def test_per_species_ot_blocks():
    from liquid_coupling_flow.ipl44.joint_flow import per_species_ot
    B, N, nA, L = 8, 44, 22, 9.38
    x0, x1 = torch.rand(B, N, 2) * L, torch.rand(B, N, 2) * L
    xa = per_species_ot(x0, x1, nA, L)
    # block structure: aligned A-block is a permutation of x0's A-block (set equality of rows)
    for b in range(B):
        s0 = set(map(tuple, x0[b, :nA].round(decimals=5).tolist()))
        sa = set(map(tuple, xa[b, :nA].round(decimals=5).tolist()))
        assert s0 == sa


def test_denoiser_eval_runs():
    from liquid_coupling_flow.ipl44.joint_flow import JointSpeciesFlow, denoiser_eval
    m = JointSpeciesFlow(n_particles=44, L=9.38, hidden_nf=16, n_layers=2)
    x1 = torch.rand(6, 44, 2) * 9.38
    s1 = torch.zeros(6, 44, dtype=torch.long); s1[:, 22:] = 1
    out = denoiser_eval(m, x1, s1, t_eval=0.9, n_rep=2)
    assert 0.0 <= out["acc"] <= 1.0 and 0.0 <= out["ece"] <= 1.0
