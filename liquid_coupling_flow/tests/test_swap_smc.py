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
