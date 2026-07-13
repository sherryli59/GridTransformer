import torch
from liquid_coupling_flow.ka3d_coarse_head import CoarseFineHead, coarse_tilt_V, cells_allowed
from liquid_coupling_flow.ka3d_scaffold_ar import ball_squash
from liquid_coupling_flow.ka3d_ebm_batched import KA3DScaffoldEBMBatched
from liquid_coupling_flow.ka_energy import SIGMA


def test_roundtrip_u_to_bins_to_u():
    h = CoarseFineHead(d_model=32)
    u = (torch.rand(500, 3) - 0.5) * 2 * 2.5 * 0.999
    ci = h.cell_index(u)
    fb = h.fine_bin(u, ci)
    ctr = h.fine_ctr(ci, fb)
    assert ci.min() >= 0 and ci.max() < 4096
    assert fb.min() >= 0 and fb.max() < 8
    # every point lies within half a fine bin of its reconstructed center
    assert (u - ctr).abs().max() <= h.bwf / 2 + 1e-6


def test_grid_constants():
    h = CoarseFineHead(d_model=32)
    assert abs(h.cw - 5.0 / 16) < 1e-9
    assert abs(h.bwf - 5.0 / 128) < 1e-9        # == current fl.bw -> same final resolution
    assert abs(h.half_diag - h.cw * 3 ** 0.5 / 2) < 1e-9


def test_cell_center_inverse():
    h = CoarseFineHead(d_model=32)
    idx = torch.arange(4096)
    assert torch.equal(h.cell_index(h.cell_center(idx)), idx)


def test_coarse_tilt_matches_bruteforce():
    torch.manual_seed(0)
    model = KA3DScaffoldEBMBatched(cat_bins=128, cat_range=2.5)
    model.use_frame = False
    # De-zero the phi net's output layer to make the test non-vacuous
    torch.nn.init.normal_(model.phi[-1].weight, std=1.0)
    torch.nn.init.normal_(model.phi[-1].bias, std=1.0)
    model.eval()
    head = CoarseFineHead(d_model=model.d_model)
    S, Kc, R = 5, 6, 4.5
    anchor_y = (torch.rand(S, 3) - 0.5) * 2.0
    cage_x = (torch.rand(S, Kc, 3) - 0.5) * 2 * R * 0.9
    cage_s = torch.randint(0, 2, (S, Kc))
    cage_v = torch.ones(S, Kc, dtype=torch.bool)
    cage_v[:, -1] = False   # exercise padding mask
    sj = torch.randint(0, 2, (S,))

    with torch.no_grad():
        V = coarse_tilt_V(model, anchor_y, cage_x, cage_s, cage_v, sj, R, head)
    assert V.shape == (S, head.n_cells)
    assert V.abs().max() > 0, "Tilt potential is zero (test is vacuous)"

    idxs = torch.randperm(head.n_cells)[:40]
    with torch.no_grad():
        for i in idxs.tolist():
            center = head.centers[i]
            y = anchor_y + center[None, :]
            x, _ = ball_squash(y, R)
            d = (x[:, None, :] - cage_x).norm(dim=-1)                  # [S,Kc]
            phi = model._phi_pair(d, sj, cage_s, model.phi, model.pair_emb)
            v_i = (phi * cage_v).sum(-1)
            assert torch.allclose(V[:, i], v_i, atol=1e-5), i


def test_cells_allowed_never_forbids_a_truly_safe_cell():
    torch.manual_seed(1)
    head = CoarseFineHead(d_model=16)
    S, Kc, R, cut = 200, 4, 4.5, 1.0
    anchor_y = torch.zeros(S, 3)
    u = (torch.rand(S, 3) - 0.5) * 2 * head.rng * 0.999
    cage_x = (torch.rand(S, Kc, 3) - 0.5) * 2 * R * 0.9
    cage_s = torch.randint(0, 2, (S, Kc))
    cage_v = torch.ones(S, Kc, dtype=torch.bool)
    sj = torch.randint(0, 2, (S,))

    pos, _ = ball_squash(anchor_y + u, R)
    sig_tab = torch.tensor(SIGMA)
    sig = sig_tab[sj.long()[:, None], cage_s.long()]                   # [S,Kc]
    d = (pos[:, None, :] - cage_x).norm(dim=-1)                        # [S,Kc]
    gap = (d / sig).masked_fill(~cage_v, float("inf"))
    real_gap = gap.min(-1).values                                      # [S] true (unbounded) min sigma-gap

    allowed = cells_allowed(head, anchor_y, sj, cage_x, cage_s, cage_v, R, cut)
    cell_idx = head.cell_index(u)
    safe = real_gap >= cut
    assert safe.sum() > 0, "test cage too dense: no safe points generated"
    row_allowed = allowed[torch.arange(S), cell_idx]
    assert row_allowed[safe].all()
