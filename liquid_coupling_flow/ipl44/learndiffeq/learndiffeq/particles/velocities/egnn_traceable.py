# Traceable EGNN backend adapted from flashdiv/nets/egnn_cutoff.py.
# This module keeps a learndiffeq-compatible interface: forward(t, x, a=None).

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


def log_map(X, Y, L):
    diff = Y - X
    return diff - torch.round(diff / L) * L


class EGNN_dynamics(nn.Module):
    def __init__(
        self,
        n_particles,
        n_dimension,
        hidden_nf=64,
        act_fn=torch.nn.SiLU(),
        n_layers=4,
        recurrent=True,
        attention=False,
        cutoff=None,
        max_neighbors=None,
        condition_time=True,
        tanh=False,
        mode="egnn_dynamics",
        agg="sum",
        out_node_nf=None,
        L=None,
        n_species=1,
        **kwargs,
    ):
        super().__init__()
        self.mode = mode
        self._n_particles = int(n_particles)
        self._n_dimension = int(n_dimension)
        self.max_nb_neighbors = (
            min(int(max_neighbors), self._n_particles - 1)
            if max_neighbors is not None
            else self._n_particles - 1
        )
        self.condition_time = condition_time
        self.cutoff = float(cutoff) if cutoff is not None else 100.0
        self.L = float(L) if L is not None else None

        # n_species>1: species enter ONLY as position-independent labels (one-hot in the inner-EGNN node
        # features + the central species in the pot input). They never touch the radial structure (diffij),
        # so the analytical divergence (which differentiates pot only w.r.t. the scalar rij) stays EXACT.
        self.n_species = int(n_species)
        self._sp_nf = self.n_species if self.n_species > 1 else 0   # extra one-hot width

        self.out_node_nf = 1 if out_node_nf is None else int(out_node_nf)
        self.egnn = EGNN(
            in_node_nf=1 + self._sp_nf,            # time scalar + neighbour species one-hot
            in_edge_nf=1,
            hidden_nf=hidden_nf,
            act_fn=act_fn,
            n_layers=n_layers,
            recurrent=recurrent,
            attention=attention,
            tanh=tanh,
            agg=agg,
            out_node_nf=self.out_node_nf,
            # Inner message passing operates on the *contiguous* neighbour cloud
            # already built (min-imaged about the central particle) in
            # _compute_common_terms. It must therefore use RAW coordinate
            # differences -- min-imaging neighbour-neighbour diffs with L would
            # wrongly collapse opposite-side neighbours (true sep ~L) to ~0.
            # This matches flash-div's EGNN_dynamicsPeriodic (non-periodic inner
            # E_GCL). Periodicity is handled only in the outer contiguation.
            L=None,
        )

        self.pot_model = nn.Sequential(
            nn.Linear(1 + 1 + self.out_node_nf + self._sp_nf, hidden_nf),   # rij, t, h_final, central-species
            act_fn,
            nn.Linear(hidden_nf, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, 1),
        )

        self.pot_com_model = nn.Sequential(
            nn.Linear(1 + 1 + self.max_nb_neighbors + self._sp_nf, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, 1),
        )
        self.counter = 0

    def _expand_t(self, t, batch_size, device, dtype):
        if not torch.is_tensor(t):
            return torch.full((batch_size, 1), float(t), device=device, dtype=dtype)
        if t.ndim == 0:
            return torch.full((batch_size, 1), float(t.item()), device=device, dtype=dtype)
        if t.ndim == 1:
            if t.shape[0] != batch_size:
                raise ValueError(f"Expected t shape [{batch_size}], got {tuple(t.shape)}")
            return t.view(batch_size, 1).to(device=device, dtype=dtype)
        t2 = t.reshape(batch_size, -1)
        return t2[:, :1].to(device=device, dtype=dtype)

    def _compute_common_terms(self, xs, t, a=None):
        B, P, D = xs.shape
        if P != self._n_particles or D != self._n_dimension:
            raise ValueError(
                f"Expected xs shape [B,{self._n_particles},{self._n_dimension}], got {tuple(xs.shape)}"
            )

        source_xs = xs
        neighbors = xs.unsqueeze(1).expand(B, P, P, D)
        source = xs.unsqueeze(2).expand(B, P, P, D)
        diffs = neighbors - source

        if self.L is not None:
            diffs = diffs - self.L * torch.round(diffs / self.L)
            neighbors = source + diffs

        eye = torch.eye(P, device=xs.device, dtype=torch.bool).view(1, P, P, 1)
        keep = ~eye.expand(B, -1, -1, D)
        neighbors = neighbors[keep].view(B, P, P - 1, D)
        diffs = diffs[keep].view(B, P, P - 1, D)

        distances = torch.norm(diffs, dim=-1)
        idx = torch.argsort(distances, dim=2)[:, :, : self.max_nb_neighbors]
        gather_idx = idx.unsqueeze(-1).expand(-1, -1, -1, D)
        neighbors = torch.gather(neighbors, 2, gather_idx)

        a_nbr = None
        if self.n_species > 1:
            if a is None:
                raise ValueError("Must provide species a [B,P] when n_species > 1.")
            keep_sp = ~torch.eye(P, device=xs.device, dtype=torch.bool).view(1, P, P)
            a_full = a.long().unsqueeze(1).expand(B, P, P)[keep_sp.expand(B, -1, -1)].view(B, P, P - 1)
            a_nbr = torch.gather(a_full, 2, idx)                       # [B,P,max_nb] neighbour species (same order)

        # Central particle expressed in the SAME cloud-centered frame as x_final (xs_centered subtracts the
        # cloud mean; skipping this for the periodic branch mixed frames: r_ij was |cloud-frame v_j - raw x_i|,
        # not the physical pair distance, and the velocity leaked absolute position). d(com)/d(x_i) = 0
        # (the min-imaged d_ij all shift with x_i and the mean cancels it), so d(diffij)/d(x_i) = -I still
        # holds exactly -> the analytic divergence is unchanged (re-verified by the brute-force test).
        com = neighbors.mean(dim=2)
        source_xs_com = source_xs - com
        rcom = source_xs_com.norm(dim=-1, keepdim=True) if self.L is None else None

        xs_centered = neighbors.reshape(B * P, self.max_nb_neighbors, D)
        xs_centered = xs_centered - xs_centered.mean(dim=1, keepdim=True)

        t_flat = self._expand_t(t, B, xs.device, xs.dtype)
        t_h = t_flat.repeat_interleave(P, dim=0)

        edges = self.compute_edges(xs_centered, cutoff=self.cutoff)
        x = xs_centered.reshape(-1, D)
        h = torch.ones(B * P, self.max_nb_neighbors, device=xs.device, dtype=xs.dtype)
        if self.condition_time:
            h = h * t_h
        h = h.reshape(-1, 1)
        if self.n_species > 1:                                         # append neighbour species one-hot (per cloud node)
            h_sp = F.one_hot(a_nbr.reshape(-1), self.n_species).to(h.dtype)
            h = torch.cat([h, h_sp], dim=-1)

        edge_attr = torch.sum((x[edges[0]] - x[edges[1]]) ** 2, dim=1, keepdim=True)
        h_final, x_final = self.egnn(h, x, edges, edge_attr=edge_attr)

        h_final = h_final.reshape(B, P, self.max_nb_neighbors, -1)
        vel = x_final.reshape(B, P, self.max_nb_neighbors, D)

        if self.L is None:
            diffij = source_xs_com.unsqueeze(2).expand(B, P, self.max_nb_neighbors, D) - vel
        else:
            diffij = vel - source_xs_com.unsqueeze(2).expand(B, P, self.max_nb_neighbors, D)

        t_neighbors = t_flat.view(B, 1, 1).expand(-1, P, self.max_nb_neighbors)

        # central-species one-hot per (central, neighbour) pair -> [B,P,max_nb,n_species] (position-independent)
        a_central = None
        if self.n_species > 1:
            a_central = F.one_hot(a.long(), self.n_species).to(diffij.dtype)
            a_central = a_central.unsqueeze(2).expand(B, P, self.max_nb_neighbors, self.n_species)

        return {
            "diffij": diffij,
            "h_final": h_final,
            "source_xs_com": source_xs_com,
            "rcom": rcom,
            "t_neighbors": t_neighbors,
            "t_flat": t_flat,
            "a_central": a_central,
            "B": B,
            "P": P,
        }

    def _sp_feat(self, common, n_neighbors):
        """Central-species one-hot flattened to [B*P*n_neighbors, sp_nf] (or empty), for the pot input."""
        if self.n_species > 1:
            return common["a_central"].reshape(-1, self.n_species)
        return None

    def forward(self, t, xs, a=None):
        common = self._compute_common_terms(xs, t, a)
        diffij = common["diffij"]
        h_final = common["h_final"]
        B = common["B"]
        P = common["P"]
        n_neighbors = diffij.shape[2]

        rij = diffij.norm(dim=-1).reshape(B * P * n_neighbors, 1)
        t_pot = common["t_neighbors"].reshape(-1, 1)
        h_flat = h_final.reshape(-1, h_final.shape[-1])
        sp = self._sp_feat(common, n_neighbors)

        parts = (rij, t_pot, h_flat) if sp is None else (rij, t_pot, h_flat, sp)
        pot = self.pot_model(torch.cat(parts, dim=-1)).reshape(B, P, n_neighbors, 1)

        if self.L is not None:
            return (diffij * pot).sum(dim=2)

        source_xs_com = common["source_xs_com"]
        rcom = common["rcom"]
        cparts = [rcom.reshape(-1, 1), common["t_flat"].reshape(-1, 1, 1).expand(-1, P, 1).reshape(-1, 1),
                  h_final[:, :, :, -1].reshape(B * P, -1)]
        if self.n_species > 1:
            cparts.append(common["a_central"][:, :, 0, :].reshape(B * P, self.n_species))
        com_pot = self.pot_com_model(torch.cat(cparts, dim=-1)).reshape(B, P, 1)

        return (diffij * pot).sum(dim=2) + (com_pot * source_xs_com)

    def forward_and_divergence(self, xs, t, a=None, differentiable=True):
        common = self._compute_common_terms(xs, t, a)
        diffij = common['diffij']
        h_final = common['h_final']
        B = common['B']
        P = common['P']
        n_neighbors = diffij.shape[2]

        t_pot = common['t_neighbors'].reshape(-1, 1)
        h_flat = h_final.reshape(-1, h_final.shape[-1])
        sp = self._sp_feat(common, n_neighbors)

        with torch.enable_grad():
            rij = rearrange(diffij.norm(dim=-1), 'b p m -> (b p m) 1').requires_grad_(True)
            parts = (rij, t_pot, h_flat) if sp is None else (rij, t_pot, h_flat, sp)
            pot = self.pot_model(torch.cat(parts, dim=-1)).reshape(B, P, n_neighbors, 1)

            vel = (diffij * pot).sum(dim=2)

            dpotdr = torch.autograd.grad(pot.sum(), rij, create_graph=differentiable, retain_graph=differentiable)[0]
            dpotdr = dpotdr.reshape(B, P, n_neighbors, 1)
            rij_reshaped = rij.reshape(B, P, n_neighbors, 1)

            divergence_term = (
                pot + (diffij * dpotdr * diffij / (rij_reshaped + 1e-8))
            ).sum((-1, -2))

        divergence = -divergence_term.sum(-1)
        self.counter += 1
        return vel, divergence

    def forward_and_perparticle_divergence(self, xs, t, a=None, differentiable=True):
        """Same as forward_and_divergence but returns per-particle divergence [B,P] (sum(-1) == the scalar div).
        Periodic (L set) branch only — the KA cluster flow uses L."""
        common = self._compute_common_terms(xs, t, a)
        diffij = common['diffij']; h_final = common['h_final']; B = common['B']; P = common['P']
        n_neighbors = diffij.shape[2]
        t_pot = common['t_neighbors'].reshape(-1, 1); h_flat = h_final.reshape(-1, h_final.shape[-1])
        sp = self._sp_feat(common, n_neighbors)
        with torch.enable_grad():
            rij = rearrange(diffij.norm(dim=-1), 'b p m -> (b p m) 1').requires_grad_(True)
            parts = (rij, t_pot, h_flat) if sp is None else (rij, t_pot, h_flat, sp)
            pot = self.pot_model(torch.cat(parts, dim=-1)).reshape(B, P, n_neighbors, 1)
            vel = (diffij * pot).sum(dim=2)
            dpotdr = torch.autograd.grad(pot.sum(), rij, create_graph=differentiable, retain_graph=differentiable)[0]
            dpotdr = dpotdr.reshape(B, P, n_neighbors, 1); rij_r = rij.reshape(B, P, n_neighbors, 1)
            divergence_term = (pot + (diffij * dpotdr * diffij / (rij_r + 1e-8))).sum((-1, -2))   # [B,P]
        return vel, -divergence_term                                                             # [B,P,2],[B,P]

    def divergence(self, xs, t, a=None, differentiable=True):
        common = self._compute_common_terms(xs, t, a)
        diffij = common['diffij']
        h_final = common['h_final']
        source_xs_com = common['source_xs_com']
        rcom = common['rcom']
        B = common['B']
        P = common['P']
        n_neighbors = diffij.shape[2]

        t_pot = common['t_neighbors'].reshape(-1, 1)
        h_flat = h_final.reshape(-1, h_final.shape[-1])
        sp = self._sp_feat(common, n_neighbors)

        with torch.enable_grad():
            rij = rearrange(diffij.norm(dim=-1), 'b p m -> (b p m) 1').requires_grad_(True)
            parts = (rij, t_pot, h_flat) if sp is None else (rij, t_pot, h_flat, sp)
            pot = self.pot_model(torch.cat(parts, dim=-1)).reshape(B, P, n_neighbors, 1)

            dpotdr = torch.autograd.grad(pot.sum(), rij, create_graph=differentiable, retain_graph=differentiable)[0]
            dpotdr = dpotdr.reshape(B, P, n_neighbors, 1)
            rij_reshaped = rij.reshape(B, P, n_neighbors, 1)

            divergence_pairs = (
                pot + (diffij * dpotdr * diffij / (rij_reshaped + 1e-8))
            ).sum((-1, -2))

            rcom_req = rcom.requires_grad_(True)
            cparts = [rcom_req.reshape(-1, 1),
                      common['t_flat'].reshape(-1, 1, 1).expand(-1, P, 1).reshape(-1, 1),
                      h_final[:, :, :, -1].reshape(B * P, -1)]
            if self.n_species > 1:
                cparts.append(common["a_central"][:, :, 0, :].reshape(B * P, self.n_species))
            com_pot = self.pot_com_model(torch.cat(cparts, dim=-1)).reshape(B, P, 1)

            dcomdr = torch.autograd.grad(com_pot.sum(), rcom_req, create_graph=differentiable, retain_graph=differentiable)[0]
            divergence_com = com_pot + dcomdr * rcom_req

        divergence = (divergence_pairs + divergence_com.squeeze(-1)).sum(-1)
        return divergence

    def compute_div(self, t, xs, a=None, return_f_val=False, e=None, make_e=None):
        del e, make_e  # This backend currently provides exact/custom divergence only.
        if self.L is not None:
            if return_f_val:
                return self.forward_and_divergence(xs, t, a)
            return self.forward_and_divergence(xs, t, a)[1]

        if return_f_val:
            vel = self.forward(t, xs, a)
            return vel, self.divergence(xs, t, a)
        return self.divergence(xs, t, a)
    
    def compute_edges(self, x, cutoff):
        # x: [B, P, D]
        B, P, _ = x.shape

        # x is the CONTIGUATED neighbour cloud (min-imaged about the central particle upstream) — non-periodic
        # by construction. Folding pairwise diffs with the box L here wrongly creates edges between genuinely
        # distant cloud members (phantom edges) whose edge_attr is the RAW difference — inconsistent. Raw only.
        pvec = x.unsqueeze(-3) - x.unsqueeze(-2)
        dist = torch.norm(pvec, dim=-1)

        eye = torch.eye(P, dtype=torch.bool, device=x.device).unsqueeze(0)
        mask = (dist < cutoff) & (~eye)
        batch_idx, rows, cols = torch.nonzero(mask, as_tuple=True)
        rows = rows + batch_idx * P
        cols = cols + batch_idx * P
        return [rows, cols]


class EGNN(nn.Module):
    def __init__(
        self,
        in_node_nf,
        in_edge_nf,
        hidden_nf,
        device="cpu",
        act_fn=nn.SiLU(),
        n_layers=4,
        recurrent=True,
        attention=False,
        norm_diff=True,
        out_node_nf=None,
        tanh=False,
        coords_range=15,
        agg="sum",
        L=None,
    ):
        super(EGNN, self).__init__()
        if out_node_nf is None:
            out_node_nf = in_node_nf
        self.hidden_nf = hidden_nf
        self.device = device
        self.n_layers = n_layers
        self.coords_range_layer = float(coords_range) / self.n_layers
        if agg == "mean":
            self.coords_range_layer = self.coords_range_layer * 19

        self.embedding = nn.Linear(in_node_nf, self.hidden_nf)
        self.embedding_out = nn.Linear(self.hidden_nf, out_node_nf)
        for i in range(0, n_layers):
            self.add_module(
                "gcl_%d" % i,
                E_GCL(
                    self.hidden_nf,
                    self.hidden_nf,
                    self.hidden_nf,
                    edges_in_d=in_edge_nf,
                    act_fn=act_fn,
                    recurrent=recurrent,
                    attention=attention,
                    norm_diff=norm_diff,
                    tanh=tanh,
                    coords_range=self.coords_range_layer,
                    agg=agg,
                    L=L,
                ),
            )
        self.to(self.device)

    def forward(self, h, x, edges, edge_attr=None, node_mask=None, edge_mask=None):
        h = self.embedding(h)
        for i in range(0, self.n_layers):
            h, x, _ = self._modules["gcl_%d" % i](
                h,
                edges,
                x,
                edge_attr=edge_attr,
                node_mask=node_mask,
                edge_mask=edge_mask,
            )
        h = self.embedding_out(h)
        if node_mask is not None:
            h = h * node_mask
        return h, x


class E_GCL(nn.Module):
    def __init__(
        self,
        input_nf,
        output_nf,
        hidden_nf,
        edges_in_d=0,
        nodes_att_dim=0,
        act_fn=nn.SiLU(),
        recurrent=True,
        attention=False,
        clamp=False,
        norm_diff=True,
        tanh=False,
        coords_range=1,
        agg="sum",
        L=None,
    ):
        super(E_GCL, self).__init__()
        input_edge = input_nf * 2
        self.recurrent = recurrent
        self.attention = attention
        self.norm_diff = norm_diff
        self.agg_type = agg
        self.tanh = tanh
        edge_coords_nf = 1

        self.edge_mlp = nn.Sequential(
            nn.Linear(input_edge + edge_coords_nf + edges_in_d, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, hidden_nf),
            act_fn,
        )

        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_nf + input_nf + nodes_att_dim, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, output_nf),
        )

        layer = nn.Linear(hidden_nf, 1, bias=False)
        torch.nn.init.xavier_uniform_(layer.weight, gain=0.001)

        coord_mlp = [nn.Linear(hidden_nf, hidden_nf), act_fn, layer]
        if self.tanh:
            coord_mlp.append(nn.Tanh())
            self.coords_range = coords_range

        self.coord_mlp = nn.Sequential(*coord_mlp)
        self.clamp = clamp

        if self.attention:
            self.att_mlp = nn.Sequential(nn.Linear(hidden_nf, 1), nn.Sigmoid())

        self.L = L
        if (agg != "sum") and (self.L is not None):
            raise ValueError("Only agg = 'sum' supported when L is not None.")

    def edge_model(self, source, target, radial, edge_attr, edge_mask):
        if edge_attr is None:
            out = torch.cat([source, target, radial], dim=1)
        else:
            out = torch.cat([source, target, radial, edge_attr], dim=1)
        out = self.edge_mlp(out)

        if self.attention:
            out = out * self.att_mlp(out)

        if edge_mask is not None:
            out = out * edge_mask
        return out

    def node_model(self, x, edge_index, edge_attr, node_attr):
        row, col = edge_index
        agg = unsorted_segment_sum(edge_attr, row, num_segments=x.size(0))
        if node_attr is not None:
            agg = torch.cat([x, agg, node_attr], dim=1)
        else:
            agg = torch.cat([x, agg], dim=1)
        out = self.node_mlp(agg)
        if self.recurrent:
            out = x + out
        return out, agg

    def coord_model(self, coord, edge_index, coord_diff, radial, edge_feat, node_mask, edge_mask):
        row, col = edge_index
        if self.tanh:
            trans = coord_diff * self.coord_mlp(edge_feat) * self.coords_range
        else:
            trans = coord_diff * self.coord_mlp(edge_feat)
        if edge_mask is not None:
            trans = trans * edge_mask

        if self.agg_type == "sum":
            unsorted_segment_sum(trans, row, num_segments=coord.size(0), out=coord, L=self.L)
        elif self.agg_type == "mean":
            if node_mask is not None:
                agg = unsorted_segment_sum(trans, row, num_segments=coord.size(0))
                M = unsorted_segment_sum(node_mask[col], row, num_segments=coord.size(0))
                coord += agg / (M - 1)
            else:
                unsorted_segment_mean(trans, row, num_segments=coord.size(0), out=coord)
        else:
            raise Exception("Wrong coordinates aggregation type")

        if self.L is not None:
            coord = torch.remainder(coord, self.L)
        return coord

    def forward(self, h, edge_index, coord, edge_attr=None, node_attr=None, node_mask=None, edge_mask=None):
        row, col = edge_index
        radial, coord_diff = self.coord2radial(edge_index, coord)

        edge_feat = self.edge_model(h[row], h[col], radial, edge_attr, edge_mask)
        coord = self.coord_model(coord, edge_index, coord_diff, radial, edge_feat, node_mask, edge_mask)

        h, agg = self.node_model(h, edge_index, edge_feat, node_attr)

        if node_mask is not None:
            h = h * node_mask
            coord = coord * node_mask
        return h, coord, edge_attr

    def coord2radial(self, edge_index, coord):
        row, col = edge_index
        if self.L is not None:
            coord_diff = log_map(coord[col], coord[row], self.L)
        else:
            coord_diff = coord[row] - coord[col]
        radial = torch.sum((coord_diff) ** 2, 1).unsqueeze(1)

        norm = torch.sqrt(radial + 1e-8)
        coord_diff = coord_diff / (norm + 1)
        return radial, coord_diff


def broadcast(src, other, dim):
    if dim < 0:
        dim = other.dim() + dim
    if src.dim() == 1:
        for _ in range(0, dim):
            src = src.unsqueeze(0)
    for _ in range(src.dim(), other.dim()):
        src = src.unsqueeze(-1)
    src = src.expand(other.size())
    return src


def scatter_sum(src, index, dim=-1, out=None, dim_size=None, L=None):
    index = broadcast(index, src, dim)
    if out is None:
        size = list(src.size())
        if dim_size is not None:
            size[dim] = dim_size
        elif index.numel() == 0:
            size[dim] = 0
        else:
            size[dim] = int(index.max()) + 1
        out = torch.zeros(size, dtype=src.dtype, device=src.device)
        if L is not None:
            return torch.remainder(torch.scatter_add(out, dim, index, src), L)
        return torch.scatter_add(out, dim, index, src)
    return out.scatter_add_(dim, index, src)


def scatter_mean(src, index, dim=-1, out=None, dim_size=None):
    out = scatter_sum(src, index, dim, out, dim_size)
    dim_size = out.size(dim)
    index_dim = dim
    if index_dim < 0:
        index_dim = index_dim + src.dim()
    if index.dim() <= index_dim:
        index_dim = index.dim() - 1
    ones = torch.ones(index.size(), dtype=src.dtype, device=src.device)
    count = scatter_sum(ones, index, index_dim, None, dim_size)
    count[count < 1] = 1
    count = broadcast(count, out, dim)
    if out.is_floating_point():
        out.true_divide_(count)
    else:
        out.div_(count, rounding_mode="floor")
    return out


def unsorted_segment_sum(data, segment_ids, num_segments, out=None, L=None):
    return scatter_sum(data, segment_ids, dim=0, dim_size=num_segments, out=out, L=L)


def unsorted_segment_mean(data, segment_ids, num_segments, out=None):
    return scatter_mean(data, segment_ids, dim=0, dim_size=num_segments, out=out)
