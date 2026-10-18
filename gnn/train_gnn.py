"""Unified training harness for geometry models on LnQM graphs.

Models (batch 1): schnet | gcn | gin | gat  (radius graph r=5.0 A)
Target: HOMO alpha (eV). Split: scaffold train/val/test from graphs_lnqm.pt.
Reference anchor: composition RF = 0.418 eV (test).

usage:
  python train_gnn.py --model schnet --subset 2000 --epochs 60
  python train_gnn.py --model gcn --epochs 300
"""
import argparse
import json
import time
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import (SchNet, GCNConv, GINConv, GATConv,
                                global_mean_pool, DimeNetPlusPlus)
import torch_geometric.nn.models.dimenet as _dimenet_mod

DATA = Path(os.environ.get("METAL_ML_DATA", "data"))
OUT = Path(os.environ.get("GRAPH_DIR", DATA / "graphs"))
R_EDGE = 5.0
MAXN = 32


def radius_edges(pos, r=R_EDGE, maxn=MAXN):
    """kNN-capped radius graph, pure torch (no torch_cluster / pyg-lib).

    Mimics torch_cluster.radius_graph for one molecule: for every node take at
    most maxn nearest neighbours within r; returns (2, E) edge_index.
    """
    n = pos.size(0)
    d = torch.cdist(pos, pos)
    d.fill_diagonal_(1e9)
    k = min(n - 1, maxn)
    dist, idx = torch.topk(d, k=k, largest=False, dim=1)
    src = torch.arange(n, device=pos.device).repeat_interleave(k)
    dst = idx.reshape(-1)
    m = dist.reshape(-1) < r
    return torch.stack([src[m], dst[m]])


def radius_graph_batch(x, batch=None, r=R_EDGE, max_num_neighbors=MAXN,
                       loop=False, flow="source_to_target", **kw):
    """Drop-in `interaction_graph` for PyG's SchNet: called as (pos, batch),
    returns (edge_index, edge_weight) with per-graph kNN-capped radius edges."""
    if batch is None:
        batch = x.new_zeros(x.size(0), dtype=torch.long)
    edges, weights = [], []
    for g in batch.unique():
        idx = (batch == g).nonzero(as_tuple=True)[0]
        e = radius_edges(x[idx], r=r, maxn=max_num_neighbors)
        edges.append(idx[e])
        weights.append((x[idx][e[0]] - x[idx][e[1]]).norm(dim=-1))
    return torch.cat(edges, dim=1), torch.cat(weights)


def radius_graph_idx(x, batch=None, r=R_EDGE, max_num_neighbors=MAXN, **kw):
    """edge_index-only variant (DimeNet's forward uses the single return)."""
    ei, _ = radius_graph_batch(x, batch=batch, r=r,
                               max_num_neighbors=max_num_neighbors)
    return ei


# DimeNet's forward calls module-level radius_graph (torch_cluster ext) ->
# rebind to the pure-torch implementation.
_dimenet_mod.radius_graph = radius_graph_idx


def triplets_pure(edge_index, num_nodes):
    """Pure-torch equivalent of torch_geometric.nn.models.dimenet.triplets
    (which needs torch-sparse's SparseTensor).

    Semantics (row, col = edge_index): for every edge t, enumerate all edges s
    with col[s] == row[t] (i.e. neighbours k of j); drop i == k. Returns
    (col, row, idx_i, idx_j, idx_k, idx_kj, idx_ji) with idx_kj/idx_ji the edge
    ids of (k<-j) and (j<-i) for each triplet.
    """
    row, col = edge_index
    E = row.numel()
    dev = row.device
    key = col.to(torch.int64) * (num_nodes + 1) + row.to(torch.int64)
    order = torch.argsort(key)                     # CSR-like (col, row) order
    col_sorted = col[order]
    lo = torch.searchsorted(col_sorted, row, right=False)
    hi = torch.searchsorted(col_sorted, row, right=True)
    cnt = (hi - lo).clamp(min=0)
    offsets = torch.repeat_interleave(lo, cnt)
    total = int(cnt.sum())
    starts = torch.repeat_interleave(torch.cumsum(cnt, 0) - cnt, cnt)
    s_pos = offsets + (torch.arange(total, device=dev) - starts)
    s_idx = order[s_pos]                           # edge ids s
    idx_t = torch.repeat_interleave(torch.arange(E, device=dev), cnt)
    idx_i = col[idx_t]
    idx_j = row[idx_t]
    idx_k = row[s_idx]
    mask = idx_i != idx_k
    return (col, row, idx_i[mask], idx_j[mask], idx_k[mask],
            s_idx[mask], idx_t[mask])


_dimenet_mod.triplets = triplets_pure

# ------------------------------------------------------------ ThreeD-GAT port
# Architecture adapted from a ThreeD_GAT implementation:
# edge-featured GAT + LayerNorm/residual + metal-centric readout
# [global-mean, global-max, metal-mean]. SMILES-bond features replaced by a
# metal-ligand edge flag (RBF-only configuration, BO = 0).
_EN = {1: 2.20, 5: 2.04, 6: 2.55, 7: 3.04, 8: 3.44, 9: 3.98,
       15: 2.19, 16: 2.58, 17: 3.16, 35: 2.96, 53: 2.66}
_RAD = {1: 0.31, 5: 0.84, 6: 0.76, 7: 0.71, 8: 0.66, 9: 0.57,
        15: 1.07, 16: 1.05, 17: 1.02, 35: 1.20, 53: 1.39}
_RBF_C = torch.linspace(1.0, R_EDGE, 16)
_RBF_S = 0.2
NODE_DIM, EDGE_DIM = 4, 17  # 4 node feats; 16 RBF + 1 metal-ligand flag


def node_feat(zs):
    feats = []
    for z in zs.tolist():
        is_ln = float(57 <= z <= 71)
        en = _EN.get(z, 1.20 if is_ln else 1.5) / 4.0
        rad = _RAD.get(z, 1.90 if is_ln else 0.9) / 2.0
        feats.append([z / 100.0, is_ln, en, rad])
    return torch.tensor(feats, dtype=torch.float32)


def edge_attr_of(pos, ei, zs):
    dist = (pos[ei[0]] - pos[ei[1]]).norm(dim=-1)
    rbf = torch.exp(-((dist.unsqueeze(-1) - _RBF_C.to(dist.device)) ** 2) / _RBF_S ** 2)
    is_ln = ((zs >= 57) & (zs <= 71)).float()
    ml = (is_ln[ei[0]] != is_ln[ei[1]]).float().unsqueeze(-1)
    return torch.cat([rbf, ml], dim=-1)


def _segment_softmax(scores, dst, N):
    scores = scores.float()
    mx = torch.full((N, scores.size(1)), float("-inf"), device=scores.device)
    mx.scatter_reduce_(0, dst.unsqueeze(-1).expand(-1, scores.size(1)), scores,
                       reduce="amax", include_self=False)
    exp = torch.exp(scores - mx[dst])
    sm = torch.zeros(N, scores.size(1), device=scores.device)
    sm.scatter_add_(0, dst.unsqueeze(-1).expand(-1, scores.size(1)), exp)
    return (exp / (sm[dst] + 1e-8)).to(scores.dtype)


class _ZwGATConv(torch.nn.Module):
    def __init__(self, in_dim, out_per_head, edge_dim, heads):
        super().__init__()
        self.heads, self.out_dim = heads, out_per_head
        total = out_per_head * heads
        self.node_lin = torch.nn.Linear(in_dim, total, bias=False)
        self.attn_src = torch.nn.Linear(total, heads, bias=False)
        self.attn_dst = torch.nn.Linear(total, heads, bias=False)
        self.edge_attn = torch.nn.Linear(edge_dim, heads, bias=False)

    def forward(self, x, ei, ea):
        src, dst = ei[0], ei[1]
        h = self.node_lin(x)
        attn = self.attn_src(h[src]) + self.attn_dst(h[dst])
        if ea is not None and ea.size(0) > 0:
            attn = attn + self.edge_attn(ea)
        attn = F.leaky_relu(attn, 0.2)
        alpha = _segment_softmax(attn, dst, x.size(0))
        hh = h.view(-1, self.heads, self.out_dim)
        msg = (hh[src] * alpha.unsqueeze(-1)).view(-1, self.out_dim * self.heads)
        out = torch.zeros(x.size(0), msg.size(1), device=x.device, dtype=msg.dtype)
        out.scatter_add_(0, dst.unsqueeze(-1).expand_as(msg), msg)
        return F.relu(out)


class ZwGATReg(torch.nn.Module):
    def __init__(self, node_dim=NODE_DIM, edge_dim=EDGE_DIM, hidden=384,
                 layers=5, heads=12, dropout=0.1):
        super().__init__()
        self.embed = torch.nn.Linear(node_dim, hidden)
        self.convs = torch.nn.ModuleList(
            [_ZwGATConv(hidden, hidden // heads, edge_dim, heads) for _ in range(layers)])
        self.norms = torch.nn.ModuleList([torch.nn.LayerNorm(hidden) for _ in range(layers)])
        self.drop = torch.nn.Dropout(dropout)
        self.readout = torch.nn.Sequential(
            torch.nn.Linear(hidden * 3, hidden), torch.nn.ReLU(),
            torch.nn.Dropout(dropout), torch.nn.Linear(hidden, hidden // 2),
            torch.nn.ReLU(), torch.nn.Dropout(dropout), torch.nn.Linear(hidden // 2, 1))

    def forward(self, batch):
        x, ei, ea, bt = batch.x, batch.edge_index, batch.edge_attr, batch.batch
        h = self.embed(x)
        for conv, norm in zip(self.convs, self.norms):
            h = norm(h + conv(h, ei, ea))
            h = self.drop(h)
        from torch_geometric.nn import global_mean_pool, global_max_pool
        gmean = global_mean_pool(h, bt)
        gmax = global_max_pool(h, bt)
        mask = x[:, 1] > 0.5                      # lanthanide atoms
        ng = int(bt.max()) + 1
        msum = torch.zeros(ng, h.size(1), device=h.device).index_add_(0, bt[mask], h[mask])
        mcnt = torch.zeros(ng, device=h.device).index_add_(
            0, bt[mask], torch.ones(int(mask.sum()), device=h.device))
        mmean = msum / mcnt.clamp(min=1).unsqueeze(-1)
        return self.readout(torch.cat([gmean, gmax, mmean], dim=1)).squeeze(-1)


def build_dataset(subset=None, with_attr=False, series="lnqm", multi=False,
                  target="homo", node_target=None, edge_target=None):
    path = OUT / ("graphs_lnqm.pt" if series == "lnqm" else f"graphs_{series}.pt")
    d = torch.load(path, weights_only=False)
    zs, pos = d["z"], d["pos"]
    tr, va, te = d["train_rows"], d["val_rows"], d["test_rows"]
    if edge_target:
        bo_pairs, bo_vals = d["bo_pairs"], d["bo_vals"]
    elif node_target:
        y = d[f"y_{node_target}"]
    elif multi:
        y = torch.stack([d["y_homo"], d["y_lumo"], d["y_gap"], d["y_dipole"]], dim=1)
    else:
        y = d[f"y_{target}"]
    if subset:
        # subsample TRAIN ONLY (keep full val/test for comparable evaluation)
        g = torch.Generator().manual_seed(42)
        tr = tr[torch.randperm(len(tr), generator=g)[:min(subset, len(tr))]]

    chg_t = d.get("charge")
    unp_t = d.get("unpaired")

    def make(idx):
        out = []
        for i in idx.tolist():
            ei = radius_edges(pos[i])
            kw = {}
            if with_attr:   # zw_gat: node features + RBF/metal-ligand edge attrs
                kw["x"] = node_feat(zs[i])
                kw["edge_attr"] = edge_attr_of(pos[i], ei, zs[i])
            if chg_t is not None:   # ccschnet conditioning (idx = charge+2 etc.)
                kw["charge"] = torch.tensor([int(chg_t[i]) + 2]).clamp(0, 7)
                kw["spin"] = torch.tensor([min(int(unp_t[i]), 9)])
            if edge_target:
                # Per-edge labels: recorded pairs take the bond-order value;
                # all other edges take 0 and are placed in the mask. Records
                # are aligned by atom-pair lookup, independent of edge order.
                na = int(zs[i].size(0))
                key_e = (torch.minimum(ei[0], ei[1]).to(torch.int64) * na
                         + torch.maximum(ei[0], ei[1]).to(torch.int64))
                bp = bo_pairs[i].to(torch.int64)
                key_r = bp[:, 0] * na + bp[:, 1]
                order = torch.argsort(key_r)
                key_s, val_s = key_r[order], bo_vals[i][order]
                ins = torch.searchsorted(key_s, key_e)
                valid = ins < key_s.numel()
                ins = ins.clamp(max=key_s.numel() - 1)
                hit = valid & (key_s[ins] == key_e)
                kw["edge_mask"] = hit.float()
                yv = torch.where(hit, val_s[ins], torch.zeros_like(val_s[ins]))
            elif node_target:
                yv = y[i]
            elif multi:
                yv = y[i].view(-1)
            else:
                yv = y[i].view(1)
            out.append(Data(z=zs[i], pos=pos[i], edge_index=ei, y=yv, **kw))
        return out

    return make(tr), make(va), make(te)


class GNNReg(torch.nn.Module):
    def __init__(self, kind, hidden=128, layers=3, cutoff=R_EDGE, out_dim=1):
        super().__init__()
        self.kind = kind
        self.out_dim = out_dim
        self.emb = torch.nn.Embedding(119, hidden)
        self.cutoff = cutoff
        if kind == "gcn":
            self.convs = torch.nn.ModuleList(
                [GCNConv(hidden, hidden) for _ in range(layers)])
        elif kind == "gin":
            self.convs = torch.nn.ModuleList(
                [GINConv(torch.nn.Sequential(
                    torch.nn.Linear(hidden, hidden), torch.nn.ReLU(),
                    torch.nn.Linear(hidden, hidden))) for _ in range(layers)])
        elif kind == "gat":
            self.convs = torch.nn.ModuleList(
                [GATConv(hidden, hidden // 4, heads=4) for _ in range(layers)])
        self.head = torch.nn.Sequential(
            torch.nn.Linear(hidden, hidden), torch.nn.ReLU(),
            torch.nn.Linear(hidden, out_dim))

    def forward(self, batch):
        x = self.emb(batch.z)
        for conv in self.convs:
            x = F.relu(conv(x, batch.edge_index)) + x
        g = global_mean_pool(x, batch.batch)
        out = self.head(g)
        return out.squeeze(-1) if self.out_dim == 1 else out


class DimeNetReg(torch.nn.Module):
    def __init__(self, hidden=128, cutoff=R_EDGE, maxn=MAXN, out_dim=1):
        super().__init__()
        self.out_dim = out_dim
        self.net = DimeNetPlusPlus(hidden_channels=hidden, out_channels=out_dim,
                                   num_blocks=4, int_emb_size=64,
                                   basis_emb_size=8, out_emb_channels=256,
                                   num_spherical=7, num_radial=6,
                                   cutoff=cutoff, max_num_neighbors=maxn)
        # PyG's DimeNet emb table is 95 (< 5f actinides Z up to 103): enlarge
        # the INNER table of EmbeddingBlock (not the block itself!).
        self.net.emb.emb = torch.nn.Embedding(120, hidden)

    def forward(self, batch):
        out = self.net(batch.z, batch.pos, batch.batch)
        return out.squeeze(-1) if self.out_dim == 1 else out


class _CFConv(torch.nn.Module):
    """Continuous-filter convolution (own implementation, no PyG components)."""

    def __init__(self, in_ch, out_ch, n_rbf):
        super().__init__()
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(n_rbf, 2 * out_ch), torch.nn.ReLU(),
            torch.nn.Linear(2 * out_ch, out_ch))

    def forward(self, x, ei, W):
        src, dst = ei
        msg = x[src] * self.mlp(W)
        out = torch.zeros_like(x)
        out.scatter_add_(0, dst.unsqueeze(-1).expand_as(msg), msg)
        return out


class CCSchnetReg(torch.nn.Module):
    """Charge-conditioned SchNet (the CC-Schnet model).

    Motivation: SchNet and DimeNet see geometry only; charge/spin enter solely
    through geometry. This variant injects charge and spin embeddings directly
    into the per-atom features of a SchNet-style interaction stack, isolating
    the independent value of charge information inside a geometric network.

    Fully self-contained (own RBF + continuous-filter convolution).
    """

    def __init__(self, hidden=128, layers=6, cutoff=R_EDGE,
                 n_charge=8, n_spin=10, out_dim=1, edge_head=False):
        super().__init__()
        self.out_dim = out_dim
        self.node_mode = False
        self.edge_mode = False
        self.emb = torch.nn.Embedding(120, hidden)
        self.charge_emb = torch.nn.Embedding(n_charge, hidden)
        self.spin_emb = torch.nn.Embedding(n_spin, hidden)
        self.register_buffer("rbf_centers", torch.linspace(0.0, cutoff, hidden))
        self.rbf_gamma = 10.0 / cutoff
        self.interactions = torch.nn.ModuleList([
            _CFConv(hidden, hidden, hidden) for _ in range(layers)])
        self.norms = torch.nn.ModuleList(
            [torch.nn.LayerNorm(hidden) for _ in range(layers)])
        self.head = torch.nn.Sequential(
            torch.nn.Linear(hidden, hidden // 2), torch.nn.ReLU(),
            torch.nn.Linear(hidden // 2, out_dim))
        # The edge head is built only for edge-level tasks, keeping the RNG
        # stream of existing tasks unchanged. Input: final-layer features of
        # the two endpoint nodes concatenated with the edge-distance RBF
        # (hidden dims each).
        if edge_head:
            self.edge_head = torch.nn.Sequential(
                torch.nn.Linear(3 * hidden, hidden), torch.nn.ReLU(),
                torch.nn.Linear(hidden, hidden // 2), torch.nn.ReLU(),
                torch.nn.Linear(hidden // 2, 1))

    def forward(self, batch):
        x = self.emb(batch.z)
        ei = batch.edge_index
        dist = (batch.pos[ei[0]] - batch.pos[ei[1]]).norm(dim=-1)
        W = torch.exp(-self.rbf_gamma * (dist.unsqueeze(-1)
                                         - self.rbf_centers) ** 2)
        # charge/spin conditioning: broadcast per-graph embedding to atoms
        cond = (self.charge_emb(batch.charge)
                + self.spin_emb(batch.spin))[batch.batch]
        x = x + cond
        for conv, norm in zip(self.interactions, self.norms):
            x = norm(x + conv(x, ei, W))
        if self.edge_mode:
            # Concatenate (h_i, h_j, RBF_ij) per edge, pass through an MLP,
            # and output the bond order.
            h_in = torch.cat([x[ei[0]], x[ei[1]], W], dim=-1)
            return self.edge_head(h_in).squeeze(-1)
        if self.node_mode:
            out = self.head(x)
            return out.squeeze(-1) if self.out_dim == 1 else out
        from torch_geometric.nn import global_mean_pool
        g = global_mean_pool(x, batch.batch)
        out = self.head(g)
        return out.squeeze(-1) if self.out_dim == 1 else out


class SchNetReg(torch.nn.Module):
    def __init__(self, hidden=128, layers=6, cutoff=R_EDGE, out_dim=1):
        super().__init__()
        self.out_dim = out_dim
        # PyG's SchNet already reads out a scalar per graph (pool + lin1/lin2)
        self.net = SchNet(hidden_channels=hidden, num_filters=hidden,
                          num_interactions=layers, cutoff=cutoff,
                          interaction_graph=radius_graph_batch)
        # PyG's SchNet emb table is 100 (< 5f actinides Z up to 103): enlarge.
        self.net.embedding = torch.nn.Embedding(120, hidden, padding_idx=0)
        if getattr(self.net, "atomref", None) is not None:
            self.net.atomref = torch.nn.Embedding(120, 1)
        if out_dim != 1:
            self.net.lin2 = torch.nn.Linear(hidden // 2, out_dim)

    def forward(self, batch):
        out = self.net(batch.z, batch.pos, batch.batch)
        return out.squeeze(-1) if self.out_dim == 1 else out


def evaluate(model, loader, ym, ys, device, node_mode=False, edge_mode=False):
    model.eval()
    mae, n = 0.0, 0
    with torch.no_grad():
        for b in loader:
            b = b.to(device)
            p = model(b) * ys + ym
            if edge_mode:
                err = (p - b.y).abs() * b.edge_mask
                mae += err.sum().item()
                n += int(b.edge_mask.sum().item())
                continue
            err = (p - b.y.view(p.shape)).abs()
            if err.dim() > 1:
                mae += err.mean(1).sum().item()
                n += b.num_graphs
            elif node_mode:
                mae += err.sum().item()
                n += err.numel()
            else:
                mae += err.sum().item()
                n += b.num_graphs
    return mae / max(n, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="schnet",
                    choices=["schnet", "gcn", "gin", "gat", "dimenet", "zw_gat",
                             "ccschnet"])
    ap.add_argument("--subset", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--series", default="lnqm",
                    choices=["lnqm", "3d", "4d", "5d", "5f"])
    ap.add_argument("--multi", action="store_true",
                    help="multi-task: homo/lumo/gap/dipole jointly")
    ap.add_argument("--target", default="homo",
                    choices=["homo", "lumo", "gap", "dipole", "pol"],
                    help="single-target key y_<target> (ignored with --multi)")
    ap.add_argument("--node-target", default=None,
                    choices=["hirsh_q", "hirsh_s"],
                    help="node-level regression (atomic charge/spin, "
                         "ccschnet only)")
    ap.add_argument("--edge-target", default=None,
                    choices=["bo"],
                    help="edge-level regression (Mayer bond order, "
                         "ccschnet only)")
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--layers", type=int, default=None)
    args = ap.parse_args()
    hidden_explicit = args.hidden != 128
    layers_explicit = args.layers is not None
    if args.layers is None:
        args.layers = 6 if args.model in ("schnet", "ccschnet") else (
            4 if args.model == "dimenet" else 3)
    if args.smoke:
        args.subset = args.subset or 2000
        args.epochs = min(args.epochs, 60)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if dev == "cpu":
        print("WARNING: CUDA unavailable -> running on CPU (much slower). "
              "If this is a GPU node, check the driver / restart the job.",
              flush=True)
    else:
        print(f"device: cuda ({torch.cuda.get_device_name(0)})", flush=True)
    tr, va, te = build_dataset(args.subset, with_attr=(args.model == "zw_gat"),
                               series=args.series, multi=args.multi,
                               target=args.target, node_target=args.node_target,
                               edge_target=args.edge_target)
    trl = DataLoader(tr, batch_size=args.batch, shuffle=True)
    val = DataLoader(va, batch_size=args.batch)
    tel = DataLoader(te, batch_size=args.batch)
    if args.edge_target:
        v = torch.cat([d.y[d.edge_mask > 0.5] for d in tr])
        ym = v.mean().to(dev)
        ys = v.std().to(dev)
    elif args.node_target:
        ym = torch.cat([d.y for d in tr]).mean().to(dev)
        ys = torch.cat([d.y for d in tr]).std().to(dev)
    else:
        ym = torch.stack([d.y for d in tr]).mean(0).to(dev)
        ys = torch.stack([d.y for d in tr]).std(0).to(dev)
    out_dim = ym.numel()

    if args.model == "schnet":
        model = SchNetReg(hidden=args.hidden, layers=args.layers,
                          out_dim=out_dim).to(dev)
    elif args.model == "dimenet":
        model = DimeNetReg(hidden=args.hidden, out_dim=out_dim).to(dev)
    elif args.model == "zw_gat":
        model = ZwGATReg().to(dev)
    elif args.model == "ccschnet":
        model = CCSchnetReg(hidden=args.hidden, layers=args.layers,
                            out_dim=out_dim,
                            edge_head=bool(args.edge_target)).to(dev)
    else:
        model = GNNReg(args.model, hidden=args.hidden, layers=args.layers,
                       out_dim=out_dim).to(dev)
    if args.node_target:
        if args.model != "ccschnet":
            raise SystemExit("node-level targets require the ccschnet model")
        model.node_mode = True
    if args.edge_target:
        if args.model != "ccschnet":
            raise SystemExit("edge-level targets require the ccschnet model")
        if args.node_target:
            raise SystemExit("node-level and edge-level targets are mutually "
                             "exclusive")
        model.edge_mode = True
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=15)

    best, best_ep, bad = 1e9, 0, 0
    t0 = time.time()
    for ep in range(1, args.epochs + 1):
        model.train()
        tot, n = 0.0, 0
        for b in trl:
            b = b.to(dev)
            opt.zero_grad()
            p = model(b) * ys + ym
            if args.edge_target:
                err = (p - b.y).abs() * b.edge_mask
                loss = err.sum() / b.edge_mask.sum().clamp(min=1.0)
            else:
                loss = F.l1_loss(p, b.y.view(p.shape))
            loss.backward()
            opt.step()
            tot += loss.item() * b.num_graphs
            n += b.num_graphs
        vmae = evaluate(model, val, ym, ys, dev,
                        node_mode=bool(args.node_target),
                        edge_mode=bool(args.edge_target))
        sched.step(vmae)
        if vmae < best - 1e-4:
            best, best_ep, bad = vmae, ep, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
        if ep % 10 == 0 or ep == 1:
            print(f"[{args.model}] ep{ep:4d} train {tot/n:.4f} val {vmae:.4f} "
                  f"best {best:.4f}@{best_ep} ({time.time()-t0:.0f}s)", flush=True)
        if bad >= 50:
            print(f"early stop at ep{ep}", flush=True)
            break
    model.load_state_dict(best_state)
    tmae = evaluate(model, tel, ym, ys, dev,
                    node_mode=bool(args.node_target),
                    edge_mode=bool(args.edge_target))
    # save test predictions (parity / audit)
    preds, trues, zall = [], [], []
    esrc, edst, emask, emol = [], [], [], []
    seen = 0
    model.eval()
    with torch.no_grad():
        for b in tel:
            b = b.to(dev)
            p = model(b) * ys + ym
            preds.append(p.cpu())
            trues.append(b.y.view(p.shape).cpu())
            if args.node_target:
                zall.append(b.z.cpu())
            if args.edge_target:
                ei = b.edge_index
                gb = b.batch[ei[0]]
                esrc.append((ei[0] - b.ptr[gb]).cpu())
                edst.append((ei[1] - b.ptr[b.batch[ei[1]]]).cpu())
                emask.append(b.edge_mask.cpu())
                emol.append(gb.cpu() + seen)
            seen += b.num_graphs
    P = torch.cat(preds).numpy()
    T = torch.cat(trues).numpy()
    if args.edge_target:
        M = torch.cat(emask).numpy()
        mae_all = float(np.abs(P - T).mean())
    if args.node_target:
        tgt = f"node_{args.node_target.replace('_', '')}_"
    elif args.edge_target:
        tgt = f"edge_{args.edge_target}_"
    else:
        tgt = "" if (args.target == "homo" or args.multi) else f"{args.target}_"
    tag = f"{args.series}_{'multi_' if args.multi else tgt}{'smoke' if args.smoke else 'full'}"
    if args.seed != 42:
        tag += f"_s{args.seed}"
    if hidden_explicit or layers_explicit:
        tag += f"_h{args.hidden}l{args.layers}"
    if args.edge_target:
        np.savez(OUT / f"preds_{args.model}_{tag}_seed{args.seed}.npz",
                 pred=P, true=T, mask=M.astype(bool),
                 mol=torch.cat(emol).numpy(), src=torch.cat(esrc).numpy(),
                 dst=torch.cat(edst).numpy())
    elif args.node_target:
        np.savez(OUT / f"preds_{args.model}_{tag}_seed{args.seed}.npz",
                 pred=P, true=T, z=torch.cat(zall).numpy())
    else:
        np.savez(OUT / f"preds_{args.model}_{tag}_seed{args.seed}.npz",
                 pred=P, true=T)
    if args.multi:
        names = ["homo", "lumo", "gap", "dipole"]
        per = np.abs(P - T).mean(0)
        res = {"model": args.model, "series": args.series, "multi": True,
               "subset": args.subset or len(tr), "seed": args.seed,
               "hidden": args.hidden, "layers": args.layers, "lr": args.lr,
               "best_val_mae_eV": round(best, 4), "best_epoch": best_ep,
               "test_mae": {names[k]: round(float(per[k]), 4) for k in range(len(names))},
               "anchor_composition_rf": 0.418,
               "anchor_note": "anchor is the HOMO random-forest value"}
        print(f"== {args.model} ({tag}) == " +
              " ".join(f"{names[k]} {per[k]:.4f}" for k in range(len(names))), flush=True)
    else:
        if args.node_target:
            unit = "e"
            res_target = args.node_target
        elif args.edge_target:
            unit = "1"
            res_target = f"mayer_{args.edge_target}"
        else:
            unit = {"homo": "eV", "lumo": "eV", "gap": "eV",
                    "dipole": "D", "pol": "a.u."}[args.target]
            res_target = args.target
        res = {"model": args.model, "series": args.series,
               "subset": args.subset or len(tr), "seed": args.seed,
               "target": res_target, "unit": unit,
               "best_val_mae_eV": round(best, 4), "best_epoch": best_ep,
               "test_mae_eV": round(tmae, 4),
               "anchor_composition_rf": (0.418 if (args.target == "homo"
                                                   and not args.node_target
                                                   and not args.edge_target) else None)}
        if args.edge_target:
            res.update({
                "test_mae_all_edges": round(mae_all, 4),
                "n_test_edges": int(P.size),
                "n_masked_edges": int(M.sum()),
                "label_rule": "pairs with |Mayer BO| >= 0.1 are recorded; "
                              "unrecorded graph edges carry label 0 and are "
                              "excluded from the loss and from the masked MAE"})
            print(f"== {args.model} ({tag}) == val {best:.4f}  "
                  f"TEST {tmae:.4f} (masked edges)  all edges {mae_all:.4f}",
                  flush=True)
        else:
            print(f"== {args.model} ({tag}) == val {best:.4f}  TEST {tmae:.4f} "
                  f"{unit} (anchor RF 0.418)", flush=True)
    (OUT / f"result_{args.model}_{tag}.json").write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
