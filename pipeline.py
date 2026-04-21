import argparse
import json
import math
import os
import random
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import BatchNorm, GATConv
from torch_geometric.utils import degree


def seed_all(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def load_cora(data_root: Optional[str]) -> Tuple[object, int]:
    root = data_root if data_root is not None else os.path.join(os.path.expanduser("~"), "datasets", "Cora")
    ds = Planetoid(root=root, name="Cora", transform=T.NormalizeFeatures())
    return ds, ds.num_classes


def select_features_for_batch_prefix(original_x: torch.Tensor, rng: random.Random) -> torch.Tensor:
    min_features = original_x.shape[1] * 5 // 10
    max_features = original_x.shape[1]
    feature_size = rng.randint(min_features, max_features)
    return original_x[:, 0:feature_size]


def apply_prefix_keep_with_fixed_dim(x_full: torch.Tensor, keep_ratio: float) -> torch.Tensor:
    x_view = x_full.clone()
    keep_dim = max(1, int(round(x_full.size(1) * keep_ratio)))
    keep_dim = min(keep_dim, x_full.size(1))
    if keep_dim < x_full.size(1):
        x_view[:, keep_dim:] = 0.0
    return x_view


def mask_feature_rows_fixed_dim(x: torch.Tensor, node_p: float, seed: int) -> Tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator(device=x.device)
    g.manual_seed(seed)
    mask = torch.rand((x.size(0),), generator=g, device=x.device) < node_p
    out = x.clone()
    out[mask, :] = 0.0
    return out, mask


def dropout_edge_with_seed(edge_index: torch.Tensor, edge_p: float, seed: int) -> Tuple[torch.Tensor, torch.Tensor]:
    num_edges = edge_index.size(1)
    g = torch.Generator(device=edge_index.device)
    g.manual_seed(seed)
    stay_prob = torch.zeros(num_edges, device=edge_index.device) + 1 - edge_p
    stay = torch.bernoulli(stay_prob, generator=g).to(torch.bool)
    dropped_mask = ~stay
    row, col = edge_index[0][stay], edge_index[1][stay]
    return torch.stack([row, col], dim=0).long(), dropped_mask


def make_eval_view_with_dig_protocol(
    x_full: torch.Tensor,
    edge_index: torch.Tensor,
    feature_keep_ratio: float,
    node_p: float,
    edge_p: float,
    seed: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    x_view = apply_prefix_keep_with_fixed_dim(x_full, feature_keep_ratio)
    x_view, _ = mask_feature_rows_fixed_dim(x_view, node_p=node_p, seed=seed + 17)
    edge_view, _ = dropout_edge_with_seed(edge_index, edge_p=edge_p, seed=seed + 29)
    return x_view, edge_view


def structural_teacher_fallback(data, emb_dim: int, device: torch.device) -> torch.Tensor:
    row = data.edge_index[0]
    deg = degree(row, num_nodes=data.num_nodes, dtype=torch.float32).to(device)
    deg = deg.unsqueeze(1)
    feat = torch.cat([deg, torch.log1p(deg), (deg > 0).float(), torch.ones_like(deg)], dim=1)
    proj = torch.randn(feat.size(1), emb_dim, device=device)
    return feat @ proj


def pca_teacher_torch(x: torch.Tensor, ratio: float, max_rank: int = 256) -> torch.Tensor:
    if ratio >= 1.0:
        return x.clone()
    _, d = x.shape
    q = max(2, min(max_rank, d, int(math.ceil(max(2, d * ratio)))))
    x_centered = x - x.mean(dim=0, keepdim=True)
    try:
        _, _, V = torch.pca_lowrank(x_centered, q=q, center=False)
        return x_centered @ V[:, :q]
    except Exception:
        _, _, Vh = torch.linalg.svd(x_centered, full_matrices=False)
        q = min(q, Vh.size(0))
        return x_centered @ Vh[:q].t()


class PredictionMLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.layer2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer2(self.layer1(x))


class DynamicGAE(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_num: int,
        head: int,
        out_channels: int,
        emb_size_1: int,
        emb_size_2: int,
        dropout: float,
    ):
        super().__init__()
        self.hidden_num = hidden_num
        self.head = head
        self.dropout = dropout
        self.conv1 = GATConv(in_channels, hidden_num, heads=head, dropout=dropout)
        self.conv2 = GATConv(hidden_num * head, out_channels, heads=1, concat=False, dropout=dropout)
        self.dictionary = nn.Parameter(torch.randn(in_channels, in_channels))
        self.bn1 = BatchNorm(hidden_num * head, momentum=0.99)
        self.p_1 = PredictionMLP(out_channels, out_channels * 2, emb_size_1)
        self.p_2 = PredictionMLP(out_channels, out_channels * 2, emb_size_2)
        self.p_12 = PredictionMLP(out_channels, out_channels * 2, emb_size_1 + emb_size_2)
        self.activation = nn.ELU()

    def sparse_encode(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(torch.matmul(x, self.dictionary))

    def sparse_decode(self, z: torch.Tensor) -> torch.Tensor:
        return torch.matmul(self.dictionary, z.t()).t()

    def encoder(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.activation(self.bn1(self.conv1(x, edge_index)))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.activation(self.conv2(x, edge_index))
        return x

    def decoder_all(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.p_1(z), self.p_2(z), self.p_12(z)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor):
        z = self.encoder(x, edge_index)
        embs = self.decoder_all(z)
        z_sparse = self.sparse_encode(x)
        emb_sparse = self.sparse_decode(z_sparse)
        return embs, emb_sparse

    def adjust_input_dim(self, new_in_channels: int, device: torch.device) -> None:
        self.conv1 = GATConv(new_in_channels, self.hidden_num, heads=self.head, dropout=self.dropout).to(device)
        self.bn1 = BatchNorm(self.hidden_num * self.head, momentum=0.99).to(device)

    def adjust_dictionary_size(self, new_in_channels: int, device: torch.device) -> None:
        self.dictionary = nn.Parameter(torch.randn(new_in_channels, new_in_channels, device=device))


def sim(z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
    z1 = F.normalize(z1)
    z2 = F.normalize(z2)
    return torch.mm(z1, z2.t())


def semi_loss(z1: torch.Tensor, z2: torch.Tensor, tau: float) -> torch.Tensor:
    f = lambda x: torch.exp(x / tau)
    refl_sim = f(sim(z1, z1))
    between_sim = f(sim(z1, z2))
    return -torch.log(between_sim.diag() / (refl_sim.sum(1) + between_sim.sum(1) - refl_sim.diag()))


def mutual_information_loss_dynamic_linear(emb1: torch.Tensor, emb2: torch.Tensor, epsilon: float = 1e-8) -> torch.Tensor:
    emb_dim = emb1.size(1)
    recon_dim = emb2.size(1)
    projection = nn.Linear(emb_dim, recon_dim).to(emb1.device)
    emb1 = projection(emb1)
    attention_weights = torch.mm(emb1, emb2.T)
    attention_weights = F.softmax(attention_weights, dim=1)
    emb1_aligned = torch.mm(attention_weights, emb2)
    emb1_norm = F.normalize(emb1_aligned, p=2, dim=1)
    emb2_norm = F.normalize(emb2, p=2, dim=1)
    similarity_matrix = torch.mm(emb1_norm, emb2_norm.T)
    similarity_matrix = torch.clamp(similarity_matrix, min=epsilon, max=1.0)
    positive_pairs = torch.diag(similarity_matrix)
    return -torch.mean(torch.log(positive_pairs / (similarity_matrix.sum(dim=1) + epsilon)))


@dataclass
class TrainConfig:
    seed: int = 1
    device: str = "cuda:0"
    epoch: int = 30
    lr: float = 0.01
    lrdec_1: float = 0.8
    lrdec_2: int = 200
    dropout: float = 0.6
    tau: float = 0.7
    hidden_num: int = 64
    head: int = 4
    out_channels: int = 128
    ratio: float = 0.3
    embedding_dim: int = 32
    node_p: float = 0.7
    edge_p: float = 0.7
    l1_e: float = 4.0
    l2_e: float = 1.0
    l12_e: float = 1.0
    l1_f: float = 1.0
    l2_f: float = 4.0
    l12_f: float = 1.0
    l1_b: float = 2.0
    l2_b: float = 2.0
    l12_b: float = -1.0
    all1: float = 1.0
    all2: float = 1.0
    all3: float = 1.0


def adjust_learning_rate(optimizer: torch.optim.Optimizer, epoch: int, base_lr: float, decay_factor: float, decay_step: int) -> None:
    lr = base_lr * (decay_factor ** (epoch // decay_step))
    for pg in optimizer.param_groups:
        pg["lr"] = lr


def build_backbone_and_teachers(cfg: TrainConfig, data_root: Optional[str]):
    seed_all(cfg.seed)
    device = torch.device(cfg.device if torch.cuda.is_available() and "cuda" in cfg.device else "cpu")
    dataset, num_classes = load_cora(data_root)
    data = dataset[0].to(device)
    emb_1 = structural_teacher_fallback(data, cfg.embedding_dim, device)
    emb_2 = data.x if cfg.ratio >= 1.0 else pca_teacher_torch(data.x, cfg.ratio)
    model = DynamicGAE(
        in_channels=data.x.size(1),
        hidden_num=cfg.hidden_num,
        head=cfg.head,
        out_channels=cfg.out_channels,
        emb_size_1=emb_1.size(1),
        emb_size_2=emb_2.size(1),
        dropout=cfg.dropout,
    ).to(device)
    return device, data, num_classes, emb_1, emb_2, model


def train_backbone(cfg: TrainConfig, data_root: Optional[str]):
    device, data, num_classes, emb_1, emb_2, model = build_backbone_and_teachers(cfg, data_root)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    rng = random.Random(cfg.seed)
    original_x = data.x.clone()
    original_edge_index = data.edge_index.clone()

    for epoch in range(1, cfg.epoch + 1):
        model.train()
        if device.type == "cuda":
            torch.cuda.empty_cache()

        data.x = select_features_for_batch_prefix(original_x, rng)
        data.edge_index = original_edge_index
        model.adjust_input_dim(data.x.size(1), device)
        model.adjust_dictionary_size(data.x.size(1), device)
        adjust_learning_rate(optimizer, epoch, cfg.lr, cfg.lrdec_1, cfg.lrdec_2)

        mask_x, mask_index_node_binary = mask_feature_rows_fixed_dim(data.x, cfg.node_p, seed=cfg.seed + epoch * 7)
        mask_edge, mask_index_edge = dropout_edge_with_seed(data.edge_index, cfg.edge_p, seed=cfg.seed + epoch * 11)

        mask_edge_node = mask_index_edge * data.edge_index[0]
        mask_index_edge_binary = torch.zeros(data.x.shape[0], device=device)
        mask_index_edge_binary[mask_edge_node] = 1
        mask_index_edge_binary = mask_index_edge_binary.to(torch.bool)
        mask_both_node_edge = mask_index_edge_binary & mask_index_node_binary
        mask_index_node_binary_sole = mask_index_node_binary & (~mask_both_node_edge)
        mask_index_edge_binary_sole = mask_index_edge_binary & (~mask_both_node_edge)

        optimizer.zero_grad()
        embs, emb_sparse = model(mask_x, mask_edge)
        recon_emb_1, recon_emb_2, recon_12 = embs

        loss_feat_recon = semi_loss(emb_sparse, mask_x, cfg.tau).mean()
        loss_mi = mutual_information_loss_dynamic_linear(emb_sparse, recon_12)

        z = torch.zeros(1, device=device)
        loss1_f = semi_loss(emb_1[mask_index_node_binary_sole], recon_emb_1[mask_index_node_binary_sole], cfg.tau) if mask_index_node_binary_sole.any() else z
        loss1_e = semi_loss(emb_1[mask_index_edge_binary_sole], recon_emb_1[mask_index_edge_binary_sole], cfg.tau) if mask_index_edge_binary_sole.any() else z
        loss1_b = semi_loss(emb_1[mask_both_node_edge], recon_emb_1[mask_both_node_edge], cfg.tau) if mask_both_node_edge.any() else z

        loss2_f = semi_loss(emb_2[mask_index_node_binary_sole], recon_emb_2[mask_index_node_binary_sole], cfg.tau) if mask_index_node_binary_sole.any() else z
        loss2_e = semi_loss(emb_2[mask_index_edge_binary_sole], recon_emb_2[mask_index_edge_binary_sole], cfg.tau) if mask_index_edge_binary_sole.any() else z
        loss2_b = semi_loss(emb_2[mask_both_node_edge], recon_emb_2[mask_both_node_edge], cfg.tau) if mask_both_node_edge.any() else z

        target12 = torch.cat((emb_1, emb_2), dim=1)
        loss12_f = semi_loss(target12[mask_index_node_binary_sole], recon_12[mask_index_node_binary_sole], cfg.tau) if mask_index_node_binary_sole.any() else z
        loss12_e = semi_loss(target12[mask_index_edge_binary_sole], recon_12[mask_index_edge_binary_sole], cfg.tau) if mask_index_edge_binary_sole.any() else z
        loss12_b = semi_loss(target12[mask_both_node_edge], recon_12[mask_both_node_edge], cfg.tau) if mask_both_node_edge.any() else z

        loss_e = cfg.l1_e * loss1_e.mean() + cfg.l2_e * loss2_e.mean() + cfg.l12_e * loss12_e.mean()
        loss_f = cfg.l1_f * loss1_f.mean() + cfg.l2_f * loss2_f.mean() + cfg.l12_f * loss12_f.mean()
        loss_both = cfg.l1_b * loss1_b.mean() + cfg.l1_b * loss2_b.mean() + cfg.l12_b * loss12_b.mean()

        loss_stct_recon = loss_e + loss_f + loss_both
        info_loss = cfg.all1 * loss_stct_recon + cfg.all2 * loss_feat_recon + cfg.all3 * loss_mi * 0.2
        info_loss.backward()
        optimizer.step()

        if epoch == 1 or epoch == cfg.epoch or epoch % max(1, cfg.epoch // 5) == 0:
            print(f"  [train epoch {epoch}/{cfg.epoch}] loss={float(info_loss.detach().item()):.4f}", flush=True)

    data.x = original_x
    data.edge_index = original_edge_index
    model.adjust_input_dim(data.x.size(1), device)
    model.to(device)
    model.eval()
    return {"device": device, "data": data, "num_classes": num_classes, "emb_1": emb_1, "emb_2": emb_2, "model": model}


def anomaly_scores_from_model(
    model: DynamicGAE,
    x: torch.Tensor,
    edge_index: torch.Tensor,
    emb_1: torch.Tensor,
    emb_2: torch.Tensor,
    tau: float,
) -> torch.Tensor:
    model.eval()
    z = model.encoder(x, edge_index)
    recon_emb_1, recon_emb_2, recon_12 = model.decoder_all(z)
    s1 = semi_loss(emb_1, recon_emb_1, tau)
    s2 = semi_loss(emb_2, recon_emb_2, tau)
    s12 = semi_loss(torch.cat((emb_1, emb_2), dim=1), recon_12, tau)
    return s1 + s2 + s12


def select_target_nodes(scores: torch.Tensor, top_targets: int) -> torch.Tensor:
    k = min(int(top_targets), int(scores.numel()))
    return torch.topk(scores, k=k).indices


def candidate_nodes_from_targets(edge_index: torch.Tensor, target_nodes: torch.Tensor, max_candidates: int) -> torch.Tensor:
    target_set = set(target_nodes.detach().cpu().tolist())
    row, col = edge_index.detach().cpu()
    cand = set(target_set)
    for u, v in zip(row.tolist(), col.tolist()):
        if u in target_set or v in target_set:
            cand.add(u)
            cand.add(v)
            if len(cand) >= max_candidates:
                break
    cand = sorted(cand)
    return torch.tensor(cand[:max_candidates], device=target_nodes.device, dtype=torch.long)


def candidate_edge_tuples_from_targets(edge_index: torch.Tensor, target_nodes: torch.Tensor, max_candidates: int) -> List[Tuple[int, int]]:
    target_set = set(target_nodes.detach().cpu().tolist())
    row, col = edge_index.detach().cpu()
    cands: List[Tuple[int, int]] = []
    seen = set()
    for u, v in zip(row.tolist(), col.tolist()):
        if u in target_set or v in target_set:
            t = (int(u), int(v))
            if t not in seen:
                cands.append(t)
                seen.add(t)
            if len(cands) >= max_candidates:
                break
    if not cands:
        for u, v in zip(row.tolist(), col.tolist()):
            t = (int(u), int(v))
            if t not in seen:
                cands.append(t)
                seen.add(t)
            if len(cands) >= max_candidates:
                break
    return cands


def edge_tuple_to_indices(edge_index: torch.Tensor) -> Dict[Tuple[int, int], List[int]]:
    row, col = edge_index.detach().cpu()
    mapping: Dict[Tuple[int, int], List[int]] = {}
    for idx, (u, v) in enumerate(zip(row.tolist(), col.tolist())):
        t = (int(u), int(v))
        mapping.setdefault(t, []).append(idx)
    return mapping


def topk_from_edge_tuple_candidates(
    edge_scores: torch.Tensor, edge_index: torch.Tensor, candidate_tuples: Sequence[Tuple[int, int]], k: int
) -> List[Tuple[int, int]]:
    tuple2idx = edge_tuple_to_indices(edge_index)
    scored: List[Tuple[float, Tuple[int, int]]] = []
    for t in candidate_tuples:
        idxs = tuple2idx.get(t, [])
        if len(idxs) == 0:
            score = 0.0
        else:
            idx_tensor = torch.tensor(idxs, device=edge_scores.device, dtype=torch.long)
            score = float(edge_scores[idx_tensor].mean().detach().cpu().item())
        scored.append((score, t))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [t for _, t in scored[: max(1, min(int(k), len(scored)))]]


def _target_objective(
    model: DynamicGAE,
    x: torch.Tensor,
    edge_index: torch.Tensor,
    emb_1: torch.Tensor,
    emb_2: torch.Tensor,
    tau: float,
    target_nodes: torch.Tensor,
) -> torch.Tensor:
    scores = anomaly_scores_from_model(model, x, edge_index, emb_1, emb_2, tau)
    return scores[target_nodes].mean()


def explain_exdig(
    model: DynamicGAE,
    x: torch.Tensor,
    edge_index: torch.Tensor,
    emb_1: torch.Tensor,
    emb_2: torch.Tensor,
    tau: float,
    target_nodes: torch.Tensor,
    max_node_candidates: int,
    max_edge_candidates: int,
    local_linear_samples: int,
    seed: int,
) -> Dict[str, object]:
    device = x.device
    base_obj = _target_objective(model, x, edge_index, emb_1, emb_2, tau, target_nodes).detach()

    cand_nodes = candidate_nodes_from_targets(edge_index, target_nodes, max_node_candidates)
    node_scores = torch.zeros(x.size(0), device=device)
    for nid in cand_nodes.tolist():
        x_pert = x.clone()
        x_pert[nid, :] = 0.0
        obj = _target_objective(model, x_pert, edge_index, emb_1, emb_2, tau, target_nodes).detach()
        node_scores[nid] = (base_obj - obj).abs()

    candidate_edge_tuples = candidate_edge_tuples_from_targets(edge_index, target_nodes, max_edge_candidates)
    tuple2idx = edge_tuple_to_indices(edge_index)
    edge_scores = torch.zeros(edge_index.size(1), device=device)
    for t in candidate_edge_tuples:
        idxs = tuple2idx.get(t, [])
        if len(idxs) == 0:
            continue
        keep_mask = torch.ones(edge_index.size(1), dtype=torch.bool, device=device)
        idx_tensor = torch.tensor(idxs, device=device, dtype=torch.long)
        keep_mask[idx_tensor] = False
        edge_pert = edge_index[:, keep_mask]
        obj = _target_objective(model, x, edge_pert, emb_1, emb_2, tau, target_nodes).detach()
        edge_scores[idx_tensor] = (base_obj - obj).abs()

    rng = random.Random(seed)
    d = x.size(1)
    masks = []
    ys = []
    for _ in range(max(8, int(local_linear_samples))):
        keep_dim = rng.randint(max(1, d // 2), d)
        m = torch.zeros(d, device=device)
        m[:keep_dim] = 1.0
        x_pert = x.clone()
        x_pert[target_nodes] = x_pert[target_nodes] * m.unsqueeze(0)
        obj = _target_objective(model, x_pert, edge_index, emb_1, emb_2, tau, target_nodes).detach()
        masks.append(m)
        ys.append(obj)
    M = torch.stack(masks, dim=0)
    y = torch.stack(ys, dim=0).unsqueeze(1)
    X_aug = torch.cat([M, torch.ones(M.size(0), 1, device=device)], dim=1)
    beta = torch.linalg.lstsq(X_aug, y).solution.squeeze(1)
    feature_scores = beta[:-1].abs()

    return {
        "node_scores": node_scores,
        "edge_scores": edge_scores,
        "feature_scores": feature_scores,
        "edge_candidate_tuples": candidate_edge_tuples,
    }


def spearman_rho_np(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    mask = np.isfinite(a) & np.isfinite(b)
    a, b = a[mask], b[mask]
    if a.size < 3:
        return float("nan")
    ra = np.argsort(np.argsort(a)).astype(np.float64)
    rb = np.argsort(np.argsort(b)).astype(np.float64)
    c = np.corrcoef(ra, rb)[0, 1]
    return float(c) if np.isfinite(c) else float("nan")


def relative_drop(base: float, new: float, eps: float = 1e-8) -> float:
    return float((base - new) / (abs(base) + eps))


def faithfulness_spearman_node(
    model: DynamicGAE,
    x: torch.Tensor,
    edge_index: torch.Tensor,
    emb_1: torch.Tensor,
    emb_2: torch.Tensor,
    tau: float,
    target_nodes: torch.Tensor,
    node_scores: torch.Tensor,
    node_candidates: torch.Tensor,
    n_sample: int,
    sub_seed: int,
) -> float:
    rng = random.Random(sub_seed)
    cands = node_candidates.detach().cpu().tolist()
    if len(cands) < 3:
        return float("nan")
    rng.shuffle(cands)
    picked = cands[: min(int(n_sample), len(cands))]
    base = float(_target_objective(model, x, edge_index, emb_1, emb_2, tau, target_nodes).detach().item())
    attrs: List[float] = []
    effects: List[float] = []
    for nid in picked:
        x_ = x.clone()
        x_[nid, :] = 0.0
        newv = float(_target_objective(model, x_, edge_index, emb_1, emb_2, tau, target_nodes).detach().item())
        attrs.append(float(node_scores[nid].detach().item()))
        effects.append(abs(base - newv))
    return spearman_rho_np(np.array(attrs), np.array(effects))


def faithfulness_spearman_edge(
    model: DynamicGAE,
    x: torch.Tensor,
    edge_index: torch.Tensor,
    emb_1: torch.Tensor,
    emb_2: torch.Tensor,
    tau: float,
    target_nodes: torch.Tensor,
    edge_scores: torch.Tensor,
    candidate_edge_tuples: Sequence[Tuple[int, int]],
    n_sample: int,
    sub_seed: int,
) -> float:
    rng = random.Random(sub_seed)
    tuples = list(candidate_edge_tuples)
    if len(tuples) < 3:
        return float("nan")
    rng.shuffle(tuples)
    picked = tuples[: min(int(n_sample), len(tuples))]
    tuple2idx = edge_tuple_to_indices(edge_index)
    base = float(_target_objective(model, x, edge_index, emb_1, emb_2, tau, target_nodes).detach().item())
    attrs: List[float] = []
    effects: List[float] = []
    for t in picked:
        idxs = tuple2idx.get(t, [])
        if not idxs:
            continue
        keep_mask = torch.ones(edge_index.size(1), dtype=torch.bool, device=edge_index.device)
        idx_tensor = torch.tensor(idxs, device=edge_index.device, dtype=torch.long)
        keep_mask[idx_tensor] = False
        edge_pert = edge_index[:, keep_mask]
        newv = float(_target_objective(model, x, edge_pert, emb_1, emb_2, tau, target_nodes).detach().item())
        idx_tensor2 = torch.tensor(idxs, device=edge_scores.device, dtype=torch.long)
        attrs.append(float(edge_scores[idx_tensor2].mean().detach().item()))
        effects.append(abs(base - newv))
    if len(attrs) < 3:
        return float("nan")
    return spearman_rho_np(np.array(attrs), np.array(effects))


def faithfulness_spearman_feat(
    model: DynamicGAE,
    x: torch.Tensor,
    edge_index: torch.Tensor,
    emb_1: torch.Tensor,
    emb_2: torch.Tensor,
    tau: float,
    target_nodes: torch.Tensor,
    feature_scores: torch.Tensor,
    n_sample: int,
    sub_seed: int,
) -> float:
    rng = random.Random(sub_seed)
    d = x.size(1)
    if d < 3 or target_nodes.numel() == 0:
        return float("nan")
    dims = list(range(d))
    rng.shuffle(dims)
    picked = dims[: min(int(n_sample), d)]
    base = float(_target_objective(model, x, edge_index, emb_1, emb_2, tau, target_nodes).detach().item())
    attrs: List[float] = []
    effects: List[float] = []
    for kk in picked:
        x_ = x.clone()
        x_[target_nodes, kk] = 0.0
        newv = float(_target_objective(model, x_, edge_index, emb_1, emb_2, tau, target_nodes).detach().item())
        attrs.append(float(feature_scores[kk].detach().item()))
        effects.append(abs(base - newv))
    return spearman_rho_np(np.array(attrs), np.array(effects))


def evaluate_exdig(
    model: DynamicGAE,
    data,
    emb_1: torch.Tensor,
    emb_2: torch.Tensor,
    tau: float,
    target_nodes: torch.Tensor,
    scores: Dict[str, object],
    topk_node: int,
    topk_edge: int,
    topk_feat: int,
    max_node_candidates: int,
    max_edge_candidates: int,
    dig_ratio_pair: Tuple[float, float],
    stability_node_p_pair: Tuple[float, float],
    stability_edge_p_pair: Tuple[float, float],
    stability_seed_pair: Tuple[int, int],
    local_linear_samples: int,
) -> Dict[str, float]:
    node_candidates = candidate_nodes_from_targets(data.edge_index, target_nodes, max_node_candidates)
    candidate_edge_tuples = candidate_edge_tuples_from_targets(data.edge_index, target_nodes, max_edge_candidates)

    base = float(_target_objective(model, data.x, data.edge_index, emb_1, emb_2, tau, target_nodes).detach().item())

    node_scores = scores["node_scores"]
    feat_scores = scores["feature_scores"]
    edge_scores = scores["edge_scores"]

    top_nodes = node_candidates[torch.topk(node_scores[node_candidates], k=min(topk_node, node_candidates.numel())).indices] if node_candidates.numel() else torch.empty(0, dtype=torch.long, device=data.x.device)
    x_top = data.x.clone()
    if top_nodes.numel() > 0:
        x_top[top_nodes, :] = 0.0
    new_top = float(_target_objective(model, x_top, data.edge_index, emb_1, emb_2, tau, target_nodes).detach().item())
    rel_drop_node_top = relative_drop(base, new_top)

    top_edge_tuples = topk_from_edge_tuple_candidates(edge_scores, data.edge_index, candidate_edge_tuples, topk_edge)
    tuple2idx = edge_tuple_to_indices(data.edge_index)
    keep_mask = torch.ones(data.edge_index.size(1), dtype=torch.bool, device=data.edge_index.device)
    for t in top_edge_tuples:
        idxs = tuple2idx.get(t, [])
        if not idxs:
            continue
        keep_mask[torch.tensor(idxs, device=data.edge_index.device, dtype=torch.long)] = False
    edge_pert = data.edge_index[:, keep_mask]
    new_edge_top = float(_target_objective(model, data.x, edge_pert, emb_1, emb_2, tau, target_nodes).detach().item())
    rel_drop_edge_top = relative_drop(base, new_edge_top)

    top_feat = torch.topk(feat_scores, k=min(topk_feat, feat_scores.numel())).indices
    x_feat = data.x.clone()
    if target_nodes.numel() > 0 and top_feat.numel() > 0:
        x_feat[target_nodes.unsqueeze(1), top_feat.unsqueeze(0)] = 0.0
    new_feat_top = float(_target_objective(model, x_feat, data.edge_index, emb_1, emb_2, tau, target_nodes).detach().item())
    rel_drop_feat_top = relative_drop(base, new_feat_top)

    x1, e1 = make_eval_view_with_dig_protocol(
        data.x, data.edge_index,
        feature_keep_ratio=dig_ratio_pair[0],
        node_p=stability_node_p_pair[0],
        edge_p=stability_edge_p_pair[0],
        seed=stability_seed_pair[0],
    )
    x2, e2 = make_eval_view_with_dig_protocol(
        data.x, data.edge_index,
        feature_keep_ratio=dig_ratio_pair[1],
        node_p=stability_node_p_pair[1],
        edge_p=stability_edge_p_pair[1],
        seed=stability_seed_pair[1],
    )
    s1 = explain_exdig(model, x1, e1, emb_1, emb_2, tau, target_nodes,
                       max_node_candidates=max_node_candidates, max_edge_candidates=max_edge_candidates,
                       local_linear_samples=local_linear_samples, seed=stability_seed_pair[0])
    s2 = explain_exdig(model, x2, e2, emb_1, emb_2, tau, target_nodes,
                       max_node_candidates=max_node_candidates, max_edge_candidates=max_edge_candidates,
                       local_linear_samples=local_linear_samples, seed=stability_seed_pair[1])

    def _jaccard(a: Sequence, b: Sequence) -> float:
        sa, sb = set(a), set(b)
        u = len(sa | sb)
        return 1.0 if u == 0 else len(sa & sb) / u

    node_top1 = torch.topk(s1["node_scores"][node_candidates], k=min(topk_node, node_candidates.numel())).indices.detach().cpu().tolist() if node_candidates.numel() else []
    node_top2 = torch.topk(s2["node_scores"][node_candidates], k=min(topk_node, node_candidates.numel())).indices.detach().cpu().tolist() if node_candidates.numel() else []
    jaccard_node = _jaccard(node_top1, node_top2)

    edge_top1 = topk_from_edge_tuple_candidates(s1["edge_scores"], e1, candidate_edge_tuples, topk_edge)
    edge_top2 = topk_from_edge_tuple_candidates(s2["edge_scores"], e2, candidate_edge_tuples, topk_edge)
    jaccard_edge = _jaccard(edge_top1, edge_top2)

    feat_top1 = torch.topk(s1["feature_scores"], k=min(topk_feat, s1["feature_scores"].numel())).indices.detach().cpu().tolist()
    feat_top2 = torch.topk(s2["feature_scores"], k=min(topk_feat, s2["feature_scores"].numel())).indices.detach().cpu().tolist()
    jaccard_feat = _jaccard(feat_top1, feat_top2)

    sub_base = int(stability_seed_pair[0]) + 9000 + 12345
    faith_node = faithfulness_spearman_node(model, data.x, data.edge_index, emb_1, emb_2, tau, target_nodes, node_scores, node_candidates, n_sample=24, sub_seed=sub_base + 1)
    faith_edge = faithfulness_spearman_edge(model, data.x, data.edge_index, emb_1, emb_2, tau, target_nodes, edge_scores, candidate_edge_tuples, n_sample=24, sub_seed=sub_base + 2)
    faith_feat = faithfulness_spearman_feat(model, data.x, data.edge_index, emb_1, emb_2, tau, target_nodes, feat_scores, n_sample=32, sub_seed=sub_base + 3)

    return {
        "rel_drop_node_top": float(rel_drop_node_top),
        "rel_drop_edge_top": float(rel_drop_edge_top),
        "rel_drop_feat_top": float(rel_drop_feat_top),
        "jaccard_node": float(jaccard_node),
        "jaccard_edge": float(jaccard_edge),
        "jaccard_feat": float(jaccard_feat),
        "faith_spearman_node": float(faith_node),
        "faith_spearman_edge": float(faith_edge),
        "faith_spearman_feat": float(faith_feat),
    }


def _write_json(path: str, obj: Dict) -> None:
    d = os.path.dirname(path)
    if d:
        ensure_dir(d)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", type=str, default=None)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--epoch", type=int, default=30)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--lrdec_1", type=float, default=0.8)
    p.add_argument("--lrdec_2", type=int, default=200)
    p.add_argument("--dropout", type=float, default=0.6)
    p.add_argument("--tau", type=float, default=0.7)
    p.add_argument("--hidden_num", type=int, default=64)
    p.add_argument("--head", type=int, default=4)
    p.add_argument("--out_channels", type=int, default=128)
    p.add_argument("--ratio", type=float, default=0.3)
    p.add_argument("--embedding_dim", type=int, default=32)
    p.add_argument("--node_p", type=float, default=0.7)
    p.add_argument("--edge_p", type=float, default=0.7)
    p.add_argument("--top_targets", type=int, default=16)
    p.add_argument("--topk_node", type=int, default=10)
    p.add_argument("--topk_edge", type=int, default=10)
    p.add_argument("--topk_feat", type=int, default=10)
    p.add_argument("--max_node_candidates", type=int, default=128)
    p.add_argument("--max_edge_candidates", type=int, default=256)
    p.add_argument("--local_linear_samples", type=int, default=32)
    p.add_argument("--dig_ratio_low", type=float, default=0.1)
    p.add_argument("--dig_ratio_high", type=float, default=0.5)
    p.add_argument("--stability_node_p_low", type=float, default=0.3)
    p.add_argument("--stability_node_p_high", type=float, default=0.7)
    p.add_argument("--stability_edge_p_low", type=float, default=0.3)
    p.add_argument("--stability_edge_p_high", type=float, default=0.7)
    p.add_argument("--out_dir", type=str, default=None, help="If set, write one summary.json here.")
    args = p.parse_args()

    cfg = TrainConfig(
        seed=args.seed,
        device=args.device,
        epoch=args.epoch,
        lr=args.lr,
        lrdec_1=args.lrdec_1,
        lrdec_2=args.lrdec_2,
        dropout=args.dropout,
        tau=args.tau,
        hidden_num=args.hidden_num,
        head=args.head,
        out_channels=args.out_channels,
        ratio=args.ratio,
        embedding_dim=args.embedding_dim,
        node_p=args.node_p,
        edge_p=args.edge_p,
    )
    t0 = time.time()
    artifacts = train_backbone(cfg, data_root=args.data_root)
    model = artifacts["model"]
    data = artifacts["data"]
    emb_1 = artifacts["emb_1"]
    emb_2 = artifacts["emb_2"]
    tau = cfg.tau

    scores = anomaly_scores_from_model(model, data.x, data.edge_index, emb_1, emb_2, tau)
    target_nodes = select_target_nodes(scores, args.top_targets)
    ex_scores = explain_exdig(
        model=model,
        x=data.x,
        edge_index=data.edge_index,
        emb_1=emb_1,
        emb_2=emb_2,
        tau=tau,
        target_nodes=target_nodes,
        max_node_candidates=args.max_node_candidates,
        max_edge_candidates=args.max_edge_candidates,
        local_linear_samples=args.local_linear_samples,
        seed=args.seed,
    )
    metrics = evaluate_exdig(
        model=model,
        data=data,
        emb_1=emb_1,
        emb_2=emb_2,
        tau=tau,
        target_nodes=target_nodes,
        scores=ex_scores,
        topk_node=args.topk_node,
        topk_edge=args.topk_edge,
        topk_feat=args.topk_feat,
        max_node_candidates=args.max_node_candidates,
        max_edge_candidates=args.max_edge_candidates,
        dig_ratio_pair=(args.dig_ratio_low, args.dig_ratio_high),
        stability_node_p_pair=(args.stability_node_p_low, args.stability_node_p_high),
        stability_edge_p_pair=(args.stability_edge_p_low, args.stability_edge_p_high),
        stability_seed_pair=(args.seed + 101, args.seed + 202),
        local_linear_samples=args.local_linear_samples,
    )

    node_scores = ex_scores["node_scores"]
    edge_scores = ex_scores["edge_scores"]
    feature_scores = ex_scores["feature_scores"]
    top_nodes = torch.topk(node_scores, k=min(args.topk_node, node_scores.numel())).indices.detach().cpu().tolist()
    top_edges = torch.topk(edge_scores, k=min(args.topk_edge, edge_scores.numel())).indices.detach().cpu().tolist()
    top_feats = torch.topk(feature_scores, k=min(args.topk_feat, feature_scores.numel())).indices.detach().cpu().tolist()

    summary = {
        "seed": args.seed,
        "epoch": args.epoch,
        "train_seconds": time.time() - t0,
        "metrics": metrics,
        "targets": target_nodes.detach().cpu().tolist(),
        "topk": {"node": top_nodes, "edge_index": top_edges, "feat_dim": top_feats},
    }
    if args.out_dir:
        ensure_dir(args.out_dir)
        out_path = os.path.join(args.out_dir, "summary.json")
        _write_json(out_path, summary)
    else:
        print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()

