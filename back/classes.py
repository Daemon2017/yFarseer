import json
import os

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset

import config


class HierarchyTopologyManager:
    def __init__(self):
        self.synonym_to_snp = {}
        self.snp_to_ancestors = {}
        self.all_snps = []
        self.parent_indices = []
        self.sibling_matrix = []

    def load_topology(self, active_haplogroups):
        with open(config.TOPOLOGY_FILE, 'r', encoding='utf-8') as f:
            topology_data = json.load(f)
        nodes = topology_data['allNodes']
        self.synonym_to_snp = {
            f"{node['root']}-{synonym['variant']}": node['name']
            for node in nodes.values()
            for synonym in node['variants']
        }
        for node_id, node in nodes.items():
            path = []
            curr = node
            while curr:
                path.append(curr['name'])
                parent_id = curr.get('parentId')
                curr = nodes.get(str(parent_id)) if parent_id else None
            path.reverse()
            self.snp_to_ancestors[node['name']] = path
        unique_active_snps = set()
        for h in active_haplogroups:
            canonical = self.synonym_to_snp.get(h, h)
            ancestors = self.snp_to_ancestors.get(canonical, [canonical])
            unique_active_snps.update(ancestors)
        self.all_snps = sorted(list(unique_active_snps))
        snp_to_idx = {snp: idx for idx, snp in enumerate(self.all_snps)}
        self.parent_indices = [-1] * len(self.all_snps)
        for snp in self.all_snps:
            target_node = next((n for n in nodes.values() if n['name'] == snp), None)
            if target_node:
                p_id = target_node.get('parentId')
                p_node = nodes.get(str(p_id)) if p_id else None
                if p_node and p_node['name'] in snp_to_idx:
                    self.parent_indices[snp_to_idx[snp]] = snp_to_idx[p_node['name']]
        parent_to_children = {}
        for child_idx, parent_idx in enumerate(self.parent_indices):
            if parent_idx != -1:
                if parent_idx not in parent_to_children:
                    parent_to_children[parent_idx] = []
                parent_to_children[parent_idx].append(child_idx)
        sibling_groups = [children for children in parent_to_children.values() if len(children) > 1]
        num_snps = len(self.all_snps)
        num_groups = len(sibling_groups)
        if num_groups > 0:
            sib_matrix = np.zeros((num_groups, num_snps), dtype=np.float32)
            for g_idx, group in enumerate(sibling_groups):
                for snp_idx in group:
                    sib_matrix[g_idx, snp_idx] = 1.0
            self.sibling_matrix = sib_matrix
        else:
            self.sibling_matrix = np.zeros((0, num_snps), dtype=np.float32)
        os.makedirs(config.MODEL_DIR, exist_ok=True)
        with open(os.path.join(config.MODEL_DIR, 'snp_list.json'), 'w', encoding='utf-8') as f:
            json.dump(self.all_snps, f, ensure_ascii=False)
        with open(os.path.join(config.MODEL_DIR, 'parent_indices.json'), 'w', encoding='utf-8') as f:
            json.dump(self.parent_indices, f, ensure_ascii=False)

    def generate_labels_and_masks(self, haplogroups):
        num_samples = len(haplogroups)
        num_snps = len(self.all_snps)
        labels = np.zeros((num_samples, num_snps), dtype=np.float32)
        loss_masks = np.ones((num_samples, num_snps), dtype=np.float32)
        snp_to_idx = {snp: idx for idx, snp in enumerate(self.all_snps)}
        children_map = {i: [] for i in range(-1, num_snps)}
        for child_idx, parent_idx in enumerate(self.parent_indices):
            children_map[parent_idx].append(child_idx)
        for i, h in enumerate(haplogroups):
            canonical = self.synonym_to_snp.get(h, h)
            ancestors = self.snp_to_ancestors.get(canonical, [canonical])
            terminal_idx = -1
            for snp in ancestors:
                if snp in snp_to_idx:
                    idx = snp_to_idx[snp]
                    labels[i, idx] = 1.0
                    if snp == canonical:
                        terminal_idx = idx
            if terminal_idx != -1:
                queue = list(children_map[terminal_idx])
                while queue:
                    curr_idx = queue.pop(0)
                    loss_masks[i, curr_idx] = 0.0
                    queue.extend(children_map[curr_idx])
        return labels, loss_masks


class GeneticDataset(Dataset):
    def __init__(self, features, masks, labels, loss_masks, is_training=True):
        self.base_features = features
        self.masks = masks
        self.labels = labels
        self.loss_masks = loss_masks
        self.is_training = is_training
        self.num_features = features.shape[1]
        if self.is_training:
            self.assigned_lengths = np.zeros(len(features), dtype=np.int32)
            self.update_epoch_augmentation()
        self.mutation_rates_array = np.array(
            [config.STR_MUTATION_RATES.get(col, 0.002) for col in config.EXTENDED_STR_COLS], dtype=np.float32)

    def update_epoch_augmentation(self):
        num_samples = len(self.base_features)
        shuffled_indices = np.random.permutation(num_samples)
        splits = np.array_split(shuffled_indices, 5)
        lengths = [12, 25, 37, 67, 111]
        for split, length in zip(splits, lengths):
            self.assigned_lengths[split] = length

    def __len__(self):
        return len(self.base_features)

    def __getitem__(self, idx):
        feat = self.base_features[idx].copy()
        mask = self.masks[idx].copy()
        if self.is_training:
            chosen_length = self.assigned_lengths[idx]
            if chosen_length < self.num_features:
                feat[chosen_length:] = 0.0
                mask[chosen_length:] = 0.0
            valid_indices = np.where((mask == 1.0) & (feat != 0.0) & (~np.isnan(feat)))[0]
            if len(valid_indices) > 0:
                mutation_mapping = {12: 1, 25: 1, 37: 1, 67: 2, 111: 3}
                num_mutations = mutation_mapping.get(chosen_length, 1)
                num_mutations = min(num_mutations, len(valid_indices))
                lvl_rates = self.mutation_rates_array[valid_indices]
                rates_sum = np.sum(lvl_rates)
                if rates_sum > 0:
                    p_normalized = lvl_rates / rates_sum
                else:
                    p_normalized = None
                chosen_cols = np.random.choice(valid_indices, size=num_mutations, replace=False, p=p_normalized)
                for col in chosen_cols:
                    step = np.random.choice([1.0, 2.0, 3.0], p=[0.88, 0.10, 0.02])
                    direction = np.random.choice([1.0, -1.0])
                    feat[col] += (step * direction)
                cols = config.EXTENDED_STR_COLS
                for base_col, expected_len in config.MULTICOPIES.items():
                    suffixes = ['a', 'b', 'c', 'd'][:expected_len]
                    sub_cols = [f"{base_col}{suf}" for suf in suffixes]
                    try:
                        start_idx = cols.index(sub_cols[0])
                        end_idx = start_idx + expected_len
                        if start_idx < chosen_length:
                            actual_end = min(end_idx, chosen_length)
                            feat[start_idx:actual_end] = np.sort(feat[start_idx:actual_end])
                    except ValueError:
                        continue
        feat = np.clip(feat, a_min=0.0, a_max=float(config.MAX_ALLELE - 1))
        inputs = np.hstack([feat, mask])
        return (torch.tensor(inputs, dtype=torch.float32),
                torch.tensor(self.labels[idx], dtype=torch.float32),
                torch.tensor(self.loss_masks[idx], dtype=torch.float32))


class GeneticEmbeddingMLP(nn.Module):
    def __init__(self, num_str_markers, max_allele_val, embedding_dim, output_dim, parent_indices=None,
                 sibling_matrix=None):
        super().__init__()
        self.num_str_markers = num_str_markers
        self.embedding_dim = embedding_dim
        self.max_allele_val = max_allele_val
        self.embeddings = nn.ModuleList(
            [nn.Embedding(num_embeddings=max_allele_val, embedding_dim=embedding_dim, padding_idx=0) for _ in
             range(num_str_markers)])
        total_input_dim = num_str_markers * (embedding_dim + 4)
        self.input_layer = nn.Sequential(
            nn.Linear(total_input_dim, config.LAYER_DIM),
            nn.BatchNorm1d(config.LAYER_DIM),
            nn.ReLU()
        )
        self.hidden_layer = nn.Sequential(
            nn.Linear(config.LAYER_DIM + total_input_dim, config.LAYER_DIM * 2),
            nn.BatchNorm1d(config.LAYER_DIM * 2),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        final_mlp_dim = (config.LAYER_DIM * 2) + config.LAYER_DIM + total_input_dim
        self.output_layer = nn.Linear(final_mlp_dim, output_dim)
        self.register_buffer('parent_tensor', torch.tensor(parent_indices, dtype=torch.long), persistent=False)
        if sibling_matrix is not None and len(sibling_matrix) > 0:
            self.register_buffer('sibling_tensor', torch.tensor(sibling_matrix, dtype=torch.float32), persistent=False)
        else:
            self.register_buffer('sibling_tensor', torch.empty(0), persistent=False)
        levels = [-1] * len(parent_indices)
        for i in range(len(parent_indices)):
            path_len = 0
            curr = parent_indices[i]
            while curr != -1:
                path_len += 1
                curr = parent_indices[curr]
            levels[i] = path_len
        self.register_buffer('level_tensor', torch.tensor(levels, dtype=torch.long), persistent=False)
        self.max_level = max(levels) if len(levels) > 0 else 0

    def forward(self, x):
        features = x[:, :self.num_str_markers].long()
        masks = x[:, self.num_str_markers:]
        embedded_list = []
        for i in range(self.num_str_markers):
            emb = self.embeddings[i](features[:, i])
            x_val = (features[:, i].float() / float(self.max_allele_val)).unsqueeze(1)
            sin_1 = torch.sin(x_val * 1.0)
            cos_1 = torch.cos(x_val * 1.0)
            sin_2 = torch.sin(x_val * 10.0)
            cos_2 = torch.cos(x_val * 10.0)
            geom_signal = torch.cat([sin_1, cos_1, sin_2, cos_2], dim=1)
            emb_combined = torch.cat([emb, geom_signal], dim=1)
            emb_combined = emb_combined * masks[:, i].unsqueeze(1)
            embedded_list.append(emb_combined)
        x_emb = torch.cat(embedded_list, dim=1)
        feat1 = self.input_layer(x_emb)
        feat2_input = torch.cat([feat1, x_emb], dim=1)
        feat2 = self.hidden_layer(feat2_input)
        combined = torch.cat([feat2, feat1, x_emb], dim=1)
        return self.output_layer(combined)


class MaskedBCELoss(nn.Module):
    def __init__(self, parent_indices, pos_weight, sibling_matrix=None, level_tensor=None, max_level=0):
        super().__init__()
        self.latest_sibling_loss = 0
        self.latest_hierarchy_loss = 0
        self.latest_base_loss = 0
        self.max_level = max_level
        self.bce = nn.BCEWithLogitsLoss(reduction="none", pos_weight=pos_weight)
        self.register_buffer("parent_tensor", torch.tensor(parent_indices, dtype=torch.long))
        if level_tensor is not None:
            self.register_buffer("level_tensor", level_tensor.clone().detach())
        else:
            self.register_buffer("level_tensor", torch.empty(0))
        if sibling_matrix is not None and len(sibling_matrix) > 0:
            self.register_buffer("sibling_matrix", torch.tensor(sibling_matrix, dtype=torch.float32))
            num_groups = sibling_matrix.shape[0]
            group_parents = []
            for g_idx in range(num_groups):
                first_child_idx = np.where(sibling_matrix[g_idx] == 1.0)[0][0]
                parent_idx = parent_indices[first_child_idx]
                group_parents.append(parent_idx)
            self.register_buffer("group_parents_tensor", torch.tensor(group_parents, dtype=torch.long))
        else:
            self.register_buffer("sibling_matrix", torch.empty(0))
            self.register_buffer("group_parents_tensor", torch.empty(0))

    def forward(self, preds, targets, masks):
        loss = self.bce(preds, targets)
        depth_multipliers = 1.0 + (self.max_level - self.level_tensor.float()) * 0.5
        weighted_loss = loss * depth_multipliers.unsqueeze(0)
        masked_loss = weighted_loss * masks
        base_loss = masked_loss.sum() / (masks.sum() + 1e-8)
        probs = torch.sigmoid(preds)
        mask = self.parent_tensor != -1
        valid_children = torch.where(mask)[0]
        valid_parents = self.parent_tensor[valid_children]
        parent_negative_mask = probs[:, valid_parents] < 0.5
        descendant_violation = (probs[:, valid_children] * parent_negative_mask.float())
        child_positive_mask = probs[:, valid_children] >= 0.5
        ancestor_violation = (torch.clamp(probs[:, valid_children] - probs[:, valid_parents],
                                          min=0.0) * child_positive_mask.float())
        total_h_violation = (descendant_violation + ancestor_violation) * masks[:, valid_children]
        if total_h_violation.any():
            violation_loss = total_h_violation[total_h_violation > 0].mean()
        else:
            violation_loss = torch.tensor(0.0, device=preds.device)
        sibling_loss = torch.tensor(0.0, device=preds.device)
        if self.sibling_matrix.numel() > 0 and self.sibling_matrix.size(0) > 0:
            group_sums = torch.matmul(probs, self.sibling_matrix.t())
            group_squares_sums = torch.matmul(probs ** 2, self.sibling_matrix.t())
            pairwise_products = 0.5 * (group_sums ** 2 - group_squares_sums)
            group_levels = self.level_tensor[self.group_parents_tensor]
            depth_weights = 1.0 + group_levels.float()
            weighted_pairwise = pairwise_products * depth_weights.unsqueeze(0)
            sibling_loss = weighted_pairwise.mean()
        self.latest_base_loss = base_loss.item()
        self.latest_hierarchy_loss = (config.HIERARCHY_PENALTY_WEIGHT * violation_loss).item()
        self.latest_sibling_loss = (config.SIBLING_PENALTY_WEIGHT * sibling_loss).item()
        return (base_loss + (config.HIERARCHY_PENALTY_WEIGHT * violation_loss) +
                (config.SIBLING_PENALTY_WEIGHT * sibling_loss))
