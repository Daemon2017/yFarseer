import json
import math
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

    def prepare_topology(self, active_haplogroups, topology_data):
        nodes = topology_data['allNodes']
        name_to_node = {node['name']: node for node in nodes.values()}
        self.synonym_to_snp = {
            f"{node['root']}-{synonym['variant']}": node['name']
            for node in nodes.values()
            for synonym in node['variants']
        }
        self.snp_to_ancestors = {}

        def get_ancestors(node_name):
            if node_name in self.snp_to_ancestors:
                return self.snp_to_ancestors[node_name]
            node = name_to_node.get(node_name)
            if not node:
                return [node_name]
            p_id = node.get('parentId')
            p_node = nodes.get(str(p_id)) if p_id else None
            if p_node:
                path = get_ancestors(p_node['name']) + [node_name]
            else:
                path = [node_name]
            self.snp_to_ancestors[node_name] = path
            return path

        for node in nodes.values():
            get_ancestors(node['name'])
        unique_active_snps = set()
        for h in active_haplogroups:
            canonical = self.synonym_to_snp.get(h, h)
            unique_active_snps.update(get_ancestors(canonical))
        self.all_snps = sorted(list(unique_active_snps))
        snp_to_idx = {snp: idx for idx, snp in enumerate(self.all_snps)}
        self.parent_indices = [-1] * len(self.all_snps)
        for idx, snp in enumerate(self.all_snps):
            target_node = name_to_node.get(snp)
            if target_node:
                p_id = target_node.get('parentId')
                p_node = nodes.get(str(p_id)) if p_id else None
                if p_node and p_node['name'] in snp_to_idx:
                    self.parent_indices[idx] = snp_to_idx[p_node['name']]
        os.makedirs(config.MODEL_DIR, exist_ok=True)
        with open(os.path.join(config.MODEL_DIR, 'snp_list.json'), 'w', encoding='utf-8') as f:
            json.dump(self.all_snps, f, ensure_ascii=False)
        with open(os.path.join(config.MODEL_DIR, 'parent_indices.json'), 'w', encoding='utf-8') as f:
            json.dump(self.parent_indices, f, ensure_ascii=False)

    def generate_labels_and_masks(self, haplogroups):
        num_samples = len(haplogroups)
        num_snps = len(self.all_snps)
        labels = np.zeros((num_samples, num_snps), dtype=np.uint8)
        loss_masks = np.ones((num_samples, num_snps), dtype=np.uint8)
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
                    labels[i, idx] = 1
                    if snp == canonical:
                        terminal_idx = idx
            if terminal_idx != -1:
                queue = list(children_map[terminal_idx])
                while queue:
                    curr_idx = queue.pop(0)
                    loss_masks[i, curr_idx] = 0
                    queue.extend(children_map[curr_idx])
        return labels, loss_masks


class GeneticDataset(Dataset):
    def __init__(self, features, masks, labels, loss_masks, is_training=True, all_snps=None, snp_to_tmrca=None,
                 parent_indices=None):
        self.base_features = features
        self.masks = masks
        self.labels = labels
        self.loss_masks = loss_masks
        self.is_training = is_training
        self.num_features = features.shape[1]
        self.all_snps = all_snps
        self.snp_to_tmrca = snp_to_tmrca
        self.parent_indices = parent_indices
        self.assigned_lengths = np.zeros(len(features), dtype=np.int32)
        if self.is_training:
            self.update_epoch_augmentation()
        self.idx_389i = config.EXTENDED_STR_COLS.index('DYS389i') if 'DYS389i' in config.EXTENDED_STR_COLS else None
        self.idx_389ii = config.EXTENDED_STR_COLS.index('DYS389ii') if 'DYS389ii' in config.EXTENDED_STR_COLS else None
        base_rates = [config.STR_MUTATION_RATES.get(col, 0.002) for col in config.EXTENDED_STR_COLS]
        if self.idx_389i is not None and self.idx_389ii is not None:
            rate_i = config.STR_MUTATION_RATES.get('DYS389i', 0.00186)
            rate_ii = config.STR_MUTATION_RATES.get('DYS389ii', 0.00242)
            base_rates[self.idx_389ii] = max(0.0001, rate_ii - rate_i)
        self.mutation_rates_array = np.array(base_rates, dtype=np.float32)
        self.snp_levels = np.zeros(len(self.all_snps or []), dtype=np.int32)
        if self.parent_indices:
            for i in range(len(self.parent_indices)):
                path_len = 0
                curr = self.parent_indices[i]
                while curr != -1:
                    path_len += 1
                    curr = self.parent_indices[curr]
                self.snp_levels[i] = path_len
        self.snp_evolution_intervals = {}
        if self.all_snps and self.snp_to_tmrca and self.parent_indices:
            num_snps = len(self.all_snps)
            parent_to_children = {i: [] for i in range(num_snps)}
            for child_idx, parent_idx in enumerate(self.parent_indices):
                if parent_idx != -1 and parent_idx < num_snps:
                    parent_to_children[parent_idx].append(child_idx)
            for idx, snp_name in enumerate(self.all_snps):
                age = self.snp_to_tmrca.get(snp_name)
                if age is None:
                    self.snp_evolution_intervals[snp_name] = 500.0
                    continue
                age = float(age)
                children_indices = parent_to_children.get(idx, [])
                children_ages = []
                for c_idx in children_indices:
                    c_name = self.all_snps[c_idx]
                    c_age = self.snp_to_tmrca.get(c_name)
                    if c_age is not None:
                        children_ages.append(float(c_age))
                if children_ages:
                    mean_children_age = sum(children_ages) / len(children_ages)
                    interval = mean_children_age - age
                else:
                    interval = 1985.0 - age
                self.snp_evolution_intervals[snp_name] = max(0.0, interval)

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
        if self.is_training:
            feat = self.base_features[idx].copy().astype(np.float32)
            mask = self.masks[idx].copy().astype(np.float32)
        else:
            feat = self.base_features[idx].astype(np.float32)
            mask = self.masks[idx].astype(np.float32)
        cols = config.EXTENDED_STR_COLS
        chosen_length = self.assigned_lengths[idx] if self.is_training else self.num_features
        use_389_sync = (self.idx_389i is not None and self.idx_389ii is not None and
                        self.idx_389i < chosen_length and self.idx_389ii < chosen_length and
                        mask[self.idx_389i] == 1.0 and mask[self.idx_389ii] == 1.0)
        if use_389_sync:
            feat[self.idx_389ii] = feat[self.idx_389ii] - feat[self.idx_389i]
        if self.is_training:
            if chosen_length < self.num_features:
                feat[chosen_length:] = 0.0
                mask[chosen_length:] = 0.0
            if use_389_sync:
                valid_indices = np.where(
                    (mask == 1.0) & ((feat > 1.0) | (np.arange(len(feat)) == self.idx_389ii)) & (~np.isnan(feat)))[0]
            else:
                valid_indices = np.where((mask == 1.0) & (feat > 1.0) & (~np.isnan(feat)))[0]
            if len(valid_indices) > 0:
                current_labels = self.labels[idx]
                active_snp_indices = np.where(current_labels == 1.0)[0]
                tmrca_years = 500.0
                if len(active_snp_indices) > 0 and self.all_snps:
                    active_levels = self.snp_levels[active_snp_indices]
                    deepest_local_idx = active_snp_indices[np.argmax(active_levels)]
                    last_snp_name = self.all_snps[deepest_local_idx]
                    interval = self.snp_evolution_intervals.get(last_snp_name)
                    if interval is not None:
                        tmrca_years = interval
                tmrca_years = max(100.0, tmrca_years)
                time_scale = tmrca_years / 500.0
                vals, base_probs = config.MUTATION_DISTRIBUTIONS[chosen_length]
                base_expected = np.sum(np.array(vals) * np.array(base_probs))
                target_expected = base_expected * time_scale
                adapted_probs = []
                for v in vals:
                    p = (target_expected ** v) * math.exp(-target_expected) / math.factorial(v)
                    adapted_probs.append(p)
                prob_sum = sum(adapted_probs)
                adapted_probs = [p / prob_sum for p in adapted_probs] if prob_sum > 0 else base_probs
                num_mutations = int(np.random.choice(vals, p=adapted_probs))
                num_mutations = min(num_mutations, len(valid_indices))
                lvl_rates = self.mutation_rates_array[valid_indices]
                rates_sum = np.sum(lvl_rates)
                p_normalized = lvl_rates / rates_sum if rates_sum > 0 else None
                chosen_cols = np.random.choice(valid_indices, size=num_mutations, replace=False, p=p_normalized)
                for col in chosen_cols:
                    marker_rate = self.mutation_rates_array[col]
                    rate_factor = marker_rate / 0.002
                    diffusion_factor = math.sqrt(time_scale * rate_factor)
                    diffusion_factor = max(0.5, diffusion_factor)
                    base_p_continue = 0.12
                    p_continue = base_p_continue * (1.0 - math.exp(-0.5 * diffusion_factor)) / (1.0 - math.exp(-0.5))
                    p_continue = max(0.05, min(0.45, p_continue))
                    step = 1.0
                    while np.random.rand() < p_continue:
                        step += 1.0
                        if step >= 10.0:
                            break
                    direction = np.random.choice([1.0, -1.0])
                    mutation_value = step * direction
                    feat[col] += mutation_value
        if use_389_sync:
            feat[self.idx_389ii] = feat[self.idx_389i] + feat[self.idx_389ii]
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
    def __init__(self, num_str_markers, max_allele_val, embedding_dim, output_dim, parent_indices=None):
        super().__init__()
        self.num_str_markers = num_str_markers
        self.embedding_dim = embedding_dim
        self.max_allele_val = max_allele_val
        self.total_embeddings = nn.Embedding(num_embeddings=num_str_markers * max_allele_val,
                                             embedding_dim=embedding_dim, padding_idx=0)
        offsets = torch.arange(0, num_str_markers) * max_allele_val
        self.register_buffer('offsets', offsets.unsqueeze(0), persistent=False)
        rates = [config.STR_MUTATION_RATES.get(col, 0.002) for col in config.EXTENDED_STR_COLS]
        self.register_buffer('mutation_rates', torch.tensor(rates, dtype=torch.float32).unsqueeze(0), persistent=False)
        total_input_dim = num_str_markers * (embedding_dim + 4)
        self.input_layer = nn.Sequential(
            nn.Linear(total_input_dim, config.LAYER_DIM),
            nn.LayerNorm(config.LAYER_DIM),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        self.hidden_layer = nn.Sequential(
            nn.Linear(config.LAYER_DIM + total_input_dim, config.LAYER_DIM * 2),
            nn.LayerNorm(config.LAYER_DIM * 2),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        final_mlp_dim = (config.LAYER_DIM * 2) + config.LAYER_DIM + total_input_dim
        self.final_dropout = nn.Dropout(0.3)
        self.output_layer = nn.Linear(final_mlp_dim, output_dim)
        self.register_buffer('parent_tensor', torch.tensor(parent_indices, dtype=torch.long), persistent=False)
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
        batch_size = x.size(0)
        features = x[:, :self.num_str_markers].long()
        masks = x[:, self.num_str_markers:]
        features_shifted = features + self.offsets
        features_shifted = features_shifted * masks.long()
        all_embs = self.total_embeddings(features_shifted)
        x_val = (features.float() / float(self.max_allele_val)).unsqueeze(-1)
        rate_modifier = self.mutation_rates.unsqueeze(-1)
        frequency_scale = 1.0 / (rate_modifier * 100.0 + 1e-5)
        sin_1 = torch.sin(x_val * 0.5 * frequency_scale)
        cos_1 = torch.cos(x_val * 0.5 * frequency_scale)
        sin_2 = torch.sin(x_val * 2.5 * frequency_scale)
        cos_2 = torch.cos(x_val * 2.5 * frequency_scale)
        geom_signal = torch.cat([sin_1, cos_1, sin_2, cos_2], dim=-1)
        geom_signal = geom_signal * masks.unsqueeze(-1)
        emb_combined = torch.cat([all_embs, geom_signal], dim=-1)
        x_emb = emb_combined.view(batch_size, -1)
        feat1 = self.input_layer(x_emb)
        feat2_input = torch.cat([feat1, x_emb], dim=1)
        feat2 = self.hidden_layer(feat2_input)
        combined = torch.cat([feat2, feat1, x_emb], dim=1)
        combined = self.final_dropout(combined)
        logits = self.output_layer(combined)
        probs = torch.sigmoid(logits)
        hierarchical_probs = probs.clone()
        for level in range(1, self.max_level + 1):
            mask_level = (self.level_tensor == level)
            if not mask_level.any():
                continue
            child_indices = torch.where(mask_level)[0]
            parent_indices = self.parent_tensor[child_indices]
            hierarchical_probs[:, child_indices] = hierarchical_probs[:, child_indices] * \
                                                   hierarchical_probs[:, parent_indices]
        eps = 1e-7
        hierarchical_probs = torch.clamp(hierarchical_probs, min=eps, max=1.0 - eps)
        hierarchical_logits = torch.log(hierarchical_probs / (1.0 - hierarchical_probs))
        return hierarchical_logits


class MaskedBCELoss(nn.Module):
    def __init__(self, pos_weight, level_tensor):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(reduction="none", pos_weight=pos_weight)
        self.register_buffer("level_tensor", level_tensor.clone().detach())

    def forward(self, preds, targets, masks):
        loss = self.bce(preds, targets)
        depth_multipliers = 1.0 + torch.log1p(self.level_tensor.float())
        weighted_loss = loss * depth_multipliers.unsqueeze(0)
        masked_loss = weighted_loss * masks
        panel_completeness = masks.sum(dim=1, keepdim=True) / masks.size(1)
        weighted_by_panel = masked_loss * panel_completeness
        base_loss = weighted_by_panel.sum() / (masks.sum() + 1e-8)
        return base_loss
