import json
import os

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, Sampler

import config
import utils


class HierarchyTopologyManager:
    def __init__(self):
        self.synonym_to_snp = {}
        self.snp_to_ancestors = {}
        self.all_snps = []
        self.parent_indices = []
        self.snp_levels = []
        self.parent_to_children_raw = {}

    def get_ancestors(self, node_name, name_to_node, nodes):
        if node_name.endswith('*'):
            base_node = node_name[:-1]
            return self.get_ancestors(base_node, name_to_node, nodes) + [node_name]
        if node_name in self.snp_to_ancestors:
            return self.snp_to_ancestors[node_name]
        node = name_to_node.get(node_name)
        if not node:
            return [node_name]
        p_id = node.get('parentId')
        p_node = nodes.get(str(p_id)) if p_id else None
        if p_node:
            path = self.get_ancestors(p_node['name'], name_to_node, nodes) + [node_name]
        else:
            path = [node_name]
        self.snp_to_ancestors[node_name] = path
        return path

    def prepare_topology(self, active_haplogroups, topology_data):
        nodes = topology_data['allNodes']
        extended_nodes = {int(k): v for k, v in nodes.items()}
        next_virtual_id = max(extended_nodes.keys()) + 1 if extended_nodes else 1
        all_parents_in_topo = set()
        for node in nodes.values():
            p_id = node.get('parentId')
            if p_id is not None:
                all_parents_in_topo.add(int(p_id))
        for p_id in all_parents_in_topo:
            parent_node = nodes[str(p_id)]
            parent_name = parent_node['name']
            virtual_name = f"{parent_name}*"
            virtual_node = {
                'id': next_virtual_id,
                'name': virtual_name,
                'parentId': p_id,
                'root': parent_node.get('root', parent_name),
                'variants': [{'variant': virtual_name}]
            }
            extended_nodes[next_virtual_id] = virtual_node
            next_virtual_id += 1
        nodes = {str(k): v for k, v in extended_nodes.items()}
        self.name_to_node = {node['name']: node for node in nodes.values()}
        self.nodes_raw = nodes
        self.synonym_to_snp = {
            f"{node['root']}-{synonym['variant']}": node['name']
            for node in nodes.values()
            for synonym in node['variants']
        }
        self.snp_to_ancestors = {}
        self.parent_to_children_raw = {}
        for node in nodes.values():
            if node['name'].endswith('*'):
                continue
            p_id = node.get('parentId')
            p_node = nodes.get(str(p_id)) if p_id else None
            if p_node:
                self.parent_to_children_raw.setdefault(p_node['name'], []).append(node['name'])
        for node in nodes.values():
            self.get_ancestors(node['name'], self.name_to_node, self.nodes_raw)
        unique_active_snps = set()
        active_set = set(self.synonym_to_snp.get(h, h) for h in active_haplogroups)
        for h in active_haplogroups:
            resolved_snp = utils.resolve_para_groups(h, self.synonym_to_snp, active_set, self.parent_to_children_raw)
            unique_active_snps.update(self.get_ancestors(resolved_snp, self.name_to_node, self.nodes_raw))
        self.all_snps = sorted(list(unique_active_snps))
        snp_to_idx = {snp: idx for idx, snp in enumerate(self.all_snps)}
        self.parent_indices = [-1] * len(self.all_snps)
        for idx, snp in enumerate(self.all_snps):
            target_node = self.name_to_node.get(snp)
            if target_node:
                p_id = target_node.get('parentId')
                p_node = self.nodes_raw.get(str(p_id)) if p_id else None
                if p_node and p_node['name'] in snp_to_idx:
                    self.parent_indices[idx] = snp_to_idx[p_node['name']]
        self.snp_levels = [0] * len(self.all_snps)
        for i in range(len(self.all_snps)):
            path_len = 0
            curr = self.parent_indices[i]
            while curr != -1:
                path_len += 1
                curr = self.parent_indices[curr]
            self.snp_levels[i] = path_len
        os.makedirs(config.MODEL_DIR, exist_ok=True)
        with open(os.path.join(config.MODEL_DIR, 'snp_list.json'), 'w', encoding='utf-8') as f:
            json.dump(self.all_snps, f, ensure_ascii=False)
        with open(os.path.join(config.MODEL_DIR, 'parent_indices.json'), 'w', encoding='utf-8') as f:
            json.dump(self.parent_indices, f, ensure_ascii=False)
        with open(os.path.join(config.MODEL_DIR, 'snp_levels.json'), 'w', encoding='utf-8') as f:
            json.dump(self.snp_levels, f, ensure_ascii=False)

    def generate_labels_and_masks(self, haplogroups):
        num_samples = len(haplogroups)
        num_snps = len(self.all_snps)
        labels = np.zeros((num_samples, num_snps), dtype=np.uint8)
        loss_masks = np.ones((num_samples, num_snps), dtype=np.uint8)
        snp_to_idx = {snp: idx for idx, snp in enumerate(self.all_snps)}
        children_map = {i: [] for i in range(-1, num_snps)}
        for child_idx, parent_idx in enumerate(self.parent_indices):
            children_map[parent_idx].append(child_idx)
        active_set = set(self.synonym_to_snp.get(h, h) for h in haplogroups)
        for i, h in enumerate(haplogroups):
            target_snp = utils.resolve_para_groups(h, self.synonym_to_snp, active_set, self.parent_to_children_raw)
            ancestors = self.get_ancestors(target_snp, self.name_to_node, self.nodes_raw)
            terminal_idx = -1
            for snp in ancestors:
                if snp in snp_to_idx:
                    idx = snp_to_idx[snp]
                    labels[i, idx] = 1
                    if snp == target_snp:
                        terminal_idx = idx
            if terminal_idx != -1:
                queue = list(children_map[terminal_idx])
                while queue:
                    curr_idx = queue.pop(0)
                    loss_masks[i, curr_idx] = 0
                    queue.extend(children_map[curr_idx])
        return labels, loss_masks


class GeneticDataset(Dataset):
    def __init__(self, features, masks, labels, loss_masks, enable_dropout=None, enable_mutations=None, all_snps=None,
                 snp_to_tmrca=None, parent_indices=None, snp_levels=None):
        self.base_features = features
        self.masks = masks
        self.labels = labels
        self.loss_masks = loss_masks
        self.enable_dropout = enable_dropout
        self.enable_mutations = enable_mutations
        self.num_features = features.shape[1]
        self.all_snps = all_snps
        self.parent_indices = parent_indices
        self.assigned_lengths = np.zeros(len(features), dtype=np.int32)
        if self.enable_dropout or self.enable_mutations:
            self.update_epoch_augmentation()
        self.idx_389i = config.EXTENDED_STR_COLS.index('DYS389i') if 'DYS389i' in config.EXTENDED_STR_COLS else None
        self.idx_389ii = config.EXTENDED_STR_COLS.index('DYS389ii') if 'DYS389ii' in config.EXTENDED_STR_COLS else None
        base_rates = [config.STR_MUTATION_RATES.get(col, 0.002) for col in config.EXTENDED_STR_COLS]
        self.mutation_rates_array = np.array(base_rates, dtype=np.float32)
        self.sample_is_terminal = np.zeros(len(features), dtype=bool)
        self.sample_tmrca_years = np.zeros(len(features), dtype=np.float32)
        if self.enable_mutations and all_snps and parent_indices and snp_levels:
            is_terminal_snp = np.ones(len(all_snps), dtype=bool)
            for p_idx in parent_indices:
                if p_idx != -1 and p_idx < len(is_terminal_snp):
                    is_terminal_snp[p_idx] = False
            snp_levels_np = np.array(snp_levels, dtype=np.int32)
            for i in range(len(features)):
                active_snp_indices = np.where(labels[i] == 1.0)[0]
                if len(active_snp_indices) > 0:
                    deepest_local_idx = active_snp_indices[np.argmax(snp_levels_np[active_snp_indices])]
                    if is_terminal_snp[deepest_local_idx]:
                        self.sample_is_terminal[i] = True
                        snp_name = all_snps[deepest_local_idx]
                        if snp_name.endswith('*'):
                            lookup_name = snp_name[:-1]
                        else:
                            lookup_name = snp_name
                        age = snp_to_tmrca.get(lookup_name, 500.0)
                        if age is None:
                            age = 500.0
                        self.sample_tmrca_years[i] = max(25.0, 1985.0 - float(age))

    def update_epoch_augmentation(self):
        num_samples = len(self.base_features)
        self.assigned_lengths = 0.1 + (np.random.rand(num_samples) * 0.9)

    def __len__(self):
        return len(self.base_features)

    def __getitem__(self, idx):
        need_mutations = self.enable_mutations and self.sample_is_terminal[idx]
        need_modify = self.enable_dropout or need_mutations
        if need_modify:
            feat = self.base_features[idx].copy().astype(np.float32)
            mask = self.masks[idx].copy().astype(np.float32)
        else:
            feat = self.base_features[idx].astype(np.float32)
            mask = self.masks[idx].astype(np.float32)
        cols = config.EXTENDED_STR_COLS
        if need_mutations:
            valid_indices = np.where((mask == 1.0) & (feat > 1.0) & (~np.isnan(feat)))[0]
            if len(valid_indices) > 0:
                tmrca_years = self.sample_tmrca_years[idx]
                generations = tmrca_years / 30.0
                lambdas = self.mutation_rates_array[valid_indices] * generations
                mutations = np.random.poisson(lambdas)
                active_mask = mutations > 0
                if np.any(active_mask):
                    mut_indices = valid_indices[active_mask]
                    mut_counts = mutations[active_mask]
                    all_mut_steps_indices = np.repeat(mut_indices, mut_counts)
                    base_pairs = len(all_mut_steps_indices) // 2
                    if base_pairs >= 2:
                        pair_delta = int(np.round(np.random.normal(loc=0.0, scale=1.0)))
                        final_pairs = max(1, base_pairs + pair_delta)
                        target_len = final_pairs * 2
                        current_len = len(all_mut_steps_indices)
                        if current_len < target_len:
                            extra_indices = np.random.choice(valid_indices, size=(target_len - current_len),
                                                             replace=True)
                            all_mut_steps_indices = np.append(all_mut_steps_indices, extra_indices)
                        elif current_len > target_len:
                            p_rates = self.mutation_rates_array[valid_indices]
                            p_norm = p_rates / np.sum(p_rates)
                            all_mut_steps_indices = np.random.choice(valid_indices, size=target_len, replace=True,
                                                                     p=p_norm)
                    else:
                        if len(all_mut_steps_indices) % 2 != 0:
                            rates = self.mutation_rates_array[valid_indices]
                            p_norm = rates / np.sum(rates)
                            extra_fake_idx = np.random.choice(valid_indices, p=p_norm)
                            all_mut_steps_indices = np.append(all_mut_steps_indices, extra_fake_idx)
                    np.random.shuffle(all_mut_steps_indices)
                    half = len(all_mut_steps_indices) // 2
                    plus_indices = all_mut_steps_indices[:half]
                    minus_indices = all_mut_steps_indices[half: half * 2]
                    np.add.at(feat, plus_indices, 1.0)
                    np.add.at(feat, minus_indices, -1.0)
                    dys389i_delta = np.sum(plus_indices == self.idx_389i) - np.sum(minus_indices == self.idx_389i)
                    if dys389i_delta != 0.0 and self.idx_389ii is not None and mask[self.idx_389ii] == 1.0:
                        feat[self.idx_389ii] += dys389i_delta
        for base_col, expected_len in config.MULTICOPIES.items():
            suffixes = ['a', 'b', 'c', 'd'][:expected_len]
            sub_cols = [f"{base_col}{suf}" for suf in suffixes]
            try:
                start_idx = cols.index(sub_cols[0])
                if start_idx < self.num_features:
                    actual_end = min(start_idx + expected_len, self.num_features)
                    feat[start_idx:actual_end] = np.sort(feat[start_idx:actual_end])
            except ValueError:
                continue
        is_orig_deletion = (self.base_features[idx] == 1)
        feat = np.where((mask == 1.0) & (~is_orig_deletion),
                        np.clip(feat, a_min=2.0, a_max=float(config.MAX_ALLELE - 1)), feat)
        if self.enable_dropout:
            keep_ratio = self.assigned_lengths[idx]
            dropout_mask = (np.random.rand(self.num_features) < keep_ratio).astype(np.float32)
            mask = mask * dropout_mask
            feat = feat * mask
        inputs = np.hstack([feat, mask])
        return (torch.tensor(inputs, dtype=torch.float32),
                torch.tensor(self.labels[idx], dtype=torch.float32),
                torch.tensor(self.loss_masks[idx], dtype=torch.float32))


class GeneticEmbeddingMLP(nn.Module):
    def __init__(self, num_str_markers, max_allele_val, embedding_dim, output_dim, snp_levels=None):
        super().__init__()
        self.num_str_markers = num_str_markers
        self.embedding_dim = embedding_dim
        self.max_allele_val = max_allele_val
        self.total_embeddings = nn.Embedding(num_embeddings=num_str_markers * max_allele_val,
                                             embedding_dim=embedding_dim, padding_idx=0)
        offsets = torch.arange(0, num_str_markers) * max_allele_val
        self.register_buffer('offsets', offsets.unsqueeze(0), persistent=False)
        total_input_dim = num_str_markers * embedding_dim
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
        self.output_layer = nn.Linear((config.LAYER_DIM * 2) + config.LAYER_DIM + total_input_dim, output_dim)
        self.register_buffer('level_tensor', torch.tensor(snp_levels or [], dtype=torch.long), persistent=False)
        self.max_level = max(snp_levels) if snp_levels and len(snp_levels) > 0 else 0

    def forward(self, x):
        batch_size = x.size(0)
        features = x[:, :self.num_str_markers].long()
        masks = x[:, self.num_str_markers:]
        features_shifted = (features + self.offsets) * masks.long()
        all_embs = self.total_embeddings(features_shifted)
        x_emb = all_embs.view(batch_size, -1)
        feat1 = self.input_layer(x_emb)
        feat2 = self.hidden_layer(torch.cat([feat1, x_emb], dim=1))
        return self.output_layer(torch.cat([feat2, feat1, x_emb], dim=1))


class MaskedBCELoss(nn.Module):
    def __init__(self, pos_weight, level_tensor, parent_indices):
        super().__init__()
        self.register_buffer("pos_weight", pos_weight.clone().detach().unsqueeze(0))
        self.register_buffer("level_tensor", level_tensor.clone().detach())
        output_dim = len(parent_indices)
        row_indices = []
        col_indices = []
        for i in range(output_dim):
            curr = parent_indices[i]
            while curr != -1:
                row_indices.append(i)
                col_indices.append(curr)
                curr = parent_indices[curr]
        indices = torch.tensor([row_indices, col_indices], dtype=torch.long)
        values = torch.ones(len(row_indices), dtype=torch.float32)
        sparse_ancestry = torch.sparse_coo_tensor(indices, values, (output_dim, output_dim))
        self.register_buffer('ancestry_matrix', sparse_ancestry, persistent=False)

    def compute_hierarchical_probs(self, logits):
        eps = 1e-7
        probs_raw = torch.sigmoid(logits)
        log_probs_raw = torch.log(probs_raw + eps)
        hierarchical_log_probs = log_probs_raw + torch.sparse.mm(self.ancestry_matrix, log_probs_raw.t()).t()
        p_h = torch.clamp(torch.exp(hierarchical_log_probs), min=eps, max=1.0 - eps)
        return p_h

    def forward(self, logits, targets, masks):
        p_h = self.compute_hierarchical_probs(logits)
        fp_weight = 1.0 + (self.level_tensor.float() / 20.0)
        fp_weight = fp_weight.unsqueeze(0)
        loss = - (self.pos_weight * targets * torch.log(p_h) + fp_weight * (1.0 - targets) * torch.log(1.0 - p_h))
        loss *= (1.0 + torch.log1p(self.level_tensor.float())).unsqueeze(0)
        loss *= masks
        panel_completeness = masks.sum(dim=1, keepdim=True) / masks.size(1)
        loss *= panel_completeness
        return loss.sum() / (masks.sum() + 1e-8)


class PairAdversarialBatchSampler(Sampler):
    def __init__(self, dataset, batch_size, df_train):
        super().__init__()
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_samples = len(dataset)
        self.top_n_convergence = 8
        feats = dataset.base_features.astype(np.float32)
        dot_product = np.dot(feats, feats.T)
        diag = np.diag(dot_product)
        str_dist = diag[:, None] + diag[None, :] - 2 * dot_product
        str_dist = np.sqrt(np.clip(str_dist, 0, None))
        np.fill_diagonal(str_dist, 1e9)
        haplogroups = df_train['Haplogroup'].values
        same_hg_mask = (haplogroups[:, None] == haplogroups[None, :])
        str_dist[same_hg_mask] = 1e9
        self.adversarial_pools = {}
        for i in range(self.num_samples):
            sorted_positions = np.argsort(str_dist[i])
            top_positions = sorted_positions[:self.top_n_convergence]
            self.adversarial_pools[i] = top_positions.tolist()

    def __iter__(self):
        indices = np.arange(self.num_samples)
        np.random.shuffle(indices)
        visited = set()
        batches = []
        current_batch = []
        for idx in indices:
            if idx in visited:
                continue
            current_batch.append(int(idx))
            visited.add(idx)
            enemy_pool = self.adversarial_pools[idx]
            enemy_idx = np.random.choice(enemy_pool)
            if enemy_idx not in visited and len(current_batch) < self.batch_size:
                current_batch.append(int(enemy_idx))
                visited.add(enemy_idx)
            if len(current_batch) >= self.batch_size:
                batches.append(current_batch)
                current_batch = []
        if current_batch:
            if len(batches) > 0 and len(current_batch) < (self.batch_size // 2):
                for b_idx, item in enumerate(current_batch):
                    batches[b_idx % len(batches)].append(item)
            else:
                batches.append(current_batch)
        np.random.shuffle(batches)
        for batch in batches:
            yield batch

    def __len__(self):
        return self.num_samples // self.batch_size
