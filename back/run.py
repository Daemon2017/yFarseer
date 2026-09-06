import json
import os
import re

import numpy as np
import torch
from flask import Flask, request, jsonify
from flask_cors import CORS
from waitress import serve

import classes
import config

app = Flask(__name__)
CORS(app)

predictor = None


def hierarchical_predict(model, x, parent_indices, level_tensor, max_level):
    model.eval()
    with torch.no_grad():
        logits = model(x)
        probs = torch.sigmoid(logits)
        batch_size = probs.size(0)
        levels_list = level_tensor.cpu().numpy().tolist()
        final_active_mask = torch.zeros_like(probs, dtype=torch.bool)
        root_indices = [idx for idx, lvl in enumerate(levels_list) if lvl == 0]
        if root_indices:
            for b in range(batch_size):
                if len(root_indices) == 1:
                    best_root_idx = root_indices[0]
                    if probs[b, best_root_idx] >= 0.5:
                        final_active_mask[b, best_root_idx] = True
                else:
                    root_probs = [probs[b, r_idx].item() for r_idx in root_indices]
                    max_idx = np.argmax(root_probs)
                    best_root_idx = root_indices[max_idx]
                    leader_prob = root_probs[max_idx]
                    if leader_prob >= 0.5:
                        sorted_probs = sorted(root_probs, reverse=True)
                        margin = sorted_probs[0] - sorted_probs[1]
                        if margin >= 0.15:
                            final_active_mask[b, best_root_idx] = True
        for lvl in range(1, max_level + 1):
            lvl_indices = [idx for idx, l in enumerate(levels_list) if l == lvl]
            if not lvl_indices:
                continue
            for b in range(batch_size):
                b_lvl_indices = []
                for idx in lvl_indices:
                    p_idx = parent_indices[idx]
                    if p_idx != -1 and final_active_mask[b, p_idx]:
                        b_lvl_indices.append(idx)
                if not b_lvl_indices:
                    continue
                parents_groups = {}
                for idx in b_lvl_indices:
                    p_idx = parent_indices[idx]
                    if p_idx not in parents_groups:
                        parents_groups[p_idx] = []
                    parents_groups[p_idx].append(idx)
                for p_idx, siblings in parents_groups.items():
                    parent_prob = probs[b, p_idx].item()
                    if len(siblings) == 1:
                        sib_idx = siblings[0]
                        max_val = probs[b, sib_idx]
                        adjusted_prob = min(max_val.item(), parent_prob)
                        probs[b, sib_idx] = adjusted_prob
                        if adjusted_prob >= 0.5:
                            final_active_mask[b, sib_idx] = True
                    else:
                        sib_probs = [probs[b, s_idx].item() for s_idx in siblings]
                        max_idx = np.argmax(sib_probs)
                        best_sib_idx = siblings[max_idx]
                        leader_prob = sib_probs[max_idx]
                        adjusted_prob = min(leader_prob, parent_prob)
                        probs[b, best_sib_idx] = adjusted_prob
                        if adjusted_prob >= 0.5:
                            sorted_probs = sorted(sib_probs, reverse=True)
                            margin = sorted_probs[0] - sorted_probs[1]
                            if margin >= 0.15:
                                final_active_mask[b, best_sib_idx] = True
        probs = torch.where(final_active_mask, probs, torch.tensor(0.0, device=probs.device))
    return probs


class GeneticSingleModel:
    def __init__(self):
        self.model = None
        self.sorted_snps = []
        self.parent_indices = []
        self.num_str_markers = len(config.EXTENDED_STR_COLS)
        self.max_allele_val = config.MAX_ALLELE
        self.embedding_dim = config.EMBEDDING_DIM

    def load_model(self):
        with open(os.path.join(config.MODEL_DIR, "snp_list.json"), "r", encoding="utf-8") as f:
            self.sorted_snps = json.load(f)
        with open(os.path.join(config.MODEL_DIR, "parent_indices.json"), "r", encoding="utf-8") as f:
            self.parent_indices = json.load(f)
        parent_to_children = {}
        for child_idx, parent_idx in enumerate(self.parent_indices):
            if parent_idx != -1:
                if parent_idx not in parent_to_children:
                    parent_to_children[parent_idx] = []
                parent_to_children[parent_idx].append(child_idx)
        sibling_groups = [children for children in parent_to_children.values() if len(children) > 1]
        num_snps = len(self.parent_indices)
        num_groups = len(sibling_groups)
        if num_groups > 0:
            sib_matrix = np.zeros((num_groups, num_snps), dtype=np.float32)
            for g_idx, group in enumerate(sibling_groups):
                for snp_idx in group:
                    sib_matrix[g_idx, snp_idx] = 1.0
            sibling_matrix = sib_matrix
        else:
            sibling_matrix = np.zeros((0, num_snps), dtype=np.float32)
        output_dim = len(self.sorted_snps)
        model_path = os.path.join(config.MODEL_DIR, "model_best_emr.pth")
        if os.path.exists(model_path):
            self.model = classes.GeneticEmbeddingMLP(
                self.num_str_markers,
                self.max_allele_val,
                self.embedding_dim,
                output_dim,
                self.parent_indices,
                sibling_matrix
            )
            self.model.load_state_dict(torch.load(model_path, map_location=config.DEVICE))
            self.model.to(config.DEVICE)
            self.model.eval()

    def predict_chains(self, features_matrix, masks_matrix):
        num_samples = features_matrix.shape[0]
        inputs = np.hstack([features_matrix, masks_matrix])
        inputs_tensor = torch.tensor(inputs, dtype=torch.float32).to(config.DEVICE)
        probs = hierarchical_predict(
            self.model,
            inputs_tensor,
            self.parent_indices,
            self.model.level_tensor,
            self.model.max_level
        ).cpu().numpy()
        results = []
        for i in range(num_samples):
            sample_chain = []
            for idx, snp_name in enumerate(self.sorted_snps):
                if probs[i, idx] >= 0.5:
                    sample_chain.append((snp_name, float(probs[i, idx])))
            results.append(sample_chain)
        return results


def build_recursive_tree(full_chain):
    if not full_chain:
        return {"name": "Y-Root", "score": 0.0, "children": []}
    current_node = None
    for name, prob in reversed(full_chain):
        node_data = {
            "name": name,
            "score": round(prob, 4),
            "children": [current_node] if current_node else []
        }
        current_node = node_data
    return current_node


def process_sample_dict(sample_dict):
    parsed_dict = {}
    for base_col in config.BASE_STR_COLS:
        val = sample_dict.get(base_col, None)
        if val is None or str(val).lower() in ['none', 'nan', 'null', '', 'unknown', '-']:
            parsed_dict[base_col] = [-1]
            continue
        found = re.findall(r'\d+(?:\.\d+)?', str(val).strip())
        if not found:
            parsed_dict[base_col] = [-1]
            continue
        alleles = sorted([int(float(x)) for x in found])
        if base_col in config.MULTICOPIES:
            expected = config.MULTICOPIES[base_col]
            if len(alleles) > expected:
                if expected == 2:
                    alleles = [alleles[0], alleles[-1]]
                elif expected == 4:
                    alleles = [alleles[0], alleles[1], alleles[-2], alleles[-1]]
            elif len(alleles) < expected:
                while len(alleles) < expected:
                    alleles.append(alleles[-1] if alleles else 0)
            parsed_dict[base_col] = alleles
        else:
            parsed_dict[base_col] = [alleles[-1]]
    features = []
    masks = []
    for ext_col in config.EXTENDED_STR_COLS:
        base_name = ext_col
        suffix_idx = -1
        for s_idx, sfx in enumerate(['a', 'b', 'c', 'd']):
            if ext_col.endswith(sfx) and ext_col[:-1] in config.MULTICOPIES:
                base_name = ext_col[:-1]
                suffix_idx = s_idx
                break
        alleles = parsed_dict.get(base_name, [-1])
        val = -1.0
        if suffix_idx != -1:
            if len(alleles) > suffix_idx:
                val = float(alleles[suffix_idx])
        else:
            if len(alleles) > 0:
                val = float(alleles[0])
        if val >= 0.0:
            features.append(val + 1.0)
            masks.append(1.0)
        else:
            features.append(0.0)
            masks.append(0.0)
    max_allowed_idx = config.MAX_ALLELE - 1
    features = np.clip(features, 0, max_allowed_idx).tolist()
    return features, masks


@app.route('/predict', methods=['POST'])
def predict_snp():
    print(f'Processing POST /predict...')
    req_json = request.get_json(silent=True)
    if not req_json or 'haplotype' not in req_json:
        return jsonify({'status': 'error', 'message': 'Некорректный или пустой JSON запрос'}), 400
    try:
        threshold_param = req_json.get('confidence', 0.5)
        haplotype_input = req_json['haplotype']
        if isinstance(haplotype_input, str):
            vals = [v.strip() for v in re.split(r'[\s,;\t]+', haplotype_input.strip()) if v.strip()]
            sample_dict = {}
            for i, col in enumerate(config.BASE_STR_COLS):
                if i < len(vals):
                    sample_dict[col] = vals[i]
                else:
                    sample_dict[col] = None
        elif isinstance(haplotype_input, dict):
            sample_dict = haplotype_input
        else:
            return jsonify({'status': 'error', 'message': 'Формат гаплотипа должен быть строкой или объектом'}), 400
        features, masks = process_sample_dict(sample_dict)
        features_matrix = np.array([features], dtype=np.float32)
        masks_matrix = np.array([masks], dtype=np.float32)
        inputs = np.hstack([features_matrix, masks_matrix])
        inputs_tensor = torch.tensor(inputs, dtype=torch.float32).to(config.DEVICE)
        probs = hierarchical_predict(
            predictor.model,
            inputs_tensor,
            predictor.parent_indices,
            predictor.model.level_tensor,
            predictor.model.max_level
        ).cpu().numpy()[0]
        active_indices = [idx for idx, p in enumerate(probs) if p >= threshold_param]
        levels_list = predictor.model.level_tensor.cpu().numpy().tolist()
        active_indices.sort(key=lambda idx: levels_list[idx])
        chain_results = [(predictor.sorted_snps[idx], float(probs[idx])) for idx in active_indices]
        tree_structure = build_recursive_tree(chain_results)
        return jsonify(tree_structure)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'status': 'error', 'message': f"Внутренняя ошибка сервера: {str(e)}"}), 500


if __name__ == '__main__':
    predictor = GeneticSingleModel()
    predictor.load_model()
    print('yFarseer ready!')
    serve(app, host='0.0.0.0', port=8080)
