import json
import os
import traceback

import numpy as np
import pandas as pd
import torch
from flask import Flask, request, jsonify
from flask_cors import CORS
from waitress import serve

import classes
import config
import utils

app = Flask(__name__)
CORS(app)

predictor = None


def compute_hierarchical_probs_inference(logits, ancestry_matrix, brother_matrix, single_child_mask):
    eps = 1e-7
    sig_probs = torch.sigmoid(logits)
    exp_logits = torch.exp(torch.clamp(logits, max=12.0))
    brother_sums = torch.sparse.mm(brother_matrix, exp_logits.t()).t()
    softmax_probs = exp_logits / (1.0 + brother_sums + eps)
    cond_probs = torch.where(single_child_mask.unsqueeze(0), sig_probs, softmax_probs)
    cond_probs = torch.clamp(cond_probs, min=eps, max=1.0)
    log_cond_probs = torch.log(cond_probs)
    hierarchical_log_probs = log_cond_probs + torch.sparse.mm(ancestry_matrix, log_cond_probs.t()).t()
    p_h = torch.clamp(torch.exp(hierarchical_log_probs), min=eps, max=1.0 - eps)
    return p_h


def hierarchical_predict(model, x, parent_indices, computed_levels, max_level, ancestry_matrix, brother_matrix,
                         single_child_mask, threshold):
    model.eval()
    with torch.no_grad():
        logits = model(x)
        probs = compute_hierarchical_probs_inference(logits, ancestry_matrix, brother_matrix, single_child_mask)
        batch_size = probs.size(0)
        final_active_mask = torch.zeros_like(probs, dtype=torch.bool)
        root_indices = [idx for idx, p_idx in enumerate(parent_indices) if p_idx == -1]
        if root_indices:
            for b in range(batch_size):
                if len(root_indices) == 1:
                    best_root_idx = root_indices[0]
                    if probs[b, best_root_idx] >= threshold:
                        final_active_mask[b, best_root_idx] = True
                else:
                    root_probs = [probs[b, r_idx].item() for r_idx in root_indices]
                    max_idx = np.argmax(root_probs)
                    best_root_idx = root_indices[max_idx]
                    if root_probs[max_idx] >= threshold:
                        final_active_mask[b, best_root_idx] = True
        for lvl in range(1, max_level + 1):
            lvl_indices = [idx for idx, l in enumerate(computed_levels) if l == lvl]
            if not lvl_indices:
                continue
            for b in range(batch_size):
                b_lvl_indices = [idx for idx in lvl_indices if parent_indices[idx] != -1
                                 and final_active_mask[b, parent_indices[idx]]]
                if not b_lvl_indices:
                    continue
                parents_groups = {}
                for idx in b_lvl_indices:
                    p_idx = parent_indices[idx]
                    parents_groups.setdefault(p_idx, []).append(idx)
                for p_idx, siblings in parents_groups.items():
                    if len(siblings) == 1:
                        sib_idx = siblings[0]
                        if probs[b, sib_idx] >= threshold:
                            final_active_mask[b, sib_idx] = True
                    else:
                        sib_probs = [probs[b, s_idx].item() for s_idx in siblings]
                        max_idx = np.argmax(sib_probs)
                        best_sib_idx = siblings[max_idx]
                        if sib_probs[max_idx] >= threshold:
                            final_active_mask[b, best_sib_idx] = True
        probs = torch.where(final_active_mask, probs, torch.tensor(0.0, device=probs.device))
    return probs


class GeneticSingleModel:
    def __init__(self):
        self.model = None
        self.sorted_snps = []
        self.parent_indices = []
        self.computed_levels = []
        self.max_level = 0
        self.ancestry_matrix = None
        self.brother_matrix = None
        self.single_child_mask = None
        self.num_str_markers = len(config.EXTENDED_STR_COLS)
        self.max_allele_val = config.MAX_ALLELE
        self.embedding_dim = config.EMBEDDING_DIM

    def load_model(self):
        with open(os.path.join(config.MODEL_DIR, "snp_list.json"), "r", encoding="utf-8") as f:
            self.sorted_snps = json.load(f)
        with open(os.path.join(config.MODEL_DIR, "parent_indices.json"), "r", encoding="utf-8") as f:
            self.parent_indices = json.load(f)
        output_dim = len(self.sorted_snps)
        print("Precalculating...")
        device = torch.device(config.DEVICE)
        self.computed_levels = [0] * output_dim
        for i in range(output_dim):
            path_len = 0
            curr = self.parent_indices[i]
            while curr != -1:
                path_len += 1
                curr = self.parent_indices[curr]
            self.computed_levels[i] = path_len
        self.max_level = max(self.computed_levels) if self.computed_levels else 0
        row_indices, col_indices = [], []
        for i in range(output_dim):
            curr = self.parent_indices[i]
            while curr != -1:
                row_indices.append(i)
                col_indices.append(curr)
                curr = self.parent_indices[curr]
        indices = torch.tensor([row_indices, col_indices], dtype=torch.long, device=device)
        values = torch.ones(len(row_indices), dtype=torch.float32, device=device)
        self.ancestry_matrix = torch.sparse_coo_tensor(indices, values, (output_dim, output_dim)).coalesce()
        parent_to_children = {}
        for child_idx, parent_idx in enumerate(self.parent_indices):
            parent_to_children.setdefault(parent_idx, []).append(child_idx)
        self.single_child_mask = torch.zeros(output_dim, dtype=torch.bool, device=device)
        b_rows, b_cols = [], []
        for children in parent_to_children.values():
            if len(children) == 1:
                self.single_child_mask[children] = True
            else:
                for c1 in children:
                    for c2 in children:
                        b_rows.append(c1)
                        b_cols.append(c2)
        b_indices = torch.tensor([b_rows, b_cols], dtype=torch.long, device=device)
        b_values = torch.ones(len(b_rows), dtype=torch.float32, device=device)
        self.brother_matrix = torch.sparse_coo_tensor(b_indices, b_values, (output_dim, output_dim)).coalesce()
        model_path = os.path.join(config.MODEL_DIR, "model_best.pth")
        if os.path.exists(model_path):
            self.model = classes.GeneticEmbeddingMLP(self.num_str_markers, self.max_allele_val, self.embedding_dim,
                                                     output_dim, snp_depth_years=None)
            print("Loading model...")
            self.model.load_state_dict(torch.load(model_path, map_location=config.DEVICE))
            self.model.to(device)
            self.model.eval()


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
    raw_sample = {col: [sample_dict.get(col, None)] for col in config.BASE_STR_COLS}
    raw_sample['Haplogroup'] = [predictor.sorted_snps[0]]
    df_raw = pd.DataFrame(raw_sample)
    df_splited = pd.DataFrame(utils.split_multicopies(df_raw))[config.EXTENDED_STR_COLS]
    features, masks = utils.build_matrices(df_splited)
    return features, masks


@app.route('/predict', methods=['POST'])
def predict_snp():
    print(f'Processing POST /predict...')
    req_json = request.get_json(silent=True)
    if not req_json or 'haplotype' not in req_json:
        return jsonify({'status': 'error', 'message': 'Некорректный или пустой JSON запрос'}), 400
    try:
        threshold_param = req_json['confidence']
        haplotype_input = req_json['haplotype']
        if not isinstance(haplotype_input, dict):
            return jsonify({'status': 'error', 'message': 'Формат гаплотипа должен быть JSON'}), 400
        features, masks = process_sample_dict(haplotype_input)
        inputs = np.hstack([features.astype(np.float32), masks.astype(np.float32)])
        inputs_tensor = torch.tensor(inputs, dtype=torch.float32).to(config.DEVICE)
        probs = hierarchical_predict(predictor.model, inputs_tensor, predictor.parent_indices,
                                     predictor.computed_levels, predictor.max_level, predictor.ancestry_matrix,
                                     predictor.brother_matrix, predictor.single_child_mask, threshold_param)
        probs = probs.cpu().numpy()[0]
        active_indices = [idx for idx, p in enumerate(probs) if p >= threshold_param]
        active_indices.sort(key=lambda idx: predictor.computed_levels[idx])
        chain_results = [(predictor.sorted_snps[idx], float(probs[idx])) for idx in active_indices]
        tree_structure = build_recursive_tree(chain_results)
        return jsonify(tree_structure)
    except Exception as e:
        traceback.print_exc()
        return jsonify({'status': 'error', 'message': f"Внутренняя ошибка сервера: {str(e)}"}), 500


if __name__ == '__main__':
    predictor = GeneticSingleModel()
    predictor.load_model()
    print('yFarseer ready!')
    serve(app, host='0.0.0.0', port=8080)
