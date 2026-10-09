import json
import os
import re
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


def hierarchical_predict(model, x, parent_indices, levels_list, max_level, threshold):
    model.eval()
    with torch.no_grad():
        logits = model(x)
        probs = torch.sigmoid(logits)
        batch_size = probs.size(0)
        final_active_mask = torch.zeros_like(probs, dtype=torch.bool)
        root_indices = [idx for idx, lvl in enumerate(levels_list) if lvl == 0]
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
                    leader_prob = root_probs[max_idx]
                    if leader_prob >= threshold:
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
                    if len(siblings) == 1:
                        sib_idx = siblings[0]
                        if probs[b, sib_idx] >= threshold:
                            final_active_mask[b, sib_idx] = True
                    else:
                        sib_probs = [probs[b, s_idx].item() for s_idx in siblings]
                        max_idx = np.argmax(sib_probs)
                        best_sib_idx = siblings[max_idx]
                        leader_prob = sib_probs[max_idx]
                        if leader_prob >= threshold:
                            final_active_mask[b, best_sib_idx] = True
        probs = torch.where(final_active_mask, probs, torch.tensor(0.0, device=probs.device))
    return probs


class GeneticSingleModel:
    def __init__(self):
        self.model = None
        self.sorted_snps = []
        self.parent_indices = []
        self.levels_list = []
        self.num_str_markers = len(config.EXTENDED_STR_COLS)
        self.max_allele_val = config.MAX_ALLELE
        self.embedding_dim = config.EMBEDDING_DIM

    def load_model(self):
        with open(os.path.join(config.MODEL_DIR, "snp_list.json"), "r", encoding="utf-8") as f:
            self.sorted_snps = json.load(f)
        with open(os.path.join(config.MODEL_DIR, "snp_levels.json"), "r", encoding="utf-8") as f:
            self.levels_list = json.load(f)
        with open(os.path.join(config.MODEL_DIR, "parent_indices.json"), "r", encoding="utf-8") as f:
            self.parent_indices = json.load(f)
        output_dim = len(self.sorted_snps)
        model_path = os.path.join(config.MODEL_DIR, "model_best_emr.pth")
        if os.path.exists(model_path):
            self.model = classes.GeneticEmbeddingMLP(self.num_str_markers, self.max_allele_val, self.embedding_dim,
                                                     output_dim, self.levels_list)
            self.model.load_state_dict(torch.load(model_path, map_location=config.DEVICE))
            self.model.to(config.DEVICE)
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
        inputs = np.hstack([features.astype(np.float32), masks.astype(np.float32)])
        inputs_tensor = torch.tensor(inputs, dtype=torch.float32).to(config.DEVICE)
        probs = hierarchical_predict(predictor.model, inputs_tensor, predictor.parent_indices, predictor.levels_list,
                                     predictor.model.max_level, threshold_param).cpu().numpy()[0]
        active_indices = [idx for idx, p in enumerate(probs) if p >= threshold_param]
        levels_list = predictor.model.level_tensor.cpu().numpy().tolist()
        active_indices.sort(key=lambda idx: levels_list[idx])
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
