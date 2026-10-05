import re

import numpy as np
import pandas as pd
import torch

import config


def parse_str_value(val, col_base):
    if pd.isna(val):
        return []
    s = str(val).strip().lower()
    if s in ['none', 'nan', 'null', '', 'unknown']:
        return []
    found = re.findall(r'\d+(?:\.\d+)?', s)
    if not found:
        return []
    alleles = sorted([int(float(x)) for x in found])
    if col_base in config.MULTICOPIES:
        expected = config.MULTICOPIES[col_base]
        actual = len(alleles)
        if actual == expected:
            return alleles
        elif actual > expected:
            if expected == 2 and actual > 2:
                return [alleles[0], alleles[-1]]
            elif expected == 4 and actual > 4:
                return [alleles[0], alleles[1], alleles[-2], alleles[-1]]
    else:
        if len(alleles) > 1:
            return [alleles[-1]]
        return alleles


def build_matrices(df):
    features = df[config.EXTENDED_STR_COLS].values
    masks = (~np.isnan(features)).astype(np.uint8)
    features = np.floor(np.nan_to_num(features, nan=-1.0)).astype(np.int16)
    features = np.where(features >= 0, features + 1, 0)
    max_allowed_idx = config.MAX_ALLELE - 1
    features = np.clip(features, a_min=0, a_max=max_allowed_idx)
    return features.astype(np.int64), masks


def accumulate_metrics_from_batch(outputs, targets, masks, stats, criterion=None):
    if criterion is not None:
        with torch.no_grad():
            probs_raw = torch.sigmoid(outputs)
            log_probs_raw = torch.log(probs_raw + 1e-7)
            h_log_probs = log_probs_raw + torch.sparse.mm(criterion.ancestry_matrix, log_probs_raw.t()).t()
            h_probs = torch.clamp(torch.exp(h_log_probs), min=1e-7, max=1.0 - 1e-7)
            outputs = torch.log(h_probs / (1.0 - h_probs))
    preds = (torch.sigmoid(outputs) > config.TRAIN_THRESHOLD).float()
    active_preds = preds * masks
    active_targets = targets * masks
    fps = ((active_preds == 1.0) & (active_targets == 0.0)).sum(dim=1)
    fns = ((active_preds == 0.0) & (active_targets == 1.0)).sum(dim=1)
    exact_matches = ((fps == 0) & (fns == 0)).sum().item()
    tp = (active_preds * active_targets).sum(dim=1)
    pred_count = active_preds.sum(dim=1)
    target_count = active_targets.sum(dim=1)
    precision = tp / (pred_count + 1e-8)
    recall = tp / (target_count + 1e-8)
    f1_path = 2 * (precision * recall) / (precision + recall + 1e-8)
    stats["exact"] += exact_matches
    stats["total_f1"] += f1_path.sum().item()
    stats["count"] += inputs_size_helper(outputs)


def inputs_size_helper(tensor):
    return tensor.size(0)


def evaluate_model(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_samples = 0
    stats = {"exact": 0, "total_f1": 0.0, "count": 0}
    is_cached = isinstance(loader, list)
    with torch.no_grad():
        for inputs, labels, masks in (loader if is_cached else loader):
            if is_cached:
                inputs = inputs.to(device)
                labels = labels.to(device)
                masks = masks.to(device)
            else:
                inputs = inputs.to(device)
                labels = labels.to(device).float()
                masks = masks.to(device).float()
            batch_size = inputs.size(0)
            total_samples += batch_size
            outputs = model(inputs)
            loss = criterion(outputs, labels, masks)
            total_loss += loss.item() * batch_size
            probs_raw = torch.sigmoid(outputs)
            log_probs_raw = torch.log(probs_raw + 1e-7)
            h_log_probs = log_probs_raw + torch.sparse.mm(criterion.ancestry_matrix, log_probs_raw.t()).t()
            h_probs = torch.exp(h_log_probs)
            h_probs = torch.clamp(h_probs, min=1e-7, max=1.0 - 1e-7)
            simulated_outputs = torch.log(h_probs / (1.0 - h_probs))
            accumulate_metrics_from_batch(outputs=simulated_outputs, targets=labels, masks=masks, stats=stats)
    mean_loss = total_loss / (total_samples + 1e-8)
    total_count = stats["count"] + 1e-8
    global_emr = stats["exact"] / total_count
    mean_f1 = stats["total_f1"] / total_count
    report_str = f"Global EMR: {global_emr:.4f} | Path-level F1: {mean_f1:.4f}"
    return mean_loss, global_emr, report_str


def get_snp_to_tmrca(data):
    snp_to_tmrca = {}
    stack = [data]
    while stack:
        node = stack.pop()
        if (name := node.get('name')) and (tmrca := node.get('tmrca')):
            snp_to_tmrca[name] = tmrca.get('mean')
        if children := node.get('children'):
            stack.extend(children)
    return snp_to_tmrca


def get_synonym_to_snp(topology):
    return {f"{node['root']}-{synonym['variant']}": node['name']
            for node in topology.get('allNodes', {}).values()
            for synonym in node['variants']}


def split_multicopies(df):
    transformed = {}
    for base_col in config.BASE_STR_COLS:
        parsed = df[base_col].apply(lambda x: parse_str_value(x, base_col))
        if base_col in config.MULTICOPIES:
            suffixes = ['a', 'b', 'c', 'd'][:config.MULTICOPIES[base_col]]
            for i, suf in enumerate(suffixes):
                transformed[f"{base_col}{suf}"] = parsed.apply(
                    lambda x: float(x[i]) if x is not None and len(x) > i else np.nan)
        else:
            transformed[base_col] = parsed.apply(lambda x: float(x[-1]) if x is not None and len(x) > 0 else np.nan)
    return transformed


def transform_dataset(df, only_complete=True, age_threshold=-3000, topology=None, snp_to_tmrca=None):
    df = df.dropna(subset=['Haplogroup'])
    df = df[~df['Haplogroup'].isin(['-'])]
    synonym_to_snp = get_synonym_to_snp(topology)
    df['Canonical_Haplogroup'] = df['Haplogroup'].astype(str).str.strip().map(synonym_to_snp) \
        .fillna(df['Haplogroup'].astype(str).str.strip())
    allowed_snps = {snp for snp, age in snp_to_tmrca.items() if age is not None and age >= age_threshold}
    df = df[df['Canonical_Haplogroup'].isin(allowed_snps)]
    df['Haplogroup'] = df['Canonical_Haplogroup']
    splited = split_multicopies(df)
    df_clean = pd.DataFrame(splited, index=df.index)[config.EXTENDED_STR_COLS]
    df_clean['Haplogroup'] = df['Haplogroup'].values
    df_clean = df_clean.drop_duplicates(subset=config.EXTENDED_STR_COLS + ['Haplogroup'])
    if only_complete:
        df_clean = df_clean.dropna(subset=config.EXTENDED_STR_COLS)
    return df_clean
