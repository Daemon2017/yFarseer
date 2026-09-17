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
    masks = (~np.isnan(features)).astype(np.float32)
    features = np.floor(np.nan_to_num(features, nan=-1.0)).astype(np.int64)
    features = np.where(features >= 0, features + 1, 0)
    max_allowed_idx = config.MAX_ALLELE - 1
    features = np.clip(features, a_min=0, a_max=max_allowed_idx)
    return features.astype(np.float32), masks


def accumulate_metrics_from_batch(inputs, outputs, targets, masks, stats, lengths_standards, force_length=None):
    num_features = inputs.size(1) // 2
    preds = (torch.sigmoid(outputs) > config.THRESHOLD).float()
    active_preds = preds * masks
    active_targets = targets * masks
    fps = ((active_preds == 1.0) & (active_targets == 0.0)).sum(dim=1)
    fns = ((active_preds == 0.0) & (active_targets == 1.0)).sum(dim=1)
    if force_length is not None:
        assigned_standards = torch.full((inputs.size(0),), force_length, dtype=torch.long, device=inputs.device)
    else:
        mask_vals = inputs[:, num_features:]
        sample_lengths = mask_vals.sum(dim=1).long()
        assigned_standards = torch.zeros_like(sample_lengths)
        assigned_standards = torch.where(sample_lengths <= 12, 12, assigned_standards)
        assigned_standards = torch.where((sample_lengths > 12) & (sample_lengths <= 25), 25, assigned_standards)
        assigned_standards = torch.where((sample_lengths > 25) & (sample_lengths <= 37), 37, assigned_standards)
        assigned_standards = torch.where((sample_lengths > 37) & (sample_lengths <= 67), 67, assigned_standards)
        assigned_standards = torch.where(sample_lengths > 67, 111, assigned_standards)
    for length in lengths_standards:
        length_mask = (assigned_standards == length)
        if not length_mask.any():
            continue
        sub_fps = fps[length_mask]
        sub_fns = fns[length_mask]
        stats[length]["exact"] += ((sub_fps == 0) & (sub_fns == 0)).sum().item()
        stats[length]["under"] += ((sub_fps == 0) & (sub_fns > 0)).sum().item()
        stats[length]["over"] += ((sub_fps > 0) & (sub_fns == 0)).sum().item()
        stats[length]["false_branch"] += ((sub_fps > 0) & (sub_fns > 0)).sum().item()
        stats[length]["count"] += length_mask.sum().item()


def evaluate_model(model, loader, criterion, device, lengths_standards):
    model.eval()
    total_loss = 0.0
    total_samples = 0
    total_b = 0.0
    total_h = 0.0
    total_s = 0.0
    stats = {l: {"exact": 0, "under": 0, "over": 0, "false_branch": 0, "count": 0} for l in lengths_standards}
    with torch.no_grad():
        for inputs, labels, masks in loader:
            inputs, labels, masks = inputs.to(device), labels.to(device), masks.to(device)
            batch_size = inputs.size(0)
            total_samples += batch_size
            outputs = model(inputs)
            loss = criterion(outputs, labels, masks)
            total_loss += loss.item() * batch_size
            total_b += criterion.latest_base_loss * batch_size
            total_h += criterion.latest_hierarchy_loss * batch_size
            total_s += criterion.latest_sibling_loss * batch_size
            num_features = inputs.size(1) // 2
            base_feat = inputs[:, :num_features]
            base_mask = inputs[:, num_features:]
            for length in lengths_standards:
                feat_sub = base_feat.clone()
                mask_sub = base_mask.clone()
                if length < num_features:
                    feat_sub[:, length:] = 0.0
                    mask_sub[:, length:] = 0.0
                inputs_sub = torch.hstack([feat_sub, mask_sub])
                outputs_sub = model(inputs_sub)
                accumulate_metrics_from_batch(inputs=inputs_sub, outputs=outputs_sub, targets=labels, masks=masks,
                                              stats=stats, lengths_standards=lengths_standards, force_length=length)
    mean_loss = total_loss / (total_samples + 1e-8)
    mean_b = total_b / (total_samples + 1e-8)
    mean_h = total_h / (total_samples + 1e-8)
    mean_s = total_s / (total_samples + 1e-8)
    val_emr = stats[111]["exact"] / (stats[111]["count"] + 1e-8)
    report_str = ""
    for length in lengths_standards:
        c = stats[length]["count"] + 1e-8
        emr = stats[length]["exact"] / c
        under = stats[length]["under"] / c
        over = stats[length]["over"] / c
        fb = stats[length]["false_branch"] / c
        report_str += f" [{length} STR -> EMR: {emr:.3f}, Und: {under:.3f}, Ovr: {over:.3f}, Fls: {fb:.3f}]"
    return mean_loss, mean_b, mean_h, mean_s, val_emr, report_str


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
    synonym_to_snp = {f"{node['root']}-{synonym['variant']}": node['name']
                      for node in topology.get('allNodes', {}).values()
                      for synonym in node['variants']}
    return synonym_to_snp


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
