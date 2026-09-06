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


def evaluate_model(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_samples = 0
    lengths_standards = [12, 25, 37, 67, 111]
    stats = {l: {"exact": 0, "under": 0, "over": 0, "false_branch": 0, "count": 0} for l in lengths_standards}
    with torch.no_grad():
        for inputs, labels, masks in loader:
            inputs, labels, masks = inputs.to(device), labels.to(device), masks.to(device)
            batch_size = inputs.size(0)
            total_samples += batch_size
            outputs = model(inputs)
            loss = criterion(outputs, labels, masks)
            total_loss += loss.item() * batch_size
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
                preds = (torch.sigmoid(outputs_sub) > 0.5).float()
                active_preds = preds * masks
                active_labels = labels * masks
                fps = ((active_preds == 1.0) & (active_labels == 0.0)).sum(dim=1)
                fns = ((active_preds == 0.0) & (active_labels == 1.0)).sum(dim=1)
                exact_mask = (fps == 0) & (fns == 0)
                under_mask = (fps == 0) & (fns > 0)
                over_mask = (fps > 0) & (fns == 0)
                false_branch_mask = (fps > 0) & (fns > 0)
                stats[length]["exact"] += exact_mask.sum().item()
                stats[length]["under"] += under_mask.sum().item()
                stats[length]["over"] += over_mask.sum().item()
                stats[length]["false_branch"] += false_branch_mask.sum().item()
                stats[length]["count"] += batch_size
    mean_loss = total_loss / (total_samples + 1e-8)
    val_emr = stats[111]["exact"] / (stats[111]["count"] + 1e-8)
    report_str = ""
    for length in lengths_standards:
        c = stats[length]["count"] + 1e-8
        emr = stats[length]["exact"] / c
        under = stats[length]["under"] / c
        over = stats[length]["over"] / c
        fb = stats[length]["false_branch"] / c
        report_str += f" [{length} STR -> EMR: {emr:.3f}, Und: {under:.3f}, Ovr: {over:.3f}, Fls: {fb:.3f}]"
    return mean_loss, val_emr, report_str


def load_and_transform_dataset(only_complete=True):
    str_dtypes = {col: str for col in config.BASE_STR_COLS}
    df = pd.read_csv(config.DATA_PATH, usecols=config.BASE_STR_COLS + ['Haplogroup'], dtype=str_dtypes,
                     low_memory=False, encoding='utf-8')
    df = df.dropna(subset=['Haplogroup'])
    df = df[~df['Haplogroup'].isin(['-'])]
    haplogroup_counts = df['Haplogroup'].value_counts()
    wgs_haplogroups = haplogroup_counts[haplogroup_counts <= 25].index
    df = df[df['Haplogroup'].isin(wgs_haplogroups)]
    df['Haplogroup'] = df['Haplogroup'].astype(str).str.strip()
    transformed = {}
    for col in config.EXTENDED_STR_COLS:
        transformed[col] = np.nan
    for base_col in config.BASE_STR_COLS:
        parsed = df[base_col].apply(lambda x: parse_str_value(x, base_col))
        if base_col in config.MULTICOPIES:
            suffixes = ['a', 'b', 'c', 'd']
            for i in range(config.MULTICOPIES[base_col]):
                sub_col = f"{base_col}{suffixes[i]}"
                transformed[sub_col] = parsed.apply(lambda x: float(x[i]) if x is not None and len(x) > i else np.nan)
        else:
            transformed[base_col] = parsed.apply(lambda x: float(x[-1]) if x is not None and len(x) > 0 else np.nan)
    df_clean = pd.DataFrame(transformed, index=df.index)
    df_clean = df_clean[config.EXTENDED_STR_COLS]
    df_clean['Haplogroup'] = df['Haplogroup']
    df_clean = df_clean.drop_duplicates(subset=config.EXTENDED_STR_COLS + ['Haplogroup'])
    if only_complete:
        df_clean = df_clean.dropna(subset=config.EXTENDED_STR_COLS)
    return df_clean
