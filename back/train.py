import json
import os
import time

import numpy as np
import pandas as pd
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

import classes
import config
import utils

if __name__ == '__main__':
    print("Loading datasets...")
    str_dtypes = {col: str for col in config.BASE_STR_COLS}
    with open(config.DATES_FILE, 'r', encoding='utf-8') as f:
        dates = json.load(f)
    with open(config.TOPOLOGY_FILE, 'r', encoding='utf-8') as f:
        topology = json.load(f)
    df = pd.read_csv(config.HAPLOTYPES_FILE, usecols=config.BASE_STR_COLS + ['Haplogroup'], dtype=str_dtypes,
                     low_memory=False, encoding='utf-8')
    snp_to_tmrca = utils.get_snp_to_tmrca(dates.get('node', {}))
    df = utils.transform_dataset(df, True, 0, topology, snp_to_tmrca)
    counts = df['Haplogroup'].value_counts()
    df_singles = df[df['Haplogroup'].isin(counts[counts == 1].index)]
    df_multiples = df[df['Haplogroup'].isin(counts[counts > 1].index)]
    df_multiples_shuffled = df_multiples.sample(frac=1, random_state=42)
    df_val = df_multiples_shuffled.drop_duplicates(subset=['Haplogroup'])
    df_train_multi = df_multiples[~df_multiples.index.isin(df_val.index)]
    df_train = pd.concat([df_train_multi, df_singles]).sample(frac=1, random_state=42).reset_index(drop=True)
    df_val = df_val.sample(frac=1, random_state=42).reset_index(drop=True)
    print(f"Total rows: {len(df)}")
    print(f"Train: {len(df_train)} rows")
    print(f"Validation: {len(df_val)} rows")
    print("Loading topology based on train set...")
    unique_train_haplogroups = df_train['Haplogroup'].unique().tolist()
    topo_manager = classes.HierarchyTopologyManager()
    topo_manager.prepare_topology(unique_train_haplogroups, topology)
    print(f"Total SNPs: {len(topo_manager.all_snps)}")
    train_feat, train_mask = utils.build_matrices(df_train)
    val_feat, val_mask = utils.build_matrices(df_val)
    train_haplogroups = df_train['Haplogroup'].tolist()
    val_haplogroups = df_val['Haplogroup'].tolist()
    print("Generating masks and labels...")
    train_labels, train_lmasks = topo_manager.generate_labels_and_masks(train_haplogroups)
    val_labels, val_lmasks = topo_manager.generate_labels_and_masks(val_haplogroups)
    samples_per_snp = train_labels.sum(axis=0)
    median_samples = np.median(samples_per_snp)
    print(f"Median samples/SNP: {median_samples:.1f}")
    print("Preparing datasets...")
    train_dataset = classes.GeneticDataset(train_feat, train_mask, train_labels, train_lmasks, is_training=True,
                                           all_snps=topo_manager.all_snps, snp_to_tmrca=snp_to_tmrca,
                                           parent_indices=topo_manager.parent_indices,
                                           snp_levels=topo_manager.snp_levels)
    val_dataset = classes.GeneticDataset(val_feat, val_mask, val_labels, val_lmasks, is_training=False,
                                         all_snps=topo_manager.all_snps, snp_to_tmrca=snp_to_tmrca,
                                         parent_indices=topo_manager.parent_indices,
                                         snp_levels=topo_manager.snp_levels)
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, drop_last=True, num_workers=1,
                              pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False, drop_last=False, num_workers=1,
                            pin_memory=True, persistent_workers=True)
    cached_val_batches = []
    for inputs, labels, masks in val_loader:
        cached_val_batches.append((inputs, labels.float(), masks.float()))
    output_dim = train_labels.shape[1]
    num_str_markers = train_feat.shape[1]
    pos_weight_tensor = torch.ones(output_dim, dtype=torch.float32).to(config.DEVICE)
    print("Preparing model...")
    model = classes.GeneticEmbeddingMLP(num_str_markers=num_str_markers, max_allele_val=config.MAX_ALLELE,
                                        embedding_dim=config.EMBEDDING_DIM, output_dim=output_dim,
                                        snp_levels=topo_manager.snp_levels).to(config.DEVICE)
    print(f"Maximum tree depth: {model.max_level} SNPs")
    criterion = classes.MaskedBCELoss(pos_weight=pos_weight_tensor, level_tensor=model.level_tensor,
                                      parent_indices=topo_manager.parent_indices).to(config.DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=1e-2)
    scheduler = CosineAnnealingLR(optimizer, T_max=config.EPOCHS, eta_min=config.LEARNING_RATE / config.EPOCHS)
    best_val_emr = 0.0
    print("Ready to epochs...")
    for epoch in range(config.EPOCHS):
        train_start_time = time.time()
        train_dataset.update_epoch_augmentation()
        model.train()
        train_loss = 0.0
        total_train_samples = 0
        train_stats = {"exact": 0, "total_f1": 0.0, "count": 0}
        for inputs, targets, masks in train_loader:
            inputs = inputs.to(config.DEVICE)
            targets = targets.to(config.DEVICE).float()
            masks = masks.to(config.DEVICE).float()
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets, masks)
            loss.backward()
            optimizer.step()
            batch_size = inputs.size(0)
            train_loss += loss.item() * batch_size
            total_train_samples += batch_size
            utils.accumulate_metrics_from_batch(outputs=outputs, targets=targets, masks=masks, stats=train_stats,
                                                criterion=criterion)
        train_loss /= total_train_samples
        train_count = train_stats["count"] + 1e-8
        train_emr = train_stats["exact"] / train_count
        train_f1 = train_stats["total_f1"] / train_count
        train_finish_time = time.time() - train_start_time
        val_start_time = time.time()
        val_loss, val_emr, val_report = utils.evaluate_model(model=model, loader=cached_val_batches,
                                                             criterion=criterion, device=config.DEVICE)
        scheduler.step()
        val_finish_time = time.time() - val_start_time
        current_lr = scheduler.get_last_lr()[0]
        print(f"Epoch {epoch + 1:02d} | LR: {current_lr:.6f} | "
              f"Train Time: {train_finish_time:.2f}s | Valid Time: {val_finish_time:.2f}s | "
              f"Train Loss: {train_loss:.4f} | Valid Loss: {val_loss:.4f}\n"
              f"  TRAIN -> Global EMR: {train_emr:.4f} | Path-level F1: {train_f1:.4f}\n"
              f"  VALID -> {val_report}")
        if val_emr > best_val_emr:
            best_val_emr = val_emr
            best_model_path = os.path.join(config.MODEL_DIR, "model_best_emr.pth")
            torch.save(model.state_dict(), best_model_path)
            print(f"  --> Сохранена новая лучшая модель с VALID GLOBAL EMR: {best_val_emr:.4f}")
