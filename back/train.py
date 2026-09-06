import os
import time

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

import classes
import config
import utils

if __name__ == '__main__':
    print("Loading dataset...")
    df = utils.load_and_transform_dataset(only_complete=True)
    df_train, df_val = train_test_split(df, test_size=0.2, random_state=42)
    print("Loading topology based on train set...")
    unique_train_haplogroups = df_train['Haplogroup'].unique().tolist()
    topo_manager = classes.HierarchyTopologyManager()
    topo_manager.load_topology(unique_train_haplogroups)
    train_feat, train_mask = utils.build_matrices(df_train)
    val_feat, val_mask = utils.build_matrices(df_val)
    train_haplogroups = df_train['Haplogroup'].tolist()
    val_haplogroups = df_val['Haplogroup'].tolist()
    print("Generating masks and labels...")
    train_labels, train_lmasks = topo_manager.generate_labels_and_masks(train_haplogroups)
    val_labels, val_lmasks = topo_manager.generate_labels_and_masks(val_haplogroups)
    print("Calculating weights...")
    pos_counts = np.sum(train_labels * train_lmasks, axis=0)
    total_active_samples = np.sum(train_lmasks, axis=0)
    smoothed_pos = pos_counts + 1.0
    smoothed_total = total_active_samples + 2.0
    calculated_weights = (smoothed_total - smoothed_pos) / smoothed_pos
    calculated_weights = np.log1p(calculated_weights) + 1.0
    calculated_weights = np.clip(calculated_weights, 1.0, config.MAX_POS_WEIGHT)
    pos_weight_tensor = torch.tensor(calculated_weights, dtype=torch.float32).to(config.DEVICE)
    print("Preparing datasets...")
    train_dataset = classes.GeneticDataset(train_feat, train_mask, train_labels, train_lmasks, is_training=True)
    val_dataset = classes.GeneticDataset(val_feat, val_mask, val_labels, val_lmasks, is_training=False)
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    input_dim = train_feat.shape[1] * 2
    output_dim = train_labels.shape[1]
    num_str_markers = train_feat.shape[1]
    print("Preparing model...")
    model = classes.GeneticEmbeddingMLP(num_str_markers=num_str_markers, max_allele_val=config.MAX_ALLELE,
                                        embedding_dim=config.EMBEDDING_DIM, output_dim=output_dim,
                                        parent_indices=topo_manager.parent_indices,
                                        sibling_matrix=topo_manager.sibling_matrix) \
        .to(config.DEVICE)
    criterion = classes.MaskedBCELoss(topo_manager.parent_indices, pos_weight_tensor, topo_manager.sibling_matrix,
                                      level_tensor=model.level_tensor, max_level=model.max_level) \
        .to(config.DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE)
    scheduler = CosineAnnealingLR(optimizer, T_max=config.EPOCHS, eta_min=config.LEARNING_RATE / config.EPOCHS)
    best_val_emr = 0.0
    lengths_standards = [12, 25, 37, 67, 111]
    print("Ready to epochs...")
    for epoch in range(config.EPOCHS):
        start_time = time.time()
        train_dataset.update_epoch_augmentation()
        model.train()
        train_loss = 0.0
        total_train_samples = 0
        train_stats = {l: {"exact": 0, "under": 0, "over": 0, "false_branch": 0, "count": 0} for l in lengths_standards}
        for inputs, targets, masks in train_loader:
            inputs, targets, masks = inputs.to(config.DEVICE), targets.to(config.DEVICE), masks.to(config.DEVICE)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets, masks)
            loss.backward()
            optimizer.step()
            batch_size = inputs.size(0)
            train_loss += loss.item() * batch_size
            total_train_samples += batch_size
            num_features = inputs.size(1) // 2
            preds = (torch.sigmoid(outputs) > 0.5).float()
            active_preds = preds * masks
            active_targets = targets * masks
            fps = ((active_preds == 1.0) & (active_targets == 0.0)).sum(dim=1)
            fns = ((active_preds == 0.0) & (active_targets == 1.0)).sum(dim=1)
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
                train_stats[length]["exact"] += ((sub_fps == 0) & (sub_fns == 0)).sum().item()
                train_stats[length]["under"] += ((sub_fps == 0) & (sub_fns > 0)).sum().item()
                train_stats[length]["over"] += ((sub_fps > 0) & (sub_fns == 0)).sum().item()
                train_stats[length]["false_branch"] += ((sub_fps > 0) & (sub_fns > 0)).sum().item()
                train_stats[length]["count"] += length_mask.sum().item()
        epoch_time = time.time() - start_time
        train_loss /= total_train_samples
        train_report = ""
        for length in lengths_standards:
            c = train_stats[length]["count"] + 1e-8
            emr = train_stats[length]["exact"] / c
            under = train_stats[length]["under"] / c
            over = train_stats[length]["over"] / c
            fb = train_stats[length]["false_branch"] / c
            train_report += f" [{length} STR -> EMR: {emr:.3f}, Und: {under:.3f}, Ovr: {over:.3f}, Fls: {fb:.3f}]"
        train_b_loss = criterion.latest_base_loss
        train_h_loss = criterion.latest_hierarchy_loss
        train_s_loss = criterion.latest_sibling_loss
        val_loss, val_emr, val_report = utils.evaluate_model(model, val_loader, criterion, config.DEVICE)
        val_b_loss = criterion.latest_base_loss
        val_h_loss = criterion.latest_hierarchy_loss
        val_s_loss = criterion.latest_sibling_loss
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        print(f"Epoch {epoch + 1:02d} | LR: {current_lr:.6f} | Time: {epoch_time:.2f}s | "
              f"Train Loss: {train_loss:.4f} (B: {train_b_loss:.4f}, H: {train_h_loss:.4f}, S: {train_s_loss:.4f}) | "
              f"Valid Loss: {val_loss:.4f} (B: {val_b_loss:.4f}, H: {val_h_loss:.4f}, S: {val_s_loss:.4f})\n"
              f"  TRAIN GROUPS ->{train_report}\n"
              f"  VALID GROUPS ->{val_report}")
        if val_emr > best_val_emr:
            best_val_emr = val_emr
            best_model_path = os.path.join(config.MODEL_DIR, "model_best_emr.pth")
            torch.save(model.state_dict(), best_model_path)
            print(f"  --> Сохранена новая лучшая модель с VALID EMR (111 STR): {best_val_emr:.4f}")
