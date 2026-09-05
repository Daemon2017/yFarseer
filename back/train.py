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
        train_stats = {l: {"tp": 0, "fp": 0, "fn": 0, "exact": 0, "count": 0} for l in lengths_standards}
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
            for i in range(batch_size):
                mask_vals = inputs[i, num_features:]
                sub_length = int(mask_vals.sum().item())
                if sub_length <= 12:
                    sub_length = 12
                elif sub_length <= 25:
                    sub_length = 25
                elif sub_length <= 37:
                    sub_length = 37
                elif sub_length <= 67:
                    sub_length = 67
                else:
                    sub_length = 111
                row_tp = ((active_preds[i] == 1.0) & (active_targets[i] == 1.0)).sum().item()
                row_fp = ((active_preds[i] == 1.0) & (active_targets[i] == 0.0)).sum().item()
                row_fn = ((active_preds[i] == 0.0) & (active_targets[i] == 1.0)).sum().item()
                row_error = ((active_preds[i] != active_targets[i]) * masks[i]).sum().item()
                train_stats[sub_length]["tp"] += row_tp
                train_stats[sub_length]["fp"] += row_fp
                train_stats[sub_length]["fn"] += row_fn
                if row_error == 0:
                    train_stats[sub_length]["exact"] += 1
                train_stats[sub_length]["count"] += 1
        epoch_time = time.time() - start_time
        train_loss /= total_train_samples
        train_report = ""
        for length in lengths_standards:
            tp = train_stats[length]["tp"]
            fp = train_stats[length]["fp"]
            fn = train_stats[length]["fn"]
            p = tp / (tp + fp + 1e-8)
            r = tp / (tp + fn + 1e-8)
            f1 = 2 * (p * r) / (p + r + 1e-8)
            emr = train_stats[length]["exact"] / (train_stats[length]["count"] + 1e-8)
            train_report += f" [{length} STR -> F1: {f1:.3f}, EMR: {emr:.3f}]"
        train_b_loss = criterion.latest_base_loss
        train_h_loss = criterion.latest_hierarchy_loss
        train_s_loss = criterion.latest_sibling_loss
        val_loss, val_f1, val_emr, val_report = utils.evaluate_model(model, val_loader, criterion, config.DEVICE)
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
