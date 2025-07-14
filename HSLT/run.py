

from eegdataset import EEGDataset
from prepare_PSD_DEAP import PrepareData
from utils import *


import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from sklearn.metrics import f1_score, cohen_kappa_score, accuracy_score
import os
import copy
from datetime import datetime
import random
from collections import Counter



def run_losocv(
    subject_ids_to_process,
    label_type,
    num_classes,
    epochs=500,
    batch_size=64,
    lr=0.001,
    random_state=42,
    device='cuda' if torch.cuda.is_available() else 'cpu',
    log_filename='result.txt',
    patience=10,
    prepare_data=False,
    metric_to_choose='f1',
    train_val_split_ratio=0.2,
    mode_version=1,
    weight_decay=0,
):
    log_setting(log_filename, {
        'kfolds': -1, 'random_state': random_state, 'label_type': label_type,
        'epochs': epochs, 'batch_size': batch_size, 'lr': lr,
    })

    seed_all(random_state)
    
    original_data_dir = '../DATA/DEAP/data_preprocessed_python' # change to your own path
    os.makedirs(original_data_dir, exist_ok=True)
    processed_data_dir = '../DATA/DEAP/processed_PSD_for_HSLT' + f'_{label_type}'   # change to your own path
    os.makedirs(processed_data_dir, exist_ok=True)

    if prepare_data:
        pd = PrepareData(processed_data_dir)
        pd.run(subject_list=subject_ids_to_process, data_path=original_data_dir, label_type=label_type, num_class=num_classes)

    all_test_acc = []
    all_test_f1_weighted = []
    all_test_kappa = []

    for sub_idx in subject_ids_to_process:
        log2txt(log_filename, f"\n============================== Processing Subject s{sub_idx+1:02d} ==============================")
        # ============================================ Dataset ============================================
        train_val_subject_ids = np.array([i for i in subject_ids_to_process if i != sub_idx])
        if train_val_split_ratio == 0:
            train_dataset = EEGDataset(subject_ids=train_val_subject_ids, data_dir=processed_data_dir)
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
            val_dataset = EEGDataset(subject_ids=[sub_idx], data_dir=processed_data_dir)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
            test_dataset = EEGDataset(subject_ids=[sub_idx], data_dir=processed_data_dir)
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
        else:
            train_ids, val_ids = train_test_split(
                train_val_subject_ids,
                test_size=train_val_split_ratio, 
                random_state=random_state, 
                shuffle=True)
            train_dataset = EEGDataset(subject_ids=train_ids, data_dir=processed_data_dir)
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
            val_dataset = EEGDataset(subject_ids=val_ids, data_dir=processed_data_dir)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
            test_dataset = EEGDataset(subject_ids=[sub_idx], data_dir=processed_data_dir)
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)


        # ============================================ Model ============================================
        if mode_version == 1:
            from network_v1 import HSLT
        elif mode_version == 2:
            from network_v2 import HSLT
        model = HSLT(num_classes=num_classes).to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        criterion = nn.CrossEntropyLoss()
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        best_val_loss, cur_patience = np.inf, 0
        best_val_acc, best_val_f1_weighted, best_val_kappa = 0, 0, 0
        best_idx = 0
        best_model = None
        
        for epoch in range(epochs):
            # ============================================ Train ============================================
            model.train()
            running_loss_train = 0.0
            train_preds, train_labels = [], []

            for batch_X, batch_y in train_loader:
                batch_X = {key: batch_X[key].to(device) for key in batch_X}
                batch_y = batch_y.to(device)
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

                running_loss_train += loss.item()
                _, predicted = torch.max(outputs, 1)
                train_preds.extend(predicted.cpu().numpy())
                train_labels.extend(batch_y.cpu().numpy())

            avg_train_loss = running_loss_train / len(train_loader)
            train_acc = accuracy_score(train_labels, train_preds)
            train_f1_weighted = f1_score(train_labels, train_preds, average='weighted')
            train_kappa = cohen_kappa_score(train_labels, train_preds)
            if (epoch + 1) % (epochs // 10) == 0 or epoch == 0 or epoch == epochs - 1:
                current_lr = optimizer.param_groups[0]['lr']
                log2txt(log_filename, f"Epoch {epoch+1:4d}/{epochs} | Current LR: {current_lr:.6f}")
                log2txt(log_filename, f">>> Train: loss={avg_train_loss:.6f}, acc={train_acc*100:.2f}%, "
                    f"f1={train_f1_weighted*100:.2f}%, kappa={train_kappa:.4f}")

            # ============================================ Validation ============================================
            model.eval()
            running_loss_val = 0.0
            all_val_preds, all_val_labels = [], []

            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X = {key: batch_X[key].to(device) for key in batch_X}
                    batch_y = batch_y.to(device)
                    outputs = model(batch_X)
                    val_loss = criterion(outputs, batch_y)
                    running_loss_val += val_loss.item()

                    _, predicted = torch.max(outputs, 1)
                    all_val_preds.extend(predicted.cpu().numpy())
                    all_val_labels.extend(batch_y.cpu().numpy())

            avg_val_loss = running_loss_val / len(val_loader)
            val_acc = accuracy_score(all_val_labels, all_val_preds)
            val_f1_weighted = f1_score(all_val_labels, all_val_preds, average='weighted')
            val_kappa = cohen_kappa_score(all_val_labels, all_val_preds)

            if (epoch + 1) % (epochs // 10) == 0 or epoch == 0 or epoch == epochs - 1:
                current_lr = optimizer.param_groups[0]['lr']
                log2txt(log_filename, f">>>   Val: loss={avg_val_loss:.6f}, acc={val_acc*100:.2f}%, "
                    f"f1={val_f1_weighted*100:.2f}%, kappa={val_kappa:.4f}")

            # ============================================ Early Stopping ============================================
            if metric_to_choose =='acc' and val_acc > best_val_acc:
                best_val_loss = avg_val_loss
                best_val_acc = val_acc
                best_val_f1_weighted = val_f1_weighted
                best_val_kappa = val_kappa
                cur_patience = 0
                best_idx = epoch+1
                best_model = copy.deepcopy(model)
            elif metric_to_choose =='f1' and val_f1_weighted > best_val_f1_weighted:
                best_val_loss = avg_val_loss
                best_val_acc = val_acc
                best_val_f1_weighted = val_f1_weighted
                best_val_kappa = val_kappa
                cur_patience = 0
                best_idx = epoch+1
                best_model = copy.deepcopy(model)
            elif metric_to_choose =='kappa' and val_kappa > best_val_kappa:
                best_val_loss = avg_val_loss
                best_val_acc = val_acc
                best_val_f1_weighted = val_f1_weighted
                best_val_kappa = val_kappa
                cur_patience = 0
                best_idx = epoch+1
                best_model = copy.deepcopy(model)
            elif metric_to_choose =='loss' and avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_val_acc = val_acc
                best_val_f1_weighted = val_f1_weighted
                best_val_kappa = val_kappa
                cur_patience = 0
                best_idx = epoch+1
                best_model = copy.deepcopy(model)
            else:
                cur_patience += 1
                if cur_patience >= patience:
                    log2txt(log_filename, f"- Early stopping at epoch {epoch+1}")
                    break
            scheduler.step()

        # ============================================ Testing ============================================
        model = best_model.to(device)
        model.eval()
        running_loss_test = 0.0
        all_test_preds, all_test_labels = [], []

        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                batch_X = {key: batch_X[key].to(device) for key in batch_X}
                batch_y = batch_y.to(device)
                outputs = model(batch_X)
                test_loss = criterion(outputs, batch_y)
                running_loss_test += test_loss.item()

                _, predicted = torch.max(outputs, 1)
                all_test_preds.extend(predicted.cpu().numpy())
                all_test_labels.extend(batch_y.cpu().numpy())

        avg_test_loss = running_loss_test / len(test_loader)
        test_acc = accuracy_score(all_test_labels, all_test_preds)
        test_f1_weighted = f1_score(all_test_labels, all_test_preds, average='weighted')
        test_kappa = cohen_kappa_score(all_test_labels, all_test_preds)

        log2txt(log_filename, f"\nTest Subject s{sub_idx+1:02d}: Choose the best model at {best_idx}.\n"
                f">>> BestVal: loss={best_val_loss:.6f}, acc={best_val_acc*100:.2f}%, f1={best_val_f1_weighted*100:.2f}%, kappa={best_val_kappa:.4f}\n"
                f">>>    Test: loss={avg_test_loss:.6f}, acc={test_acc*100:.2f}%, f1={test_f1_weighted*100:.2f}%, kappa={test_kappa:.4f}")

        all_test_acc.append(test_acc)
        all_test_f1_weighted.append(test_f1_weighted)
        all_test_kappa.append(test_kappa)

    return all_test_acc, all_test_f1_weighted, all_test_kappa
            



            