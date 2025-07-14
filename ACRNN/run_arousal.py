from networks import ACRNN
from eegdataset import *
from prepare_data_ACRNN import PrepareData 

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score
import os
import copy
from datetime import datetime
import random

def seed_all(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def log2txt(filename, content):
    print(content)
    os.makedirs('Result', exist_ok=True)
    log_path = os.path.join(os.getcwd(), 'Result', filename)
    with open(log_path, 'a') as file:
        file.write(str(content) + '\n')


def log_setting(log_filename, args:dict):
    log2txt(log_filename, f"Experiment started at {datetime.now()}")
    log2txt(log_filename, f"Device: {torch.cuda.get_device_name(0)}")
    log2txt(log_filename, f"KFolds: {args['kfolds']}, Random state: {args['random_state']}, Label type: {args['label_type']}")
    log2txt(log_filename, f"Epochs: {args['epochs']}, Batch size: {args['batch_size']}, Learning rate: {args['lr']}")


def run_full_experiment(
    subject_ids_to_process,
    label_type,
    kfolds=10,
    epochs=500,
    batch_size=64,
    lr=0.001,
    validation_split_ratio=0.2,
    random_state=42,
    device='cuda' if torch.cuda.is_available() else 'cpu',
    log_filename='result.txt',
    patience=10
):
    log_setting(log_filename, {
        'kfolds': kfolds, 'random_state': random_state, 'label_type': label_type,
        'epochs': epochs, 'batch_size': batch_size, 'lr': lr,
    })

    seed_all(random_state)
    
    original_data_dir = '../DATA/DEAP/data_preprocessed_python'
    os.makedirs(original_data_dir, exist_ok=True)
    processed_data_dir = '../DATA/DEAP/processed_for_ACRNN' + f'_{label_type}'
    os.makedirs(processed_data_dir, exist_ok=True)

    pd = PrepareData(processed_data_dir)
    pd.run(subject_list=subject_ids_to_process, data_path=original_data_dir, label_type=label_type, num_class=2)

    all_subjects_mean_accuracies = []

    for sub_idx in subject_ids_to_process:
        log2txt(log_filename, f"\n============================== Processing Subject s{sub_idx+1:02d} ==============================")

        dataset = EEGDataset(subject_ids=[sub_idx], data_dir=processed_data_dir)
        kf = KFold(n_splits=kfolds, shuffle=True, random_state=random_state)

        fold_acc = []

        for fold, (train_indices, test_indices) in enumerate(kf.split(range(len(dataset)))):
            log2txt(log_filename, f"Fold: {fold+1:2d}/{kfolds}:")
            # ============================================ Dataset ============================================
            train_dataset = torch.utils.data.Subset(dataset, train_indices)
            test_dataset = torch.utils.data.Subset(dataset, test_indices)

            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

            model = ACRNN().to(device)
            optimizer = optim.Adam(model.parameters(), lr=lr)
            criterion = nn.CrossEntropyLoss()


            # ============================================ Train ============================================
            for epoch in range(epochs):
                model.train()
                correct, total = 0, 0
                running_loss = 0.0
                for batch_X, batch_y in train_loader:
                    batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                    optimizer.zero_grad()
                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    optimizer.step()

                    running_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    total += batch_y.size(0)
                    correct += (predicted == batch_y).sum().item()

                train_acc = correct / total
                if (epoch + 1) % (epochs // 10) == 0 or epoch == 0 or epoch == epochs - 1:
                    log2txt(log_filename, f"Epoch {epoch+1:4d}/{epochs} | Average Train Loss: {running_loss / len(train_loader):.6f}"
                          f" | Train Acc: {train_acc:.4f}")

            # ============================================ Test ============================================
            model.eval()
            with torch.no_grad():
                correct_test, total_test = 0, 0
                test_loss = 0.0
                for batch_X, batch_y in test_loader:
                    batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                    outputs = model(batch_X)

                    loss = criterion(outputs, batch_y)
                    test_loss += loss.item()

                    _, predicted = torch.max(outputs.data, 1)
                    total_test += batch_y.size(0)
                    correct_test += (predicted == batch_y).sum().item()

                test_acc = correct_test / total_test
                log2txt(log_filename, f"Fold: {fold+1:2d} | Average Test Loss: {test_loss / len(test_loader):.6f} | Final Test Acc: {test_acc:.4f}")

            fold_acc.append(test_acc)
        
        
        mean_acc = np.mean(fold_acc)
        log2txt(log_filename, f"\nSubject s{sub_idx+1:02d}: Mean Accuracy: {mean_acc:.4f} | std: {np.std(fold_acc):.4f}")
        all_subjects_mean_accuracies.append(mean_acc)
        
    
    all_subjects_std_accuracies = np.std(all_subjects_mean_accuracies)
    log2txt(log_filename, f"\nAll Subjects Mean Accuracy: {np.mean(all_subjects_mean_accuracies):.4f} ± {all_subjects_std_accuracies:.4f}")
    return {
        "mean_accuracies": all_subjects_mean_accuracies,
        "std_accuracies": all_subjects_std_accuracies
    }
            


if __name__ == '__main__':
    idx_begin = 1
    idx_end = 32

    epochs = 200
    time_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    EXPERIMENT_PARAMS = {
        'subject_ids_to_process': list(range(idx_begin-1, idx_end)), 
        'label_type': 'A',        # 'A', 'V', 'AV'               
        'kfolds': 10,            
        'epochs': epochs, 
        'batch_size': 10,          
        'lr': 1e-4,       
        'validation_split_ratio': 0.2, # 80% train, 20% validation from training folds
        'random_state': 42,        
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'log_filename': f'ACRNN_DEAP_10folds_Arousal_{time_str}_result.txt',
        'patience': epochs // 1
    }


    print(f"Train on {'cuda' if torch.cuda.is_available() else 'cpu'}")
    final_results = run_full_experiment(**EXPERIMENT_PARAMS)
    
    print("\n============================== Final Experiment Results Across All Subjects ==============================")
    for idx, acc in enumerate(final_results['mean_accuracies']):
        print(f"Subject {idx+1:02d}: {acc:.4f}")
    print(f"\nOverall Mean Accuracy: {np.mean(final_results['mean_accuracies']):.4f} ± {final_results['std_accuracies']:.4f}")
