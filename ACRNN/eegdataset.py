import os.path as osp
import scipy.io as sio
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import os


class EEGDataset(Dataset):
    def __init__(self, subject_ids, data_dir='../DATA/DEAP/processed_data_for_ACRNN', transform=None):
        self.data = []
        self.labels = []
        self.transform = transform

        for subject_id in subject_ids:
            file_path = os.path.join(data_dir, f's{subject_id+1:02d}_processed_data.npz')
            if not os.path.exists(file_path):
                print(f"[Warning] File not found: {file_path}")
                continue
            npzfile = np.load(file_path)
            self.data.append(npzfile['data'])     # (800, 32, 384)
            self.labels.append(npzfile['labels']) # (800,)
        
        self.data = np.concatenate(self.data, axis=0)   # shape: (total, 32, 384)
        self.labels = np.concatenate(self.labels, axis=0) # shape: (total,)

        print(f"data shape: {self.data.shape}, labels shape: {self.labels.shape}")


    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        sample = self.data[idx]       
        label = self.labels[idx]      

        if self.transform:
            sample = self.transform(sample)

        sample = torch.tensor(sample, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.long)

        return sample, label



