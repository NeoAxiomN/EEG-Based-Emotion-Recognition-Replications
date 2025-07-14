import os.path as osp
import scipy.io as sio
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import os


class EEGDataset(Dataset):
    def __init__(self, subject_ids, data_dir='../DATA/DEAP/processed_PSD_for_ACRNN', transform=None):
        self.data = []
        self.labels = []
        self.transform = transform

        self.electrode_to_channel_map = {
            'FP1': 0, 'AF3': 1, 'AF4': 2, 'FP2': 3,
            'F7': 4, 'F3': 5, 'Fz': 6, 'F4': 7, 'F8': 8,
            'FC5': 9, 'T7': 10, 'CP5': 11,
            'FC1': 12, 'C3': 13, 'Cz': 14, 'C4': 15, 'FC2': 16,
            'FC6': 17, 'T8': 18, 'CP6': 19,
            'P7': 20, 'P3': 21, 'PO3': 22,
            'CP1': 23, 'Pz': 24, 'CP2': 25,
            'P8': 26, 'P4': 27, 'PO4': 28,
            'O1': 29, 'Oz': 30, 'O2': 31,
        }

        self.brain_regions_electrodes = {
            "PF": ["FP1", "AF3", "AF4", "FP2"],
            "F": ["F7", "F3", "Fz", "F4", "F8"],
            "LT": ["FC5", "T7", "CP5"],
            "C": ["FC1", "C3", "Cz", "C4", "FC2"],
            "RT": ["FC6", "T8", "CP6"],
            "LP": ["P7", "P3", "PO3"],
            "P": ["CP1", "Pz", "CP2"],
            "RP": ["P8", "P4", "PO4"],
            "O": ["O1", "Oz", "O2"]
        }

        self.region_channel_indices = {}
        for region_name, electrodes in self.brain_regions_electrodes.items():
            current_region_indices = []
            for electrode in electrodes:
                if electrode in self.electrode_to_channel_map:
                    current_region_indices.append(self.electrode_to_channel_map[electrode])
            if current_region_indices:
                self.region_channel_indices[region_name] = torch.tensor(
                    sorted(current_region_indices), dtype=torch.long)

        for subject_id in subject_ids:
            file_path = os.path.join(data_dir, f's{subject_id+1:02d}_processed_data.npz')
            if not os.path.exists(file_path):
                print(f"[Warning] File not found: {file_path}")
                continue
            npzfile = np.load(file_path)
            self.data.append(npzfile['data'])     # (2400, 32, 5)
            self.labels.append(npzfile['labels']) # (2400,)
        
        self.data = np.concatenate(self.data, axis=0)   # shape: (total, 32, 5)
        self.labels = np.concatenate(self.labels, axis=0) # shape: (total,)

        self.data = torch.from_numpy(self.data).to(torch.float32)
        self.labels = torch.from_numpy(self.labels).to(torch.long) 

        print(f"data shape: {self.data.shape}, labels shape: {self.labels.shape}")



    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        sample = self.data[idx]       
        label = self.labels[idx]      

        if self.transform:
            sample = self.transform(sample)

        processed_regions_data = {}
        for region_name, channel_indices_tensor in self.region_channel_indices.items():
            x_region = sample[channel_indices_tensor, :]
            processed_regions_data[region_name] = x_region

        return processed_regions_data, label


if __name__ == '__main__':
    dataset = EEGDataset(subject_ids=[0], data_dir='./DATA/DEAP/processed_PSD_for_HSLT_test')
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    for batch_idx, (batch_region, batch_label) in enumerate(dataloader):
        if batch_idx >= 1:
            break
        print(batch_label.shape)
        for region_name, region_data in batch_region.items():
            print(f"shape of {region_name}: {region_data.shape}")



