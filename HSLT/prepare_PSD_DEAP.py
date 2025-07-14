import _pickle as cPickle
import os
import numpy as np
from scipy.signal import welch
import torch

class PrepareData:
    def __init__(self, processed_data_dir='./data_processed'):
        self.data = None
        self.label = None

        self.processed_data_dir = processed_data_dir 
        os.makedirs(self.processed_data_dir, exist_ok=True)
    
        self.sfreq = 128


    def run(self, subject_list, data_path, label_type, num_class):
        self.data_path = data_path
        self.label_type = label_type
        self.num_class = num_class

        for sub_idx in subject_list:
            data, label_original_sub = self.load_data_per_subject(sub_idx)
            labels, valid_idx = self.label_selection(label_original_sub)
            data = data[valid_idx]
            

            print(f'Data and label prepared for subject {sub_idx+1:02d} !')
            print(f'Data shape: {data.shape}, Label shape: {labels.shape}') # (760, 32, 5), (760,)
            
            file_name = f's{sub_idx+1:02d}_processed_data.npz'
            save_path = os.path.join(self.processed_data_dir, file_name)
            
            np.savez_compressed(save_path, data=data, labels=labels)
            print(f'Processed data saved to: {save_path}')
            print('=====================================================================')
            
        print(f"\nAll specified subjects' processed data has been saved to '{self.processed_data_dir}'.")


    def load_data_per_subject(self, sub_idx):
        sub_code = f's{sub_idx+1:02}.dat'
        subject_path = os.path.join(self.data_path, sub_code)
        
        with open(subject_path, 'rb') as f:
            subject = cPickle.load(f, encoding='latin1')
            
        label = subject['labels'] # Keep all 4 labels here initially
        raw_data = subject['data'][:, 0:32, :]  # (32, 8064)

        feature_list = []
        new_labels = []

        for trial_idx in range(raw_data.shape[0]):  # raw_data.shape[0] = 40
            trial = raw_data[trial_idx]
            label_trial = label[trial_idx]

            trial_data = trial[:, 3*128:]  # (32, 7680 - 384 = 7680)
            baseline = trial[:, :3*128] # (32, 384)

            baseline_mean = np.mean(baseline.reshape(32, 3, 128), axis=1)   # (32, 128)
            baseline_mean = np.tile(baseline_mean, (1, trial_data.shape[1] // 128))  # extend to match data shape (32, 7680)
            
            trial_data = trial_data - baseline_mean  

            window_size = 6 * 128  # 6 seconds
            step = 3*128  # 3 seconds overlap 
            for start in range(0, trial_data.shape[1] - window_size + 1, step):   # (7680 - 6*128 + 1) // 384 = 19
                seg = trial_data[:, start:start + window_size]  # shape (32, 768)

                de_features, psd_features = self.extract_DE_PSD_features(seg, self.sfreq)  # (32, 5), (32, 5)
                combined_features = np.concatenate((de_features, psd_features), axis=1)     # (32, 10)
                
                # feature_list.append(combined_features)
                feature_list.append(psd_features)   # use PSD features only
                new_labels.append(label_trial)

        data = np.array(feature_list)  # (19*40, 32, 5)
        label = np.array(new_labels)  # (19*40, 5), need to process to binary or 4-class
        print(f'- Baseline removed and segmented data: {data.shape}, label: {label.shape}')

        return data, label

    def extract_DE_PSD_features(self, segment, sfreq):
        n_channels, n_samples = segment.shape   # (32, 128)

        BANDS = {
            'theta': (4, 7),
            'slow_alpha': (8, 10),
            'alpha': (8, 12),
            'beta': (13, 30),
            'gamma': (30, 47)
        }

        DE = []
        PSD = []

        for channel_idx in range(n_channels):
            x = segment[channel_idx, :]

            freqs, psd = welch(x, sfreq, nperseg=128, window='hamming') # 0% overlap, take 1 second data each time

            channel_de = []
            channel_psd_db = []
            for band_name, (low, high) in BANDS.items():
                idx_band = np.logical_and(freqs >= low, freqs <= high)
                band_power = psd[idx_band].mean()
                psd_db = 10 * np.log10(band_power+1e-10)
                de = 0.5 * np.log(2 * np.pi * np.e * (band_power+1e-10))  

                channel_psd_db.append(psd_db)
                channel_de.append(de)

            DE.append(channel_de)
            PSD.append(channel_psd_db)

        DE = np.array(DE) # (32, num_bands=5)
        PSD = np.array(PSD) # (32, num_bands=5)
        return DE, PSD
    

    def label_selection(self, label):
        if self.num_class == 2:
            if self.label_type == 'A':
                label = label[:, 1]  # Arousal
            elif self.label_type == 'V':
                label = label[:, 0]  # Valence
            else:
                raise ValueError("Unsupported label type for 2-class. Choose 'A' (Arousal) or 'V' (Valence).")
            valid_idx = ((label >= 1) & (label <= 4)) | ((label >= 6) & (label <= 9))
            label = label[valid_idx]
            label = np.where(label <= 4, 0, 1)
            print('- Binary label generated!')
        elif self.num_class == 4:
            if self.label_type != 'AV':
                raise ValueError("For 4-class, label_type must be 'AV' (Arousal-Valence combined).")
            
            valence = label[:, 0]
            arousal = label[:, 1]

            valid_idx = (((valence >= 1) & (valence <= 4)) | ((valence >= 6) & (valence <= 9))) & \
                (((arousal >= 1) & (arousal <= 4)) | ((arousal >= 6) & (arousal <= 9)))

            valence = valence[valid_idx]
            arousal = arousal[valid_idx]
            
            # Initialize 4-class labels
            four_class_labels = np.zeros(valence.shape, dtype=int)
            four_class_labels[(arousal >= 6) & (valence >= 6)] = 0  # HAHV
            four_class_labels[(arousal >= 6) & (valence <= 4)] = 1  # HALV
            four_class_labels[(arousal <= 4) & (valence >= 6)] = 2  # LAHV
            four_class_labels[(arousal <= 4) & (valence <= 4)] = 3  # LALV

            label = four_class_labels
            print('4-Class (Arousal-Valence) labels generated!')
        else:
            raise ValueError("Unsupported num_class. Choose 2 or 4.")
        
        return label, valid_idx


if __name__ == '__main__':
    data_path = './DATA/DEAP/data_preprocessed_python'
    os.makedirs(data_path, exist_ok=True)
    processed_data_dir = './DATA/DEAP/processed_PSD_for_HSLT_test'
    os.makedirs(processed_data_dir, exist_ok=True)

    pd = PrepareData(processed_data_dir)
    pd.run(subject_list=range(1), data_path=data_path, label_type='A', num_class=2)




