import _pickle as cPickle
import os
import numpy as np

class PrepareData:
    def __init__(self, processed_data_dir='./data_processed'):
        self.data = None
        self.label = None

        self.processed_data_dir = processed_data_dir 
        os.makedirs(self.processed_data_dir, exist_ok=True)
        
        self.original_order = ['Fp1', 'AF3', 'F3', 'F7', 'FC5', 'FC1', 'C3', 'T7', 'CP5', 'CP1', 'P3', 'P7', 'PO3',
                               'O1', 'Oz', 'Pz', 'Fp2', 'AF4', 'Fz', 'F4', 'F8', 'FC6', 'FC2', 'Cz', 'C4', 'T8', 'CP6',
                               'CP2', 'P4', 'P8', 'PO4', 'O2']


    def run(self, subject_list, data_path, label_type, num_class):
        self.data_path = data_path
        self.label_type = label_type
        self.num_class = num_class

        for sub_idx in subject_list:
            data, label_original_sub = self.load_data_per_subject(sub_idx)
            labels = self.label_selection(label_original_sub)
            

            print(f'Data and label prepared for subject {sub_idx+1:02d} !')
            print(f'Data shape: {data.shape}, Label shape: {labels.shape}')
            
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

        data_list = []
        new_labels = []

        for trial_idx in range(raw_data.shape[0]):  # raw_data.shape[0] = 40
            trial = raw_data[trial_idx]
            label_trial = label[trial_idx]

            trial_data = trial[:, 3*128:]  # (32, 7680 - 384 = 7680)
            baseline = trial[:, :3*128] # (32, 384)

            baseline_mean = np.mean(baseline.reshape(32, 3, 128), axis=1)   # (32, 128)
            baseline_mean = np.tile(baseline_mean, (1, trial_data.shape[1] // 128))  # extend to match data shape (32, 7680)
            
            trial_data = trial_data - baseline_mean  

            window_size = 3 * 128  # 384
            step = window_size
            for start in range(0, trial_data.shape[1] - window_size + 1, step): # for loop 60/3 = 20 times
                seg = trial_data[:, start:start + window_size]  # shape (32, 384)
                data_list.append(seg)
                new_labels.append(label_trial)

        data = np.array(data_list)  # (20*40, 32, 384)
        label = np.array(new_labels)  # (20*40, 4)
        print(f'- Baseline removed and segmented data: {data.shape}, label: {label.shape}')

        return data, label


    def label_selection(self, label):
        if self.num_class == 2:
            if self.label_type == 'A':
                label = label[:, 1]  # Arousal
            elif self.label_type == 'V':
                label = label[:, 0]  # Valence
            else:
                raise ValueError("Unsupported label type for 2-class. Choose 'A' (Arousal) or 'V' (Valence).")
            label = np.where(label <= 5, 0, 1)
            print('- Binary label generated!')
        elif self.num_class == 4:
            if self.label_type != 'AV':
                raise ValueError("For 4-class, label_type must be 'AV' (Arousal-Valence combined).")
            
            valence = label[:, 0]
            arousal = label[:, 1]
            
            # Initialize 4-class labels
            four_class_labels = np.zeros(valence.shape, dtype=int)
            four_class_labels[(arousal > 5) & (valence > 5)] = 0
            four_class_labels[(arousal > 5) & (valence <= 5)] = 1
            four_class_labels[(arousal <= 5) & (valence > 5)] = 2
            four_class_labels[(arousal <= 5) & (valence <= 5)] = 3

            label = four_class_labels
            print('4-Class (Arousal-Valence) labels generated!')
        else:
            raise ValueError("Unsupported num_class. Choose 2 or 4.")
        return label


if __name__ == '__main__':
    data_path = './DATA/DEAP/data_preprocessed_python'
    os.makedirs(data_path, exist_ok=True)
    processed_data_dir = './DATA/DEAP/processed_data_for_ACRNN'
    os.makedirs(processed_data_dir, exist_ok=True)

    pd = PrepareData(processed_data_dir)
    pd.run(subject_list=range(32), data_path=data_path, label_type='A', num_class=2)




