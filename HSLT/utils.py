import os
import torch
import numpy as np
import random
from datetime import datetime


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




def log_setting(log_filename, args: dict):
    log2txt(log_filename, f"Experiment started at {datetime.now()}")
    log2txt(log_filename, f"Device: {torch.cuda.get_device_name(0)}")
    log2txt(log_filename, f"KFolds: {args['kfolds']}, Random state: {args['random_state']}, Label type: {args['label_type']}")
    log2txt(log_filename, f"Epochs: {args['epochs']}, Batch size: {args['batch_size']}, Learning rate: {args['lr']}")