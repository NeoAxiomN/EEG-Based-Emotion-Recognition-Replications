from run import run_losocv
from utils import *

import argparse
import torch
from datetime import datetime
import numpy as np


if __name__ == '__main__':
    time_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--subject_ids_to_process', type=list, default=range(32), help='0-31')
    parser.add_argument('--label_type', type=str, choices=['A', 'V', 'AV'], default='A')
    parser.add_argument('--num_classes', type=int, choices=[2, 4], default=2, help='2 or 4')
    parser.add_argument('--random_state', type=int, default=0)
    parser.add_argument('--need_to_prepare_data', type=bool, default=False)

    parser.add_argument('--epochs', type=int, default=80)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--lr', type=float, default=5e-3)
    parser.add_argument('--train_val_split_ratio', type=float, default=0.2, 
                        help="if 0.2, 80% train, 20% val; if 0., use test-dataset as val-dataset")

    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--log_filename', type=str, default=f'HSLT_DEAP_{time_str}_result.txt')

    parser.add_argument('--metric_to_choose', type=str, default='loss', choices=['acc', 'f1', 'kappa', 'loss'])
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--weight_decay', type=float, default=0)

    parser.add_argument('--model_version', type=int, default=1, choices=[1, 2])



 
    args = parser.parse_args()
    all_test_acc, all_test_f1_weighted, all_test_kappa = run_losocv(
        subject_ids_to_process=args.subject_ids_to_process,
        label_type=args.label_type,
        num_classes=args.num_classes,
        epochs=args.epochs,
        batch_size=args.batch_size,
        random_state=args.random_state,
        lr=args.lr,
        device=args.device,
        log_filename=args.log_filename,
        patience=args.patience,
        prepare_data=args.need_to_prepare_data,
        metric_to_choose=args.metric_to_choose,
        train_val_split_ratio=args.train_val_split_ratio,
        mode_version=args.model_version,
        weight_decay=args.weight_decay
    )


    mean_acc = np.mean(all_test_acc)
    std_acc = np.std(all_test_acc)
    mean_f1_weighted = np.mean(all_test_f1_weighted)
    std_f1_weighted = np.std(all_test_f1_weighted)
    mean_kappa = np.mean(all_test_kappa)
    std_kappa = np.std(all_test_kappa)

    log_filename = args.log_filename


    print("\n")
    log2txt(log_filename, f"\n=============================== LOSO-CV Summary ==============================")
    args_dict = vars(args)
    for arg_name, arg_value in args_dict.items():
        log2txt(log_filename, f"{arg_name}: {arg_value}")
    log2txt(log_filename, f"")

    for i, acc in enumerate(all_test_acc):
        log2txt(log_filename, f"Subject s{i+1:02d}: P_acc={acc:.4f}, P_f={all_test_f1_weighted[i]:.4f}, P_ck={all_test_kappa[i]:.4f}")
    log2txt(log_filename, f"")
    log2txt(log_filename, f"All Subjects Mean P_acc={mean_acc*100:.2f}%, STD={std_acc*100:.2f}%")
    log2txt(log_filename, f"All Subjects Mean P_f={mean_f1_weighted*100:.2f}%, STD={std_f1_weighted*100:.2f}%")
    log2txt(log_filename, f"All Subjects Mean P_ck={mean_kappa:.4f}, STD={std_kappa:.4f}")