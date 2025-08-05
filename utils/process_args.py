import argparse


def process_args():

    parser = argparse.ArgumentParser(description='Configurations for TANGLE pretraining')

    #----> User setup args
    parser.add_argument('--data_dir', type=str, help='Path to tile embeddings.')
    parser.add_argument('--info_csv', type=str, help='Path to info csv.')
    parser.add_argument('--study', type=str, choices=['pdl1', 'p53', 'ki67'], help='study to train on')
    parser.add_argument('--dataset_csv', type=str, help='Path to the train csv file')

    #----> training args
    parser.add_argument('--folds', type=int, default=5, help='Fold number.')
    parser.add_argument('--warmup', type=bool, default=True, help='If doing warmup.')
    parser.add_argument('--warmup_epochs', type=int, default=2, help='Number of warmup epochs.')
    parser.add_argument('--epochs', type=int, default=25, help='maximum number of epochs to train (default: 2)')
    parser.add_argument('--learning_rate', type=float, default=5e-5, help='learning rate (default: 0.0001)')
    parser.add_argument('--end_learning_rate', type=float, default=5e-7, help='learning rate (default: 0.0001)')
    parser.add_argument('--temperature', type=float, default=0.01, help='InfoNCE temperature.')
    parser.add_argument('--batch_size', type=int, default=8, help='batch_size')
    parser.add_argument('--n_tokens', type=int, default=8000, help='Number of patches to sample during training.')
    parser.add_argument('--num_workers', type=int, default=0, help='number of cpu workers')
    parser.add_argument('--gpu_devices', type=str, default="0,1,2,3", help='List of GPUs.')
    parser.add_argument('--only_class_loss', action='store_true', default=False, help='If only using classification loss.')
    parser.add_argument('--seed', type=int, default=1234, help='random seed for reproducible experiment (default: 1)')

    args = parser.parse_args()

    return args
