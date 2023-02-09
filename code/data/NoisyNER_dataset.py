from torchvision.datasets import CIFAR100
from torchvision import transforms
import pandas as pd
import os
from torch.utils.data import DataLoader
from cmd_args_noisyner import EXP_SETTINGS, TRAIN_SETTINGS
from networks.wideresnet import Wide_ResNet
import torch.nn as nn
import utils
from dataset import CoNllDataset

from models import create_base_model

class CoNllWithIdx(CoNllDataset):
    def __init__(self, instances, true_labels, onehot=False):
        super().__init__(instances, onehot=onehot)
        self.targets_gt = true_labels
    def __getitem__(self, index):
        """
        Args:
            index (int):  index of element to be fetched

        Returns:
            tuple: (embed, label, word, targets_gt, index) where index is the index of this sample in dataset.
        """
        emb, label, word = super().__getitem__(index)
        targets_gt = self.targets_gt[index]
        return emb, label, word, targets_gt, index


def get_noisyner_dataset(args):

    batch_size = TRAIN_SETTINGS["BATCH_SIZE"]
    num_workers = TRAIN_SETTINGS["NUM_WORKERS"]

    all_datasets, label_representation, test_label_representation, embedding_vector_size =\
        utils.load_all_processed_data(args)
    assert label_representation.get_num_labels() == EXP_SETTINGS["NUM_LABELS"]
    args.embedding_vector_size = embedding_vector_size

    # 直接用整个数据集
    train_clean, train_noisy, dev, test = all_datasets
    train_clean_dataset, train_noisy_dataset, dev_dataset, test_dataset = map(CoNllDataset, all_datasets)
    # 跟据噪声标签和真实标签构建训练集，方便记录下标和比较真假样本
    train_dataset = CoNllWithIdx(train_noisy, train_clean_dataset.labels) #主要是噪声集，但附带真实标签，用于比较
    train_loader = DataLoader(train_dataset, batch_size=batch_size, 
         shuffle=True, num_workers=num_workers)
    dev_loader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False,
                                              num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                                              num_workers=0)
    clean_ind = []
    noisy_ind = []
    for i, (emb, label, word, targets_gt, index) in enumerate(train_dataset):
        if label == targets_gt:
            clean_ind.append(i)
        else:
            noisy_ind.append(i)

    return train_loader, dev_loader, test_loader, train_noisy, dev, test, noisy_ind, clean_ind, label_representation

def get_NoisyNER_model_and_loss_criterion(args):
    """Initializes DNN model and loss function.

    Args:
        args (argparse.Namespace):

    Returns:
        model (torch.nn.Module): DNN model.
        criterion (torch.nn.modules.loss): cross entropy loss
    """
    model, feature_extractor = create_base_model(args.embedding_vector_size,
                                                         TRAIN_SETTINGS["LSTM_SIZE"], TRAIN_SETTINGS["DENSE_SIZE"],
                                                         TRAIN_SETTINGS["DENSE_ACTIVATION"],
                                                         EXP_SETTINGS["NUM_LABELS"])
    model = model.to(args.device)
    criterion = nn.CrossEntropyLoss(reduction='none').to(args.device)
    criterion_val = nn.CrossEntropyLoss(reduction='none').to(args.device)

    return model, criterion, criterion_val