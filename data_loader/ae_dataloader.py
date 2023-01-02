import argparse
import time
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np

import pickle

from scipy import sparse
import os
import pandas as pd
from scipy import sparse
import numpy as np


class AETrainDataSet(Dataset):
    def __init__(self, data):
        self.data = data
    
    def __getitem__(self, index):
        return torch.Tensor(self.data[index].toarray())

    def __len__(self):
        return self.data.shape[0]


class AETestDataSet(Dataset):
    def __init__(self, tr_data, te_data):
        self.tr_data = tr_data
        self.te_data = te_data
    
    def __getitem__(self, index):
        tensor_tr = torch.Tensor(self.tr_data[index].toarray())
        tensor_te = torch.Tensor(self.te_data[index].toarray())
        
        return torch.concat([tensor_tr, tensor_te], dim=1)

    def __len__(self):
        return self.tr_data.shape[0]


def ae_data_load(data_path, k_fold):

    train_data_path = os.path.join(data_path, f'train_{str(k_fold)}.csv')
    test_data_path = os.path.join(data_path, f'test_{str(k_fold)}.csv')

    tr_df = pd.read_csv(train_data_path)
    te_df = pd.read_csv(test_data_path)

    n_users = len(tr_df['uid'].unique())
    n_items = 6807

    rows_tr, cols_tr = tr_df['uid'], tr_df['sid'] 
    rows_te, cols_te = te_df['uid'], te_df['sid']

    train_data = sparse.csr_matrix((np.ones_like(rows_tr),
                                (rows_tr, cols_tr)), dtype='float64', shape=(n_users, n_items))
    test_data = sparse.csr_matrix((np.ones_like(rows_te),
                                (rows_te, cols_te)), dtype='float64', shape=(n_users, n_items))

    return train_data, test_data


def get_labels(data_path):
    with open(data_path+"/user_label.pkl", "rb") as f:
        user_label = pickle.load(f)
    with open(data_path+"/item_label.pkl", "rb") as f:
        item_label = pickle.load(f)

    return user_label, item_label