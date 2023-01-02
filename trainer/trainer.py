import numpy as np
import torch
from torchvision.utils import make_grid
from base import BaseTrainer
from utils import inf_loop, MetricTracker
from sklearn.model_selection import KFold, train_test_split


class GBDTTrainer():
    """
    Trainer class
    """
    def __init__(self, config, total_df, test_df, user_num):
        self.config = config
        self.probs = np.zeros((5,user_num))
        self.total_df = total_df
        self.test_df = test_df
    

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        #index로 나누고,
        kf = KFold(n_splits = 5, shuffle = True, random_state = 42)
        for idx, (train_index, valid_index) in enumerate(kf.split(self.total_df.index)):
            #negative sampling을 fold별로 다르게 뽑히게
            print(f"[FOLD: {idx + 1}] catboost")
            #train,valid 어떻게 나눌지 - 
            #fold 별로 확률 값 평균

            