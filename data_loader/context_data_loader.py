import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset


class StaticDataset(Dataset):
    def __init__(self, data, neg_items_dict, user_dict, item_dict, config):
        """
        우선 사용할 feature는 user, item, item의 director, item의 writer, item의 genre
        지금 구현해야할 게 너무 많아서 이거 먼저 짜놓고 생각하기
        """
        self.data = data.values
        self.config = config
        self.director_maxlen = 14
        self.writer_maxlen = 24
        self.genre_maxlen = 10

        self.user_features = ['favorite_genre', 'maniatic']
        self.item_features = ['release_year', 'categorized_year_gap5', 'categorized_year_gap10', 'title', 'director', 'main_director', 'writer', 'main_writer', 'genre']
        self.neg_items_dict = neg_items_dict

        self.user_dict = user_dict
        self.item_dict = item_dict


    def make_features(self, data_dict, idx, name):
        if name == 'director':
            max_len = self.director_maxlen
        elif name == 'writer':
            max_len = self.writer_maxlen
        elif name == 'genre':
            max_len = self.genre_maxlen
        else:
            max_len = 1
        data_features = np.zeros(max_len)

        data_features[-len(data_dict[idx][name]):] = np.array(data_dict[idx][name])
        data_features = torch.tensor(data_features)
        
        return data_features

    def __getitem__(self, index):
        user_idx, pos_item_idx = self.data[index]
        neg_item_indices = np.random.choice(list(self.neg_items_dict[user_idx]), self.config['neg_ratio'])
        total_user_indices = np.array([user_idx] * (1 + self.config['neg_ratio']))
        total_item_indices = np.append(np.array([pos_item_idx]), neg_item_indices)

        concat_list = []
        for i in range(self.config['neg_ratio'] + 1):
            data = torch.tensor([total_user_indices[i], total_item_indices[i]])
            for name in self.config['using_features']:
                if name in self.user_features:
                    feature = self.make_features(self.user_dict, total_user_indices[i], name)
                else:
                    feature = self.make_features(self.item_dict, total_item_indices[i], name)
                data = torch.hstack([data, feature])
            concat_list.append(data.unsqueeze(0))
        
        features = torch.cat(concat_list, dim = 0)
        print("features shape:", features.shape)

        return features

    
    def __len__(self):
        return len(self.data)