import os
import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder


class Preprocessor:
    def __init__(self):
        self.root_dir = "/opt/ml/input/data/train/"
        self.train_data_path = os.path.join(self.root_dir, "train_ratings.csv")
        self.title_data_path = os.path.join(self.root_dir, "titles.tsv")
        self.asset_dir = "/opt/ml/level2_movierecommendation_recsys-level3-recsys-06/saved/asset"
        self.label_encoders_path = os.path.join(self.asset_dir, "labels.pkl")
        with open(self.label_encoders_path, 'rb') as f:
            label_encoders = pickle.load(f)
        # 0: 감독/작가 라벨
        # 1: 유저 라벨
        # 2: 장르 라벨
        # 3: 영화 라벨
        # 4: 제목 라벨
        self.direc_writer_encoder = label_encoders[0]
        self.user_encoder = label_encoders[1]
        self.genre_encoder = label_encoders[2]
        self.item_encoder = label_encoders[3]
        self.title_encoder = label_encoders[4]

    
    def _load_train_dataset(self):
        interaction_df = pd.read_csv(self.train_data_path, low_memory = False)
        title_df = pd.read_csv(self.title_data_path, sep = '\t', low_memory = False)
        interaction_df.sort_values(['user', 'time'], inplace=True)
        return interaction_df, title_df

    
    def _preprocess_dataset(self):
        interaction_df, title_df = self._load_train_dataset()

        interaction_df['user'] = self.user_encoder.transform(interaction_df['user'])
        interaction_df['item'] = self.item_encoder.transform(interaction_df['item'])
        title_df['item'] = self.item_encoder.transform(title_df['item'])
        interaction_df = interaction_df[['user', 'item']]
        return interaction_df, title_df

    def _preprocess_testset(self):
        interaction_df, title_df = self._preprocess_dataset()
        return interaction_df, title_df, self.user_encoder, self.item_encoder



if __name__ == '__main__':
    preprocessor = Preprocessor()
    preprocessor._load_train_dataset()