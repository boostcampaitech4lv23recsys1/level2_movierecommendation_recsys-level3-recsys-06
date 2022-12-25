import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


class Preprocessor:
    def __init__(self):
        self.root_dir = "/opt/ml/input/data/train/"
        self.train_data_path = os.path.join(self.root_dir, "train_ratings.csv")
        self.title_data_path = os.path.join(self.root_dir, "titles.tsv")

        self.user_encoder = LabelEncoder()
        self.item_encoder = LabelEncoder()
    
    def _load_train_dataset(self):
        interaction_df = pd.read_csv(self.train_data_path, low_memory = False)
        title_df = pd.read_csv(self.title_data_path, sep = '\t', low_memory = False)
        
        return interaction_df, title_df

    
    def _preprocess_dataset(self):
        interaction_df, title_df = self._load_train_dataset()
        self.user_encoder.fit(interaction_df['user'])
        self.item_encoder.fit(title_df['item'])
        #TODO: 전처리단에서 수행한 LabelEncoder 받아서 다시 수행해야한다.
        interaction_df['user'] = self.user_encoder.transform(interaction_df['user'])
        interaction_df['item'] = self.item_encoder.transform(interaction_df['item'])
        title_df['item'] = self.item_encoder.transform(title_df['item'])
        interaction_df = interaction_df[['user', 'item']]
        return interaction_df, title_df
        
        


if __name__ == '__main__':
    preprocessor = Preprocessor()
    preprocessor._load_train_dataset()