import os
import numpy as np
import pandas as pd
import pickle
import random
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder


class Preprocessor:
    def __init__(self):
        self.root_dir = "/opt/ml/input/data/train/"
        self.train_data_path = os.path.join(self.root_dir, "train_ratings.csv")
        self.title_data_path = os.path.join(self.root_dir, "titles.tsv")
        self.asset_dir = "/opt/ml/level2_movierecommendation_recsys-level3-recsys-06/saved/asset"
        self.label_encoders_path = os.path.join(self.asset_dir, "fm_labels.pkl")
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

        self.iteraction_df = None
        self.item_df = None
        self.user_df = None

        item_popular_path = os.path.join(self.asset_dir, "item_popular2.pkl")
        with open(item_popular_path, 'rb') as f:
            self.item_popular = pickle.load(f)
        
        user_item_count_path = os.path.join(self.asset_dir, "user_item_count.pkl")
        with open(user_item_count_path, 'rb') as f:
            self.user_item_count = pickle.load(f)

        

    
    def _load_train_dataset(self):
        interaction_df = pd.read_csv(self.train_data_path, low_memory = False)
        title_df = pd.read_csv(self.title_data_path, sep = '\t', low_memory = False)
        
        return interaction_df, title_df

    
    def _preprocess_dataset(self):
        interaction_df, title_df = self._load_train_dataset()

        interaction_df['user'] = self.user_encoder.transform(interaction_df['user'])
        interaction_df['item'] = self.item_encoder.transform(interaction_df['item'])
        title_df['item'] = self.item_encoder.transform(title_df['item'])
        interaction_df = interaction_df[['user', 'item']]

        self.interaction_df = interaction_df

        return interaction_df, title_df

    def _preprocess_testset(self):
        interaction_df, title_df = self._preprocess_dataset()
        return interaction_df, title_df, self.user_encoder, self.item_encoder

    def _make_dataset(self, item_dict, user_dict, use_genre):
        item_use_features = ["release_year", "categorized_year_gap5", "categorized_year_gap10", "title","director","main_director","writer","main_writer","genre"]
        item_multi_features = ["director", "writer","genre"]
        item_df = pd.DataFrame(columns=item_use_features)

        user_use_features = ["favorite_genre","maniatic"]
 

        for item, values in item_dict.items():
            temp = []
            for column, value in values.items():
                if column in item_multi_features:
                    temp.append(value)
                else:
                    temp.append(value[0]) #값이 여러개가 아닌 경우 0번째 값만 넣어줌
            item_df.loc[item] = temp
        
        genre_class = self.genre_encoder.classes_
        genre_df = pd.DataFrame(columns=genre_class)

        if use_genre: #장르의 경우 multi hot encoding으로 진행하기 위해 따로 빼줌
            for item, value in item_df.iterrows():
                temp = [0] * len(genre_class)
                for i in item_df["genre"][item]:
                    temp[i-1] = 1 #라벨링은 1부터
                genre_df.loc[item] = temp

        del item_df["genre"]
        item_df = pd.concat([item_df, genre_df], axis=1) #item_df와 gen_df를 옆으로 합친다. (index : item label)


        user_df = pd.DataFrame(columns=user_use_features)
        for item, values in user_dict.items():
            temp = []
            for column, value in values.items():
                temp.append(value[0]) #값이 여러개가 아닌 경우 0번째 값만 넣어줌
            user_df.loc[item] = temp
        
        item_df = item_df.reset_index(names=["item"])
        self.item_df = item_df
        user_df = user_df.reset_index(names=["user"])
        self.user_df = user_df

        # # 인코딩 원본으로 바꿔줌, inverse_transform 처리해주는 코드 
        # item_df["item"] = self.item_encoder.inverse_transform(item_df["item"])
        # user_df["user"] = user_df["user"].astype(int) #user encoding이 소수로 되어 있어서 정수형으로 바꿔줘야 함
        # user_df["user"] = self.user_encoder.inverse_transform(user_df["user"])

        interaction_df = self.interaction_df
        interaction_df = interaction_df.merge(self.item_df,how="inner",on="item").merge(self.user_df,how="inner",on="user")

        interaction_df["rating"] = 1 #본 영화 dataframe 생성 완료!
        

        return item_df, user_df, interaction_df
    
    #Todo : interaction을 통해 item : count dict 생성, kfold 전 후에 사용할 수 있도록 구현, 일단 전 기준..??
    def _make_negative_sampling(self,neg_ratio, threshold, sampling_mode):
        
        #새로 뽑은 negative sample의 user, item df 정보도 붙여줘야 됨.
        if sampling_mode == "popular":
            df = pd.DataFrame(columns=["user", "item"])
            _user = []
            _item = []

            for user, values in tqdm(self.item_popular.items()):
                _df = pd.DataFrame(columns=["user", "item"])

                values = values[:threshold] #인기도 상위 몇까지 자를지
                positive_num = self.user_item_count[user] #해당 유저의 interaction item 개수
                negative_num = int(positive_num * neg_ratio) #negative 개수
                negative_item = np.random.choice(values, negative_num, replace=False) #replace = False : 중복 허용 x
                # _df["user"] = user
                # _df["item"] = negative_item
                # df = pd.concat([df,_df])
                _user = _user + [user] * negative_num
                _item = np.append(_item, negative_item, axis=0)
            df["user"] = _user
            df["item"] = _item
            df["item"] = df["item"].astype(int)
            
            #item, user side information merge
            df = df.merge(self.item_df,how="inner",on="item").merge(self.user_df,how="inner",on="user")
            
            df["rating"] = 0 #interaction : 0

        return df
    
    #Todo : test dataset 생성, 처음부터 user가 보지 않은 영화의 prob를 뽑는다.
    def _make_test_dataset(self):
        df = pd.DataFrame(columns=["user", "item"])
        _user = []
        _item = []

        for user, values in tqdm(self.item_popular.items()):
            _df = pd.DataFrame(columns=["user", "item"])

            _user = _user + [user] * len(values)
            _item = _item + values
        
        df["user"] = _user
        df["item"] = _item
        df["item"] = df["item"].astype(int)
        
        #item, user side information merge
        df = df.merge(self.item_df,how="inner",on="item").merge(self.user_df,how="inner",on="user")
        
        return df

                
            

        

        




        



            



if __name__ == '__main__':
    preprocessor = Preprocessor()
    preprocessor._load_train_dataset()