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

        item_popular_path = os.path.join(self.asset_dir, "item_popular.pkl")
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
        ###################features 실험 수정############################
        # item_use_features = ["release_year", "categorized_year_gap5", "categorized_year_gap10", "title","director","main_director","writer","main_writer","genre"]
        # item_use_features = ['release_year', 'categorized_year_gap5', 'categorized_year_gap10', 'title', 'director', 'main_director', 'writer', 'main_writer', 'genre', 'series','director_genre']
        item_use_features = ['release_year', 'categorized_year_gap5', 'categorized_year_gap10', 'title', 'director', 'main_director', 'writer', 'main_writer', 'genre', 'series', 'director_genre']
        #multi class 사용시 아래 변수 사용
        item_multi_features = ["director", "writer","genre"]
        item_df = pd.DataFrame(columns=item_use_features)

        user_use_features = ['favorite_genre', 'maniatic', 'duration', 'whole_period', 'first_watch_year', 'last_watch_year', 'freq_rating_year']
        numeric_features = ["maniatic"]
 

        for item, values in item_dict.items():
            temp = []
            if item in ["main_director", "movie", "director_genre"]:
                continue
            else:
                for column, value in values.items():
                    if column in item_multi_features:
                        temp.append(value)
                    else:
                        temp.append(int(value[0])) #값이 여러개가 아닌 경우 0번째 값만 넣어줌, 정수형으로 바꿔줘야 함
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
                if column in numeric_features:
                    temp.append(value[0]) #값이 여러개가 아닌 경우 0번째 값만 넣어줌
                else:
                    v = int(value[0])
                    temp.append(v)
            user_df.loc[int(item)] = temp

        #hard coding
        user_df["favorite_genre"] = user_df["favorite_genre"].astype(int)

        item_df = item_df.reset_index(names=["item"])
        self.item_df = item_df
        user_df = user_df.reset_index(names=["user"])
        self.user_df = user_df
        
        return item_df, user_df
    
    #Todo : interaction을 통해 item : count dict 생성, kfold 전 후에 사용할 수 있도록 구현, 일단 전 기준..??
    
    
    #Todo : test dataset 생성, 처음부터 user가 보지 않은 영화의 prob를 뽑는다.
    def _make_test_dataset(self):
        df = pd.DataFrame(columns=["user", "item"])
        _user = []
        _item = []

        for user, values in tqdm(self.item_popular.items()): #안본 아이템 인기순으로 정렬된 피클 파일 순회
            _user.extend([user] * len(values))
            values = list(map(int,values))
            _item.extend(values)
        
        print("extend후")
        print(_user[:10])
        print(_item[:10])
        df["user"] = np.array(_user)
        df["item"] = np.array(_item)


        #item, user side information merge
        #itemdf, userdf 먼저 만들고 
        # df = df.merge(self.item_df,how="left",on="item").merge(self.user_df,how="left",on="user")
        # 
        # user별로 grouped하고
        
        #df : [user, item] -> user * item 안본 개수 만큼
        return df

def _make_negative_sampling(train_user, item_df, user_df, neg_ratio, threshold, sampling_mode): #need: item_popular
    asset_dir = "/opt/ml/level2_movierecommendation_recsys-level3-recsys-06/saved/asset"
    item_popular_path = os.path.join(asset_dir, "item_popular.pkl")
    with open(item_popular_path, 'rb') as f:
        item_popular = pickle.load(f)
        
    user_item_count_path = os.path.join(asset_dir, "user_item_count.pkl")
    with open(user_item_count_path, 'rb') as f:
        user_item_count = pickle.load(f)    

    #새로 뽑은 negative sample의 user, item df 정보도 붙여줘야 됨.
    if sampling_mode == "popular":
        df = pd.DataFrame(columns=["user", "item"])
        _user = []
        _item = []

        for user, values in tqdm(item_popular.items()):
            _df = pd.DataFrame(columns=["user", "item"])

            values = values[:threshold] #인기도 상위 몇까지 자를지
            positive_num = user_item_count[user] #해당 유저의 interaction item 개수
            negative_num = int(positive_num * neg_ratio) #negative 개수
            # print("values len :", len(values))
            # print("positive_num:",positive_num)
            # print("negative_num:", negative_num)
            # print("user", user)
            negative_item = np.random.choice(values, negative_num, replace=False) #replace = False : 중복 허용 x

            
            _user.extend([user] * negative_num)
            _item = np.append(_item, negative_item, axis=0)
        df["user"] = _user
        df["item"] = _item
        df["item"] = df["item"].astype(int)

        df = df[df["user"].isin(train_user)]
        
        #item, user side information merge
        df = df.merge(item_df,how="inner",on="item").merge(user_df,how="inner",on="user")
        
        df["rating"] = 0 #interaction : 0

    return df
                
        


if __name__ == '__main__':
    preprocessor = Preprocessor()
    preprocessor._load_train_dataset()