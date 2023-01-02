import numpy as np
import pandas as pd
import torch
from torchvision.utils import make_grid
from base import BaseTrainer
from utils import inf_loop, MetricTracker
from sklearn.model_selection import KFold, train_test_split
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from preprocess.preprocess import _make_negative_sampling

class GBDTTrainer():
    """
    Trainer class
    """
    def __init__(self, config, interaction_df, test_df, item_df, user_df, user_num):
        self.config = config
        self.probs = np.zeros((5,user_num))
        self.interaction_df = interaction_df
        self.test_df = test_df
        self.item_df = item_df
        self.user_df = user_df
    

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        #index로 나누고,
        kf = KFold(n_splits = 5, shuffle = True, random_state = 42)
        for idx, (train_index, valid_index) in enumerate(kf.split(self.interaction_df.index)):
            #negative sampling을 fold별로 다르게 뽑히게

            print(f"[FOLD: {idx + 1}] catboost")

            train_df = self.interaction_df.iloc[train_index].reset_index(drop = True)
            valid_df = self.interaction_df.iloc[valid_index].reset_index(drop = True)


            print(f"[BEFORE CONCAT SHAPE] {train_df.shape}, {valid_df.shape}")

            valid_grouped = valid_df.groupby("user")
            valid_for_train_idx_list = set(valid_df.index)
            valid_for_test_idx_list = []
            
            for name, group in valid_grouped:
                if len(group) > 10:
                    indices = np.random.choice(group.index, 10, replace = False)
                    valid_for_test_idx_list.extend(list(indices))
                else:
                    indices = group.index
                    valid_for_test_idx_list.extend(list(indices))
            
            valid_for_test_idx_set = set(valid_for_test_idx_list)
            valid_for_train_idx_list = list(valid_for_train_idx_list - valid_for_test_idx_set)

            valid_for_train = valid_df.iloc[valid_for_train_idx_list]
            valid_df = valid_df.iloc[valid_for_test_idx_list]

            train_df = pd.concat([train_df, valid_for_train]).reset_index(drop = True)
            valid_df = valid_df.reset_index(drop = True)

            ##negative sampling
            #TODO : train 시점에서 valid_df에 포함된 Item도 negative sampling에 추가가 되야 됨

            #0. interaction_df(interaction이 1 인 df)에서 인덱스 기준으로 8:2 나눠 -> train_df에 없는 user가 있다면 
            #1. negative sampling을 뽑고 train_df에 합쳐
            #2. validation set이랑 위에서 뽑은 negative sampling 합쳐서 fold별로 recall 계산해
            #3. test_df로 predict_proba로 inference -> fold 별로 predict_proba 평균내

            #train_df에 있는 유저 가져오기
            train_user = list(train_df["user"])
            neg_df = _make_negative_sampling(train_user, self.item_df, self.user_df,neg_ratio=1.2, threshold=3800, sampling_mode="popular") #train_user에 있는 user의 negative sampling진행
            
            #3.
            train_df = pd.concat([train_df,neg_df])
            print("=====concat 완료=========")

            #사용할 feature 빼내기
            # use_features = ['user', 'item', 'release_year', 'categorized_year_gap5',
            #         'categorized_year_gap10', 'title', 'director', 'main_director',
            #         'writer', 'main_writer', 'Action', 'Adventure', 'Animation', 'Children',
            #         'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir',
            #         'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War',
            #         'Western', 'favorite_genre', 'maniatic', 'rating']
            use_features = ['user', 'item', 'release_year', 'categorized_year_gap5',
                    'categorized_year_gap10', 'title', 'main_director','main_writer', 'Action', 'Adventure', 'Animation', 'Children',
                    'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir',
                    'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War',
                    'Western', 'favorite_genre', 'maniatic', 'rating']
            cat_features = ['user', 'item', 'release_year', 'categorized_year_gap5',
                    'categorized_year_gap10', 'title', 'main_director','main_writer', 'Action', 'Adventure', 'Animation', 'Children',
                    'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir',
                    'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War',
                    'Western', 'favorite_genre']

            train_df = train_df[use_features]

            # cl = CatBoostClassifier()
            # cl.fit(train_df.drop(['rating'], axis=1),  train_df[["rating"]],cat_features=cat_features)
            cl = LGBMClassifier()
            cl.fit(train_df.drop(['rating'], axis=1),  train_df[["rating"]],categorical_feature=cat_features)
            
            use_features.remove("rating")
            test_df = self.test_df[use_features]
            print("======predict=========")
            predict = cl.predict_proba(test_df)
            print(predict)
            breakpoint()

            #TODO : Recall 계산
            test_df["prob"] = predict[:,1]
            valid_user = list(valid_df["user"])
            _test_df = test_df[test_df["user"].isin(valid_user)]


            #validset도 예측
            val_predict = cl.predict_proba(valid_df[use_features])
            _val_df = valid_df[use_features].copy()
            _val_df["prob"] = val_predict[:,1]

            for user in valid_user:
                val_item = list(_val_df["item"]) #validation에 있는 itemset
                _test_df = _test_df[_test_df["user"]==user]
                _val_df = _val_df[_val_df["user"]==user]
                _total = pd.concat([_test_df, _val_df])
                total_output = _total[_total["user"]==user].sort_values(by="prob",ascending=False)[:10]
                print(len(set(total_output["item"]) & set(val_item)))
                #TODO : recall 계산

            #최종 예측
            
                

            






            # recall = 
            # TODO : recall 어떻게 계산하는지 분석하기, kfold 잘 돌아가는지 확인하기, predict 유저별로 뽑는거
            # 문제 : user * item(negative) datafrmae memory error, catboost memory error
            # user별로 순회하면서 predict sort_values, ##output = test_df[test_df["user"]==0].sort_values(by="prob",ascending=False)[:10]
            

            





            




                