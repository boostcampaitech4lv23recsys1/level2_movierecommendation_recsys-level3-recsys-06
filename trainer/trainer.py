import numpy as np
import pandas as pd
from tqdm import tqdm
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
        self.user_num = user_num
        
        ####label (rating) 포함 사용할 피쳐
        # self.use_features = ['user', 'item', 'release_year', 'categorized_year_gap5','categorized_year_gap10', 'title', 'main_director','main_writer', 'Action', 'Adventure', 'Animation', 'Children',
        #             'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir',
        #             'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War',
        #             'Western', 'favorite_genre', 'maniatic', "whole_period","first_watch_year","last_watch_year","freq_rating_year","series",'rating']

        ####feature 실험 수정!!#########, n_estimators : 500, learning_rate 0.03
        self.use_features = ['user', 'item', 'release_year', 'title', 'main_director','series','favorite_genre','Action', 'Adventure', 'Animation', 'Children',
                    'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir','Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War','Western',
                    'maniatic',"whole_period","first_watch_year","last_watch_year","freq_rating_year",'rating']

        
        self.cat_features =  ['user', 'item', 'release_year', 'title', 'main_director','series', 'favorite_genre','Action', 'Adventure', 'Animation', 'Children',
                    'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir','Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War','Western',
                    "whole_period","first_watch_year","last_watch_year","freq_rating_year"]

        # self.cat_features = ['user', 'item','release_year','categorized_year_gap5',
        #             'categorized_year_gap10','title', 'main_director','main_writer',"favorite_genre"] #"maniatic 빼고 all", 행별로 어떻게 쪼개나용 
    

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        #index로 나누고,
        kf = KFold(n_splits = 5, shuffle = True, random_state = 42)
        sub_result = [] #fold별 확률 담는 리스트
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
            
            #train_df에 있는 유저 가져오기
            train_user = list(train_df["user"])

            #train_user에 있는 user의 negative sampling진행
            print("======negative_sampling====")
            neg_df = _make_negative_sampling(train_user, self.item_df, self.user_df, neg_ratio=0.8, threshold=3800, sampling_mode="popular")

            print("=====positive, negative concat=========")
            train_df = pd.concat([train_df,neg_df]) #shape:
            
            use_features = self.use_features
            cat_features = self.cat_features

            train_df = train_df[use_features]
            print(train_df.columns)

            print("========training========")
            lgb = LGBMClassifier() #하이퍼파라미터 튜닝, lr : , estimator 1500, early stoping rounds 걸어주고 -> lr 0.05
            lgb.fit(train_df.drop(['rating'], axis=1),  train_df[["rating"]],categorical_feature=cat_features)

            # lgb = CatBoostClassifier()
            # lgb.fit(train_df.drop(['rating'], axis=1),  train_df[["rating"]],cat_features=cat_features)

            valid_user = list(valid_df["user"])
            recall_user = list(set(valid_user) & set(train_user))
            print(len(set(valid_user)))
            print(len(set(train_user)))
            print(len(set(recall_user)))

            _val_df = valid_df[use_features].drop(["rating"],axis=1).copy()
            _val_predict = lgb.predict_proba(_val_df)
            
            _val_df["prob"] = _val_predict[:,1]
           
            total_predict = [] #유저 X neg item 확률 값을 리스트로 넣는다. [0.9,0.2, ...] len : 2억
            grouped = self.test_df.groupby(["user"])
            recall = 0
            count = 0

            for user, group in tqdm(grouped): #name : user명, group: dataframe
                # group["user"] = np.array([name] * len(group))
                group = group.merge(self.item_df,how="left",on="item").merge(self.user_df,how="left",on="user")
                group = group[use_features[:-1]]
                predict = lgb.predict_proba(group)
                total_predict.extend(predict[:,1]) #유저의 negative item 확률 유저별 다름, (6493,)

                #negative+valid k(valid len)개 추출, valid가 얼마나 포함되는지 확인
                group["prob"] = predict[:,1]
                temp = _val_df[_val_df["user"]==user]

                k = len(temp)
                total = pd.concat([group, temp])
                total_output = total.sort_values(by="prob",ascending=False)[:k] #recall 계산을 위한 ranking
                nrecall = len(set(total_output["item"]) & set(temp["item"]))

                if k == 0:
                    count += 1
                    continue

                _recall = nrecall/k
                recall += _recall
                # print(f"{user} recall : ",_recall)
            print("========recall========")
            print(recall/self.user_num)
            
            fpredict = np.array(total_predict) #shape : (208313049,)
            np.save(f"/opt/ml/level2_movierecommendation_recsys-level3-recsys-06/saved/fold_predict/fold_{idx}.npy", fpredict)

            print(f"==========={idx} 종료=======")
            
        sub_result.append(total_predict) #total_predict : user X negitem : predict [[],[],[],[],[]] (5,(user X user neg item),1)

        #TODO : (1) fold별 평균 계산 후 (2)유저별 높은 확률 값을 갖는 item 반환
        _s = np.array(sub_result) #shape : (5,2억)
        _s = _s.mean(axis=0)

        _test_df = self.test_df.copy()
        _test_df["total_prob"] = _s

        #(2)
        _test_df = _test_df.sort_values(by = "total_prob", ascending=False)


        top = _test_df.groupby(["user"])
        top_10 = top.head(10)
        top_10.to_csv("lgbm_10.csv",index=False)

        top_20 = top.head(20)
        top_20.to_csv("lgbm_20.csv",index=False)

        top_30 = top.head(30)
        top_30.to_csv("lgbm_30.csv",index=False)
        # breakpoint()
        
    






            

            





            




                