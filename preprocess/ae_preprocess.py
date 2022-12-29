import argparse
import time
import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from scipy import sparse
import os
import pandas as pd
from scipy import sparse
import numpy as np

def get_count(tp, id):
    playcount_groupbyid = tp[[id]].groupby(id, as_index=False)
    count = playcount_groupbyid.size()

    return count

# 특정한 횟수 이상의 리뷰가 존재하는(사용자의 경우 min_uc 이상, 아이템의 경우 min_sc이상) 
# 데이터만을 추출할 때 사용하는 함수입니다.
# 현재 데이터셋에서는 결과적으로 원본그대로 사용하게 됩니다.
def filter_triplets(tp, min_uc=5, min_sc=0):
    if min_sc > 0:
        itemcount = get_count(tp, 'item')
        tp = tp[tp['item'].isin(itemcount.index[itemcount >= min_sc])]

    if min_uc > 0:
        usercount = get_count(tp, 'user')
        tp = tp[tp['user'].isin(usercount.index[usercount >= min_uc])]

    usercount, itemcount = get_count(tp, 'user'), get_count(tp, 'item')
    return tp, usercount, itemcount

#훈련된 모델을 이용해 검증할 데이터를 분리하는 함수입니다.
#100개의 액션이 있다면, 그중에 test_prop 비율 만큼을 비워두고, 그것을 모델이 예측할 수 있는지를
#확인하기 위함입니다.
def split_train_test_proportion(data, test_prop=0.2):
    data_grouped_by_user = data.groupby('user')
    tr_list = [[] for _ in range(5)]
    te_list = [[] for _ in range(5)]
    data_tr = list()
    data_te = list()
    
    for _, group in data_grouped_by_user:  # group에는 유저 하나의 로그데이터들이 묶여있음
        n_items_u = len(group)  # 유저의 아이템 개수
        
        # 중복되지 않게 5번(K-fold 횟수)만큼 아이템의 인덱스를 분류해줌
        samples = []
        temp_list = list(range(n_items_u))
        np.random.shuffle(temp_list)

        # samples에는 5개의 5등분된 샘플 인덱스가 담겨있게 됨
        for i in range(5):
            start = i * (n_items_u//5)
            end = min(start + (n_items_u//5), n_items_u)
            sample = temp_list[start:end]
            samples.append(sample)

        if n_items_u >= 5:  # 5개 이상인 애들에 대해서만 적용해줌 

            for i in range(5):  # k-폴드가 5개라고 가정한다
                idx = np.zeros(n_items_u, dtype='bool')  # 유저가 갖고 있는 아이템 개수만큼의 차원을 갖는 false 벡터를 만듦

                # valid에서 정답을 검증하는 데 쓸 녀석을 test_prop의 비율만큼 뽑아냄
                idx[samples[i]] = True  # replace=False 이면 중복선택 비허용
                # idx 중에서 랜덤하게 선택된 녀석을 True로 바꿔줌

                tr_list[i].append(group[np.logical_not(idx)]) # false인 녀석만 골라낸 "데이터프레임"이 들어감
                te_list[i].append(group[idx]) # ture 인 녀석만 골라낸 "데이터프레임"이 리스트에 추가
        
        else:
            tr_list[i].append(group)

    breakpoint()
    for i in range(5):
        data_tr.append(pd.concat(tr_list[i]))
        data_te.append(pd.concat(te_list[i]))
        
    return data_tr, data_te  # data_tr과 data_te는 k-fold에서 사용할 데이터들이 모두 담겨있게 되는 셈.


def numerize(tp, profile2id, show2id):
    uid = tp['user'].apply(lambda x: profile2id[x])
    sid = tp['item'].apply(lambda x: show2id[x])
    return pd.DataFrame(data={'uid': uid, 'sid': sid}, columns=['uid', 'sid'])


def ae_preprocess(data_path):
    np.random.seed(6)
    ae_data_path = os.path.join(data_path, 'ae_data')
    
    if not os.path.exists(ae_data_path):
        os.makedirs(ae_data_path)
    
    print("Load and Preprocess Movielens dataset")

    # Load Data
    raw_data = pd.read_csv(os.path.join(data_path, 'train_ratings.csv'), header=0)

    # Filter Data
    train_plays, user_activity, item_popularity = filter_triplets(raw_data, min_uc=5, min_sc=0)

    unique_uid = user_activity.index
    n_users = unique_uid.size #31360

    ##훈련 데이터에 해당하는 아이템들
    #Train에는 전체 데이터를 사용합니다.
    ##아이템 ID
    unique_sid = pd.unique(train_plays['item'])

    show2id = dict((sid, i) for (i, sid) in enumerate(sorted(unique_sid)))  # 영화 라벨링 딕셔너리 생성 (순서대로)
    profile2id = dict((pid, i) for (i, pid) in enumerate(sorted(unique_uid)))  # 유저 라벨링 딕셔너리 생성 (순서대로)


    # 유일한 유저 아이디를 기록해놓음
    with open(os.path.join(ae_data_path, 'unique_sid.txt'), 'w') as f: # 왜 기록하는 걸까? 랜덤한 순서를 간직하기 위해서?
        for sid in unique_sid:
            f.write('%s\n' % sid)

    #Validation과 Test에는 input으로 사용될 tr 데이터와 정답을 확인하기 위한 te 데이터로 분리되었습니다.
    train_dfs, valid_dfs = split_train_test_proportion(train_plays) 


    # numerize는 라벨링 과정
    for i in range(5):
        train_data = numerize(train_dfs[i], profile2id, show2id)
        train_data.to_csv(os.path.join(ae_data_path, f'train_{i+1}.csv'), index=False)

        test_data = numerize(valid_dfs[i], profile2id, show2id)
        test_data.to_csv(os.path.join(ae_data_path, f'test_{i+1}.csv'), index=False)

    print("preprocessing Done!")


if __name__ == "__main__":
    data_path = '../data/train'
    ae_preprocess(data_path)