import time
import os
from tqdm import tqdm

import torch
from scipy import sparse

import numpy as np
import pandas as pd
import bottleneck as bn

from parse_config import ConfigParser
from utils import prepare_device
from utils.ae_util import make_prediction_file, make_inference_data_and_mark, write_submission_file, get_loaders

from model.model import EASE

from data_loader.ae_dataloader import AETrainDataSet, AETestDataSet, ae_data_load, get_labels


# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)


def EASE_train_inference(config):
    print('========================Data Loading========================')
    root_data = config['root_data']
    data_dir = config['data_dir']
    model_name = config['model_name']
    output_path = config['output_path']
    n_users = config['n_users']
    n_items = config['n_items']
    EASE_lambda = config['EASE_lambda']

    n_gpu_use = torch.cuda.device_count() 
    device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')

    # 파일을 저장할 디렉토리 설정
    if not os.path.exists(output_path):
        os.mkdir(output_path)

    user_label, item_label = get_labels(data_dir)
    raw_data, train_mark = make_inference_data_and_mark(config, root_data, user_label, item_label)
    rating_data = pd.read_csv('./data/train/train_ratings.csv').drop(columns=['time'])
    rating_data['user'] = rating_data['user'].apply(lambda x: user_label[x])
    rating_data['item'] = rating_data['item'].apply(lambda x: item_label[x])
    inference_results = np.zeros((31360, 6807))

    print('========================Training Start========================')
    for i in range(1, 1001):
        print(f'================{i}번쨰 모델 학습 및 인퍼런스================')
        start_time = time.time()

        data_tr, data_te = split_train_test_proportion(rating_data)
        data_tr, data_te = ae_data_load(data_tr, data_te)
        
        data_tr, data_te = data_tr.toarray(), data_te.toarray()
        model = EASE(EASE_lambda)
        model.train(data_tr)
        inference_result = model.forward(data_tr)
        temp_nonzero = data_tr.nonzero()
        inference_result[temp_nonzero]=-np.inf
        recall = recall_at_k_batch(inference_result, data_te, 10)
        print(f'[================{i}번째 모델 recall] {recall}================')
        print(f'약 {round((time.time() - start_time)/60, 1)}분 걸렸습니다')

        inference_results += inference_result


    inference_results[train_mark]=-np.inf
    final_10 = bn.argpartition(-inference_results, 10, axis=1)[:, :10]  # 10개만 남겨둠

    total_recall_at_k = 'bagging1000_0_8'
    # 예측 파일을 저장함
    make_prediction_file(output_path, inference_results, config, total_recall_at_k, user_label, item_label)
    
    #제출 파일을 저장함
    write_submission_file(output_path, final_10, config, total_recall_at_k, user_label, item_label)




def recall_at_k_batch(X_pred, heldout_batch, k=10):
    batch_users = X_pred.shape[0]

    idx = bn.argpartition(-X_pred, k, axis=1)  # 큰 요소 k개의 인덱스가 앞에 오도록 해서 인덱스를 뽑아냄
    X_pred_binary = np.zeros_like(X_pred, dtype=bool)  # (500, 8607) 모양의 0으로 채워진 배열 만듦
    X_pred_binary[np.arange(batch_users)[:, np.newaxis], idx[:, :k]] = True  # k개를 true로 바꿔줌

    X_true_binary = (heldout_batch > 0)  # 바깥에 뺴놓은 데이터에 대해서만 True가 매겨짐
    tmp = (np.logical_and(X_true_binary, X_pred_binary).sum(axis=1)).astype(np.float32)  # 둘 다 and인 것을 골라내서 갯수를 세어줌
    
    n_hidden = X_true_binary.sum(axis=1)
    n_hidden[n_hidden==0] = 1  # 0인 값을 충분히 작은 숫자로 바꿔줌 (n_hidden이 0이면 어차피 분자는 0)
    recall = tmp / np.minimum(k, n_hidden) 
    recall = recall.mean(axis=0).item()
    print(f'recall: {recall}')
    
    return recall


def get_count(tp, id):
    playcount_groupbyid = tp[[id]].groupby(id, as_index=False)
    count = playcount_groupbyid.size()

    return count

def filter_triplets(tp, min_uc=5, min_sc=0):
    usercount, itemcount = get_count(tp, 'user'), get_count(tp, 'item')
    return tp, usercount, itemcount

def split_train_test_proportion(data, test_prop=0.2):
    data_grouped_by_user = data.groupby('user')
    tr_list, te_list = list(), list()

    print('==================유저 데이터 준비==================')
    for _, group in tqdm(data_grouped_by_user):
        n_items_u = len(group)
        
        idx = np.zeros(n_items_u, dtype='bool')
        idx[np.random.choice(n_items_u, size=int(test_prop * n_items_u), replace=False).astype('int64')] = True

        tr_list.append(group[np.logical_not(idx)])
        te_list.append(group[idx])
        
    data_tr = pd.concat(tr_list)
    data_te = pd.concat(te_list)

    return data_tr, data_te

def ae_data_load(data_tr, data_te):

    n_users = 31360
    n_items = 6807

    rows_tr, cols_tr = data_tr['user'], data_tr['item'] 
    rows_te, cols_te = data_te['user'], data_te['item']

    train_data = sparse.csr_matrix((np.ones_like(rows_tr),
                                (rows_tr, cols_tr)), dtype='float64', shape=(n_users, n_items))
    test_data = sparse.csr_matrix((np.ones_like(rows_te),
                                (rows_te, cols_te)), dtype='float64', shape=(n_users, n_items))

    return train_data, test_data




if __name__ == "__main__":
    config = {
        "root_data": './data/train/' ,
        "data_dir": './data/train/ae_data',
        "num_workers": 1,
        "model_name": 'EASE',
        "output_path": './output/auto_encoder',

        "n_users": 31360,
        "n_items": 6807,

        'EASE_lambda': 400
    }
    EASE_train_inference(config)
    
