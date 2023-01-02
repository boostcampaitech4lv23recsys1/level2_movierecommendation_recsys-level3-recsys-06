import datetime
import time
import math
import argparse
import collections
import os
import pickle
import csv
from tqdm import tqdm

import torch
import torch.optim as optim

import numpy as np
import bottleneck as bn
import pandas as pd
from scipy import sparse

from parse_config import ConfigParser
from utils import prepare_device

from trainer import Trainer
from trainer.ae_trainer import AETrainer

import model.loss as module_loss
from model.loss import loss_function_dae, loss_function_vae
import model.metric as module_metric
import model.model as module_arch
from model.metric import recall_at_k_batch
from model.model import MultiDAE, MultiVAE, RecVAE

from data_loader.ae_dataloader import AETrainDataSet, AETestDataSet, ae_data_load, get_labels
import data_loader.data_loaders as module_data
from data_loader.data_loaders import AEDataLoader



# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)


def make_prediction_file(output_path, inference_results, config, total_recall_at_k, user_label, item_label):
    model_name, lr, n_epochs, dropout_rate, batch_size = config['model_name'], config['lr'], config['n_epochs'], config['dropout_rate'], config['batch_size']

    with open(output_path + f'/mat_{RecVAE}_{round(lr,4)}_epoch{n_epochs}_{total_recall_at_k}_dropout{dropout_rate}_batch_{batch_size}.pkl', "wb") as file:
        pickle.dump(inference_results, file)


def make_inference_data_and_mark(config, root_data, user_label, item_label):
     # inference에서 쓸 rating 마련하기
    n_users, n_items = config['n_users'], config['n_items'] 

    ratings = pd.read_csv(root_data+'train_ratings.csv')[['user', 'item']]
    temp_rows, temp_cols = ratings['user'].apply(lambda x : user_label[x]), ratings['item'].apply(lambda x: item_label[x])
    raw_data = sparse.csr_matrix((np.ones_like(temp_rows), (temp_rows, temp_cols)), dtype='float64', shape=(n_users, n_items)).toarray()
    train_mark=raw_data.nonzero()  # 최종 인퍼런스 때 필터링해줄 마스크]

    return torch.Tensor(raw_data), train_mark  # 인퍼런스에 쓰기 위해 Tensor로 바꿔줌


def write_submission_file(output_path, final_10, config, total_recall_at_k, user_label, item_label):
    model_name, lr, n_epochs, dropout_rate, batch_size = config['model_name'], config['lr'], config['n_epochs'], config['dropout_rate'], config['batch_size']

    label_to_user = {v: k for k, v in user_label.items()}
    label_to_item = {v: k for k, v in item_label.items()}

    with open(output_path + f'/sub_{model_name}_{lr}__epoch{n_epochs}_{total_recall_at_k}_dropout{dropout_rate}_batch_{batch_size}.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
    
        # Write the header row
        writer.writerow(['user', 'item'])
        
        # Write the data rows
        print("Creating submission file: 31360 users")
        for i, row in tqdm(enumerate(final_10)):
            u_n = label_to_user[i]
            for j in row:
                writer.writerow([u_n, label_to_item[j]])


def get_set_and_loader(tr_data, te_data, config):
    batch_size, num_workers = config['batch_size'], config['num_workers']
    
    trainset = AETrainDataSet(tr_data)
    validset = AETestDataSet(tr_data, te_data)

    train_loader = AEDataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    valid_loader = AEDataLoader(validset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return trainset, validset, train_loader, valid_loader



def ae_train(config):
    start_time = time.time()
    n_kfold = config['n_kfold']
    n_epochs = config['n_epochs']
    dropout_rate = config['dropout_rate']
    lr = config['lr']
    batch_size = config['batch_size']
    root_data = config['root_data']
    data_dir = config['data_dir']
    weight_decay = config['weight_decay']
    num_workers = config['num_workers']
    model_name = config['model_name']
    output_path = config['output_path']

    n_users = config['n_users']
    n_items = config['n_items']

    p_dims = [200, 600, n_items]

    n_gpu_use = torch.cuda.device_count() 
    device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')

    all_recalls = []
    inference_results = []

    user_label, item_label = get_labels(data_dir)
    raw_data, train_mark = make_inference_data_and_mark(config, root_data, user_label, item_label)


    for fold in range(1, n_kfold+1):  # k_fold를 일단 5회로 적어놓기
        print(f'====================Start: {fold}-fold for 5 fold====================')
        model = RecVAE(600, 200, 6807).to(device)
        # optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=0.00)
        criterion = loss_function_dae
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay = weight_decay)
        tr_data, te_data = ae_data_load(data_dir, fold)
        trainset, validset, train_loader, valid_loader = get_set_and_loader(tr_data, te_data, config)
        
        trainer = AETrainer(model=model, optimizer=optimizer, config=config, device=device, train_loader=train_loader, valid_loader=valid_loader, criterion=criterion)

        print(f'==========Training Start==========')
        print(f"=================Encoder Training Start =================")
        for epoch in range(1, n_epochs+1):
            recall_epoch, X_preds, heldouts = trainer.train()
        
        # 마지막 훈련 완료된 recall 사용
        all_recalls.append(recall_epoch)

        # 이 모델을 사용해서 인퍼런스 진행 및 리스트에 저장
        inference_result = trainer.inference(raw_data)
        inference_results.append(inference_result)
        
    
    total_recall_at_k = round(sum(all_recalls)/len(all_recalls),4)
    print(f'==============최종 recall_at_k는 {total_recall_at_k}입니다===============')

    print(f'=======================Starting Inference=======================')
    print("==========이미 본 영화를 필터링해줍니다.==========")
    # 원래 본 영화를 빼주는 필터링 작업
    inference_results = np.array(inference_results)
    inference_results = np.mean(inference_results, axis=0)
    inference_results[train_mark] = -np.inf

    final_10 = bn.argpartition(-inference_results, 10, axis=1)[:, :10]  # 10개만 남겨둠


    # 파일을 저장할 디렉토리 설정
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    
    # 예측 파일을 저장함
    make_prediction_file(output_path, inference_results, config, total_recall_at_k, user_label, item_label)
    
    #제출 파일을 저장함
    write_submission_file(output_path, final_10, config, total_recall_at_k, user_label, item_label)





if __name__ == "__main__":
    config = {
        "n_kfold": 1,
        "n_epochs": 1,
        "dropout_rate": 0.5,
        "lr": 0.0005,
        "batch_size": 64,

        "root_data": './data/train/' ,
        "data_dir": './data/train/ae_data',
        "weight_decay": 0.01,
        "num_workers": 1,
        "model_name": 'RecVAE', # [MultiDAE, MultiVAE, RecVAE]
        "output_path": './output/auto_encoder',

        "n_users": 31360,
        "n_items": 6807,

        'anneal_cap': 0.2,
        'total_anneal_step': 200000
    }
    ae_train(config)
    
