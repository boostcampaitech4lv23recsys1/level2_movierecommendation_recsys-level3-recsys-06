import time
import math
import argparse
import datetime
import collections
import os

import torch
import torch.optim as optim

import numpy as np
import bottleneck as bn

import wandb

from parse_config import ConfigParser
from utils import prepare_device
from utils.ae_util import make_prediction_file, make_inference_data_and_mark, write_submission_file, get_loaders

from trainer import Trainer
from trainer.ae_trainer import AETrainer

import model.loss as module_loss
from model.loss import loss_function_dae, loss_function_vae
import model.metric as module_metric
import model.model as module_arch
from model.metric import recall_at_k_batch
from model.model import MultiDAE, MultiVAE, RecVAE, EASE

from data_loader.ae_dataloader import AETrainDataSet, AETestDataSet, ae_data_load, get_labels
import data_loader.data_loaders as module_data
from data_loader.data_loaders import AEDataLoader



# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)


def ae_train(config):
    # wandb.login()

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
    model_saved_path = os.path.join(config['model_saved_path'], model_name)
    n_users = config['n_users']
    n_items = config['n_items']
    p_dims = [200, 600, n_items]

    n_gpu_use = torch.cuda.device_count() 
    device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')

    # 파일을 저장할 디렉토리 설정
    if not os.path.exists(output_path):
        os.mkdir(output_path)

    if not os.path.exists(model_saved_path):
        os.mkdir(model_saved_path)

    all_recalls = []
    inference_results = []

    user_label, item_label = get_labels(data_dir)
    raw_data, train_mark = make_inference_data_and_mark(config, root_data, user_label, item_label)


    for fold in range(1, n_kfold+1):  # k_fold를 일단 5회로 적어놓기
        # wandb.init(project=f"Movie_Rec_{model_name}", config=config)
        # wandb.run.name = f'{model_name}_n_epoch{n_epochs}_lr{lr}_dropout{dropout_rate}_batch{batch_size}_fold{fold}'

        # print(f'====================Start: {fold}-fold for 5 fold====================')
        # if model_name == "MultiVAE":
        #     model = MultiVAE(config, p_dims, dropout=dropout_rate).to(device)
        # elif model_name == "MultiDAE":
        #     model = MultiDAE(config, p_dims, dropout=dropout_rate).to(device)
        # else:
        model = RecVAE(*p_dims, config).to(device)
        model.load_state_dict(torch.load(f'/opt/ml/movie_template/lr0.0005_dropout0.6_epoch250_fold{fold}.pth'))
        # # optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=0.00)
        # if model_name == "MultiDAE":
        #     criterion = loss_function_dae
        # elif model_name == "MultiVAE":
        #     criterion = loss_function_vae
        # else:
        #     criterion=None
            
        # optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay = weight_decay)
        tr_data, te_data = ae_data_load(data_dir, fold)
        # train_loader, valid_loader = get_loaders(tr_data, te_data, config)
        
        # trainer = AETrainer(fold=fold, model=model, optimizer=optimizer, config=config, device=device, train_loader=train_loader, valid_loader=valid_loader, criterion=criterion)

        print(f'Training Start')
        # recall_epoch, X_preds, heldouts = trainer.train()
        
        # 마지막 훈련 완료된 recall 사용
        model.eval()
        X_preds = model(torch.Tensor(tr_data.toarray()).to(device), calculate_loss=False)
        heldouts = te_data.toarray()
        X_preds[tr_data.nonzero()] = -np.inf
        recall_epoch = recall_at_k_batch(X_preds.cpu().detach().numpy(), heldouts, 10)
        all_recalls.append(recall_epoch)

        # 이 모델을 사용해서 인퍼런스 진행 및 리스트에 저장
        inference_result = model(torch.Tensor(raw_data).to(device), calculate_loss=False).cpu().detach().numpy()
        inference_results.append(inference_result)

        # wandb save for each fold
        # wandb.join()
        
    
    total_recall_at_k = round(sum(all_recalls)/len(all_recalls),4)
    
    print(f'==============최종 recall_at_k는 {total_recall_at_k}입니다===============')

    print(f'=======================Starting Inference=======================')
    print("==========이미 본 영화를 필터링해줍니다.==========")
    # 원래 본 영화를 빼주는 필터링 작업
    inference_results = np.array(inference_results)
    inference_results = np.mean(inference_results, axis=0)
    inference_results[train_mark] = -np.inf

    final_10 = bn.argpartition(-inference_results, 10, axis=1)[:, :10]  # 10개만 남겨둠

    
    # 예측 파일을 저장함
    make_prediction_file(output_path, inference_results, config, total_recall_at_k, user_label, item_label)
    
    #제출 파일을 저장함
    write_submission_file(output_path, final_10, config, total_recall_at_k, user_label, item_label)


if __name__ == "__main__":
    config = {
        "n_kfold": 5,
        "n_epochs": 50,
        "dropout_rate": 0.6,
        "lr": 0.001,
        "batch_size": 64,

        "root_data": './data/train/' ,
        "data_dir": './data/train/ae_data',
        "weight_decay": 0.00,
        "num_workers": 4,
        "model_name": 'RecVAE', # [MultiDAE, MultiVAE, RecVAE]
        "output_path": './output/auto_encoder',
        "model_saved_path": './saved_model',

        "n_users": 31360,
        "n_items": 6807,

        'total_anneal_steps': 200000,
        'anneal_cap': 0.2
    }
    ae_train(config)