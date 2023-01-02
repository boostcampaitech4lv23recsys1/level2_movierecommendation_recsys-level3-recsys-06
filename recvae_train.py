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

from copy import deepcopy

from parse_config import ConfigParser
from utils import prepare_device, k_recoommended_movies

from trainer import Trainer
from trainer.ae_trainer import AETrainer

import model.loss as module_loss
from model.loss import loss_function_dae, loss_function_vae
import model.metric as module_metric
import model.model as module_arch
from model.metric import Recall_at_k_batch
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



def ae_train(n_epochs=20, lr=0.0005, dropout_rate=0.6, batch_size=512):
    # 우선 mult_DAE로 해보기
    start_time = time.time()
    n_kfold = 5
    batch_size = batch_size
    n_epochs = n_epochs # 에폭 숫자
    root_data = './data/train/' 
    data_dir = os.path.join(root_data, 'ae_data')
    n_users = 31360
    n_items = 6807
    p_dims = [200, 600, n_items]
    dropout_rate = dropout_rate
    lr = lr

    n_gpu_use = torch.cuda.device_count() 
    device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')

    all_recalls = []
    inference_results = []

    user_label, item_label = get_labels(data_dir)
    
    # inference에서 쓸 rating 마련하기
    ratings = pd.read_csv(root_data+'train_ratings.csv')[['user', 'item']]
    temp_rows, temp_cols = ratings['user'].apply(lambda x : user_label[x]), ratings['item'].apply(lambda x: item_label[x])
    raw_data = sparse.csr_matrix((np.ones_like(temp_rows), (temp_rows, temp_cols)), dtype='float64', shape=(n_users, n_items)).toarray()

    train_mark=raw_data.nonzero()  # 최종 인퍼런스 때 필터링해줄 마스크]
    raw_data = torch.Tensor(raw_data)  # 인퍼런스에 쓰기 위해 Tensor로 바꿔줌
    del ratings

    for i in range(1, n_kfold+1):  # k_fold를 일단 5회로 적어놓기
        best_epoch = 0
        best_score = 0
        print(f'====================Start: {i}-fold for 5 fold====================')
        model = RecVAE(600, 200, 6807).to(device)
        # optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=0.00)
        criterion = loss_function_dae
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0005, weight_decay = 0.01)
        tr_data, te_data = ae_data_load(data_dir, i)
        
        trainset = AETrainDataSet(tr_data)
        validset = AETestDataSet(tr_data, te_data)

        train_loader = AEDataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=1)
        valid_loader = AEDataLoader(validset, batch_size=batch_size, shuffle=False, num_workers=1)

        # decoder_params = set(model.decoder.parameters())
        # encoder_params = set(model.encoder.parameters())
        # optimizer_encoder = optim.Adam(encoder_params, lr=0.0005, weight_decay=0.00)
        # optimizer_decoder = optim.Adam(decoder_params, lr=0.0005, weight_decay=0.00)

        print(f'==========Training Start==========')
        # before_loss_avr = 0.0
        print(f"=================Encoder Training Start =================")
        for epoch in range(1, n_epochs+1):
            epoch_stime = time.time()
            model.train()
            train_loss = 0

            print(f"=================Epoch {epoch} Start =================")
            for step, batch in enumerate(train_loader):
                # (batch, 1, n_items)의 데이터가 들어옴
                batch = batch.squeeze(1).to(device)
                optimizer.zero_grad()
                _, loss = model(user_ratings = batch, beta = None, gamma = 0.0005, dropout_rate = dropout_rate)
                train_loss += loss.item()
                loss.backward()
                optimizer.step()
                
            train_loss /= math.ceil(n_users//batch_size)
                # if step % 50 == 0 and step > 0:
                #     print('| epoch {:3d} | {:4d}/{:4d} batches| '
                #             'loss {:4.2f}'.format(
                #                 epoch, step, len(range(0, n_users, batch_size)),
                #                 train_loss / batch_size))
            
        # model.update_prior()
        # print(f"=================Decoder Training Start =================")
        # for epoch in range(1, 2):
        #     print(f"=================Epoch {epoch} Start =================")
        #     for step, batch in enumerate(train_loader):
        #         # (batch, 1, n_items)의 데이터가 들어옴
        #         batch = batch.squeeze(1).to(device)
        #         optimizer_decoder.zero_grad()
        #         _, loss = model(batch, dropout_rate=0)
                
        #         loss.backward()

        #         optimizer_decoder.step()

            
            # now_loss_avr = all_loss / math.ceil(n_users/batch_size)
            # print(f'total average loss for epoch {epoch} : {now_loss_avr}')
            # if epoch != 1:
            #     improvement = before_loss_avr - now_loss_avr
            #     if improvement > 0:
            #         print(f'{improvement} point better than before.')
            #     else:
            #         print('this time no more improvement!')
            # before_loss_avr = now_loss_avr
            # all_loss = 0.0
            

            # if epoch == n_epochs:
            model.eval()
            X_preds = []  # 배치마다 나오는 예측치를 저장해줄 리스트
            heldouts = []  # 숨겨놓은 아이템을 기록해놓는 테이블 (concat하면 전체 유저의 데이터가 될 것)
            with torch.no_grad():
                for step, batch in tqdm(enumerate(valid_loader)):
                    batch = batch.squeeze(1).to(device)
                    input_batch, heldout_data = torch.split(batch, n_items, dim=1)
                    output_batch = model(input_batch, calculate_loss=False)

                    # Exclude examples from training set
                    input_batch, output_batch, heldout_data = input_batch.cpu().numpy(), output_batch.cpu().numpy(), heldout_data.cpu().numpy()
                    output_batch[input_batch.nonzero()] = -np.inf  # 이미 본 영화는 모두 -np.inf 처리
                
                    X_preds.append(output_batch)
                    heldouts.append(heldout_data)

            X_preds = np.concatenate(X_preds)  # 모든 유저에 대한 예측
            heldouts = np.concatenate(heldouts)  # 모든 유저들의 숨겨놓은 영화 목록 (1)

            recall_epoch = Recall_at_k_batch(X_preds, heldouts, 10)
            if recall_epoch > best_score:
                best_epoch = epoch
                best_score = recall_epoch
            print(f'============Recall for Epoch {epoch} : {recall_epoch}============')
            print(f'============Epoch: {epoch:3d}| Train loss: {train_loss:.5f}======')  
            print(f'epoch에 약 {time.time()-epoch_stime}초 걸렸습니다')
        
        # 마지막 훈련 완료된 recall 사용
        all_recalls.append(recall_epoch)
        # 이 모델을 사용해서 진행한 인퍼런스
        raw_data = raw_data.to(device)  
        inference_result = model(raw_data, calculate_loss=False).cpu().detach().numpy()
        inference_results.append(inference_result)
        
    
    now = datetime.datetime.now()
    time_str = now.strftime("%m_%d_%H_%M_%S")
    total_recall_at_k = round(sum(all_recalls)/len(all_recalls),4)
    print(f'==============최종 recall_at_k는 {total_recall_at_k}입니다===============')
    

    print(f'=======================Starting Inference=======================')
    print("==========이미 본 영화를 필터링해줍니다.==========")
    # 원래 본 영화를 빼주는 필터링 작업
    inference_results = np.array(inference_results)
    inference_results = np.mean(inference_results, axis=0)
    inference_results[train_mark] = -np.inf

    final_10 = bn.argpartition(-inference_results, 10, axis=1)[:, :10]  # 10개만 남겨둠

    label_to_user = {v: k for k, v in user_label.items()}
    label_to_item = {v: k for k, v in item_label.items()}

    if not os.path.exists('./output'):
        os.mkdir('./output')
    
    if not os.path.exists('./output/auto_encoder'):
        os.mkdir('./output/auto_encoder')
    
    output_path = './output/auto_encoder'
        # 예측 파일을 날짜, 최종 recall과 함께 저장함 (=> 5개 폴드의 평균 recall@1k)
    
    with open(output_path + f'/predictions_RecVAE_lr_{lr}_epoch{n_epochs}_{total_recall_at_k}_dropout{dropout_rate}_batch_{batch_size}_bestscore{best_epoch},{best_score}.pkl', "wb") as file:
        pickle.dump(inference_results, file)

    with open(output_path + f'/submission_RecVAE_lr_{lr}__epoch{n_epochs}_{total_recall_at_k}_dropout{dropout_rate}_batch_{batch_size}_bestscore{best_epoch},{best_score}.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
    
        # Write the header row
        writer.writerow(['user', 'item'])
        
        # Write the data rows
        print("Creating submission file: 31360 users")
        for i, row in tqdm(enumerate(final_10)):
            u_n = label_to_user[i]
            for j in row:
                writer.writerow([u_n, label_to_item[j]])

    print("submission saved!")
    print(f'약 {round((time.time()-start_time)/60, 1)}분 걸렸습니다')


if __name__ == "__main__":
    for i in [20]:
        for j in [0.0005]:
            for q in [0.4]:
                ae_train(i, j, q, 64)
    
