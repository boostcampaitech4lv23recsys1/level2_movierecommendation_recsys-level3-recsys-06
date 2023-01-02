import numpy as np
import torch
import time
import math
from copy import deepcopy

from torchvision.utils import make_grid
from base import BaseTrainer
from utils import inf_loop, MetricTracker
from model.metric import recall_at_k_batch



class AETrainer:
    """
    Trainer class
    """
    def __init__(self, fold, model, optimizer, config, device,
                 train_loader, valid_loader, criterion=None, lr_scheduler=None):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.config = config
        self.device = device
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.n_epochs = config['n_epochs']
        self.lr_scheduler = lr_scheduler
        self.dropout_rate = config['dropout_rate']
        self.n_users = config['n_users']
        self.n_items = config['n_items']
        self.batch_size = config['n_items']
        self.fold = fold

    
    def train(self):
        last_train_loss = 0.0
        best_score = float('-inf')

        for epoch in range(1, self.n_epochs+1):
            epoch_stime = time.time()

            print(f"[Fold {self.fold} | Epoch {epoch}]")
            train_loss = self.train_epoch(epoch)
            X_preds, heldouts = self.valid_epoch(epoch)

            X_preds = np.concatenate(X_preds)  # 모든 유저에 대한 예측
            heldouts = np.concatenate(heldouts)  # 모든 유저들의 숨겨놓은 영화 목록 (1)

            recall_epoch = recall_at_k_batch(X_preds, heldouts, 10)
            if recall_epoch > best_score:
                torch.save(self.model.state_dict(), f'saved_model/{self.config["model_name"]}/lr{self.config["lr"]}_dropout{self.config["dropout_rate"]}_epoch{self.config["n_epochs"]}.pth')
                best_score = recall_epoch
                print("[Model Saved] This Epoch is the best at metric so far!")

            print(f'[Recall for Epoch {epoch}] {recall_epoch}')
            print(f'Train loss: {train_loss:.5f} | Train loss Imporved: {round(last_train_loss-train_loss, 4)}')  
            print(f'epoch에 약 {round(time.time()-epoch_stime,1)}초 걸렸습니다', '\n')

            last_train_loss = train_loss
        
        # 지금까지 있었던 가장 좋은 모델을 모델에 불러줍니다.
        # if recall_epoch != best_score:
        print("[Best Model Loaded]")
        self.model = deepcopy(self.model)
        self.model.load_state_dict(torch.load(f'saved_model/{self.config["model_name"]}/lr{self.config["lr"]}_dropout{self.config["dropout_rate"]}_epoch{self.config["n_epochs"]}.pth'))

        return recall_epoch, X_preds, heldouts


    def train_epoch(self, epoch):
        self.model.train()
        train_loss = 0
        for step, batch in enumerate(self.train_loader):
            batch = batch.squeeze(1).to(self.device)
            self.optimizer.zero_grad()
            _, loss = self.model(user_ratings = batch, beta = None, gamma = 0.0005, dropout_rate = self.dropout_rate)
            train_loss += loss.item()
            loss.backward()
            self.optimizer.step()

        train_loss /= math.ceil(self.n_users//self.batch_size)
        return train_loss


    def valid_epoch(self, epoch):
        self.model.eval()
        X_preds = []  # 배치마다 나오는 예측치를 저장해줄 리스트
        heldouts = []  # 숨겨놓은 아이템을 기록해놓는 테이블 (concat하면 전체 유저의 데이터가 될 것)
        with torch.no_grad():
            for step, batch in enumerate(self.valid_loader):
                batch = batch.squeeze(1).to(self.device)
                input_batch, heldout_data = torch.split(batch, self.n_items, dim=1)
                output_batch = self.model(input_batch, calculate_loss=False)

                # Exclude examples from training set
                input_batch, output_batch, heldout_data = input_batch.cpu().numpy(), output_batch.cpu().numpy(), heldout_data.cpu().numpy()
                output_batch[input_batch.nonzero()] = -np.inf  # 이미 본 영화는 모두 -np.inf 처리
            
                X_preds.append(output_batch)
                heldouts.append(heldout_data)
        
        return X_preds, heldouts


    def inference(self, raw_data):
        raw_data = raw_data.to(self.device)
        inference_result = self.model(raw_data, calculate_loss=False).cpu().detach().numpy()

        return inference_result