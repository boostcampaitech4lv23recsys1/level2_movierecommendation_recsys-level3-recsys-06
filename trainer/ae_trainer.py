import numpy as np
import torch
import time
from copy import deepcopy

from torchvision.utils import make_grid
from base import BaseTrainer
from utils import inf_loop, MetricTracker
from model.metric import recall_at_k_batch

import wandb


class AETrainer:
    """
    Trainer class
    """
    def __init__(self, fold, model, optimizer, config, device,
                 train_loader, valid_loader, heldouts, criterion=None, lr_scheduler=None):
        self.model = model
        self.model_name = config['model_name']
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
        self.heldouts = heldouts.toarray()

    
    def train(self):
        last_train_loss = 0.0
        best_score = float('-inf')
        best_epoch = -1
        X_preds_best = None

        for epoch in range(1, self.n_epochs+1):
            epoch_stime = time.time()

            print(f"[Fold {self.fold} | Epoch {epoch}]")
            train_loss = self.train_epoch(epoch)
           
            X_preds = self.valid_epoch(epoch)
            X_preds = np.concatenate(X_preds)  # 모든 유저에 대한 예측
        
            recall_epoch = recall_at_k_batch(X_preds, self.heldouts, 10)
            if recall_epoch > best_score:
                best_epoch = epoch
                torch.save(self.model.state_dict(), f'saved_model/{self.config["model_name"]}/lr{self.config["lr"]}_dropout{self.config["dropout_rate"]}_epoch{self.config["n_epochs"]}_fold{self.fold}.pt')
                best_score = recall_epoch
                X_preds_best = X_preds
                print("[Model Saved] This Epoch is the best at metric so far!")
            
            wandb.log({"epoch": epoch, "recall epoch": recall_epoch, "best epoch": best_epoch, "train loss": train_loss, "best score": best_score})
            print(f'[Recall for Epoch {epoch}] {recall_epoch}')
            print(f'Train loss: {train_loss:.5f} | Train loss Imporved: {round(last_train_loss-train_loss, 4)}')  
            print(f'epoch에 약 {round(time.time()-epoch_stime,1)}초 걸렸습니다', '\n')

            last_train_loss = train_loss
        
        # 지금까지 있었던 가장 좋은 모델을 모델에 불러줍니다.
        if epoch != best_epoch:
            print("[Best Model Loaded]")
            print(f"Best Epoch {best_epoch}")
            self.model = deepcopy(self.model)
            self.model.load_state_dict(torch.load(f'saved_model/{self.config["model_name"]}/lr{self.config["lr"]}_dropout{self.config["dropout_rate"]}_epoch{self.config["n_epochs"]}_fold{self.fold}.pt'))

        return best_score


    def train_epoch(self, epoch):
        self.model.train()
        train_loss = 0

        if self.model_name == "MultiVAE":
            self.update_count = 0

        for step, batch in enumerate(self.train_loader):
            batch = batch.squeeze(1).to(self.device)
            self.optimizer.zero_grad()

            if self.model_name == "MultiVAE":
                anneal = self.model.get_anneal(self.update_count)
                output, mu, logvar = self.model(batch)
                loss = self.criterion(output, batch, mu, logvar, anneal)
                self.update_count += 1

            elif self.model_name == "MultiDAE":
                output = self.model(batch)
                loss = self.criterion(output, batch)

            else:   
                _, loss = self.model(user_ratings = batch, beta = None, gamma = 0.0005, dropout_rate = self.dropout_rate)
            
            train_loss += loss.item()
            loss.backward()
            self.optimizer.step()

        train_loss /= self.n_users
        return train_loss


    def valid_epoch(self, epoch):
        self.model.eval()
        X_preds = []  # 배치마다 나오는 예측치를 저장해줄 리스트
        with torch.no_grad():
            for step, batch in enumerate(self.valid_loader):
                batch = batch.squeeze(1).to(self.device)
                input_batch, _ = torch.split(batch, self.n_items, dim=1)
 
                if self.model_name == "MultiVAE":
                    anneal = self.model.get_anneal(self.update_count)
                    output_batch, mu, logvar = self.model(input_batch)

                elif self.model_name == "MultiDAE":
                    output_batch = self.model(input_batch)
                else: 
                    output_batch = self.model(input_batch, calculate_loss=False)

                # Exclude examples from training set
                input_batch, output_batch = input_batch.cpu().numpy(), output_batch.cpu().numpy()
                output_batch[input_batch.nonzero()] = -np.inf  # 이미 본 영화는 모두 -np.inf 처리
            
                X_preds.append(output_batch)
        
        return X_preds


    def inference(self, raw_data):
        raw_data = raw_data.to(self.device)
        inference_result = self._get_inference(raw_data).cpu().detach().numpy()

        return inference_result


    def _get_inference(self, raw_data):
        if self.model_name == 'MultiVAE':
            result, _, _ = self.model(raw_data)
        elif self.model_name == 'MultiDAE':
            result = self.model(raw_data)
        else:
            result = self.model(raw_data, calculate_loss=False)
        return result


