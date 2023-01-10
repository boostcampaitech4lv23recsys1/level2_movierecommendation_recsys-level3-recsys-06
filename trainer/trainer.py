import pandas as pd
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import joblib
from torchvision.utils import make_grid
from base import BaseTrainer
from sklearn.model_selection import KFold, train_test_split
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from preprocess.preprocessing import _make_negative_sampling

class GBDTTrainer():
    """
    Trainer class
    GBDTTrainer(config, train_df, valid_df, test_df, item_df, user_df,len(user_df))
    """
    def __init__(self, config, train_df, valid_df, test_df, item_df, user_df, user_num):
        self.config = config
        self.train_df = train_df
        self.valid_df = valid_df
        self.test_df = test_df
        self.item_df = item_df
        self.user_df = user_df
        self.user_num = user_num
        self.fold_predict = []
        
        self.use_features = config['using_features']
        self.cat_features = config["cat_features"]


    def _train_epoch(self, epoch):
        train_df = self.train_df
        valid_df = self.valid_df
        
        train_user = list(train_df["user"])

        #train_user에 있는 user의 negative sampling진행
        print("======negative_sampling====")
        neg_df = _make_negative_sampling(train_user, self.item_df, self.user_df, neg_ratio=0.5, threshold=3800, sampling_mode="popular")

        print("=====positive, negative concat=========")
        train_df = train_df.merge(self.item_df,how="inner",on="item").merge(self.user_df,how="inner",on="user")
        valid_df = valid_df.merge(self.item_df,how="inner",on="item").merge(self.user_df,how="inner",on="user")

        train_df["rating"] = 1 #본 영화 dataframe 생성 완료!
        valid_df["rating"] = 1 #본 영화 dataframe 생성 완료!

        train_df = pd.concat([train_df,neg_df]) #shape:
        
        use_features = self.use_features
        cat_features = self.cat_features

        train_df = train_df[use_features]
        print(train_df.columns)

        print("========training========")
        lgb = LGBMClassifier() #하이퍼파라미터 튜닝, lr : , estimator 1500, early stoping rounds 걸어주고 -> lr 0.05
        lgb.fit(train_df.drop(['rating'], axis=1),  train_df[["rating"]],categorical_feature=cat_features)

        valid_user = list(valid_df["user"])
        recall_user = list(set(valid_user) & set(train_user))
        print(len(set(valid_user)))
        print(len(set(train_user)))
        print(len(set(recall_user)))

        _val_df = valid_df[use_features].drop(["rating"],axis=1).copy()
        _val_predict = lgb.predict_proba(_val_df)
        
        _val_df["prob"] = _val_predict[:,1]
        
        grouped = self.test_df.groupby(["user"])
        recall = 0
        count = 0

        total_predict = []

        for user, group in tqdm(grouped): #user : user명, group: dataframe
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
        print("========recall========")
        print(recall/self.user_num)

        self.fold_predict.append(total_predict)

    def make_csv(self):
        #TODO : (1) fold별 평균 계산 후 (2)유저별 높은 확률 값을 갖는 item 반환
        _s = np.array(self.fold_predict) #shape : (5,2억)
        _s = _s.mean(axis=0)

        self.test_df["total_prob"] = _s

        #(2)
        test_df = self.test_df.sort_values(by = "total_prob", ascending=False)

        top = test_df.groupby(["user"])
        top_10 = top.head(10)

        save_dir = self.config['trainer']['save_dir']
        top_10.sort_values(["user","total_prob"], ascending=[True,False]).to_csv(f"{save_dir}lgb30.csv",index=False)
         
    
INF = int(1e9)

class Trainer(BaseTrainer):
    """
    Trainer class
    """
    def __init__(self, model, criterion, metric_ftns, optimizer, config, device,
                 data_loader, valid_loader, valid_target,
                 fold_num, pos_items_dict, lr_scheduler=None, len_epoch=None):
        super().__init__(model, criterion, metric_ftns, optimizer, config, fold_num)
        self.config = config
        self.device = device
        self.data_loader = data_loader
        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.data_loader)
        else:
            # iteration-based training
            self.data_loader = inf_loop(data_loader)
            self.len_epoch = len_epoch
        self.valid_loader = valid_loader
        self.valid_target = valid_target
        self.do_validation = self.valid_loader is not None
        self.lr_scheduler = lr_scheduler
        self.log_step = int(np.sqrt(data_loader.batch_size))
        self.fold_num = fold_num
        self.pos_items_dict = pos_items_dict

        self.metric = self.metric_ftns[0]


    def _train_epoch(self, epoch):
        """
        Training logic for an epoch
        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        train_loss = np.array([])
        epoch_iterator = tqdm(
        self.data_loader, desc="Training (X / X Steps) (loss=X.X)", dynamic_ncols=True, mininterval = 1
        )
        self.model.train()
        for batch_idx, (data, target) in enumerate(epoch_iterator):
            if self.config['name'] == 'DeepFM':
                data = data.view(-1, data.shape[-1])
                target = target.view(-1, 1).squeeze().float()
            data, target = data.to(self.device), target.to(self.device)

            self.optimizer.zero_grad()
            output = self.model(data)
            if self.config['name'] == 'Bert4Rec':
                output = output.view(-1, output.size(-1)) #[Batch_size, seq_len, 6808] -> [Batch_size * seq_len, 6808]
                target = target.view(-1)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()
            
            current = batch_idx
            total = self.len_epoch
            epoch_iterator.set_description("[FOLD - %s, EPOCH: %s] Training (%d / %d Steps) (loss=%2.5f)" % (self.fold_num, epoch, current, total, loss.item()))
            
            train_loss = np.append(train_loss, loss.detach().cpu().numpy())

        if self.do_validation:
            if self.config['name'] == 'DeepFM':
                val_recallk_score = self._valid_epoch(epoch)
            elif self.config['name'] == 'Bert4Rec':
                val_recallk_score = self._valid_epoch_seq(epoch)
            print(f"[VALIDATION RECALL@K SCORE]: {val_recallk_score}")

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        log = {'train_loss': train_loss.mean(), 'recall': val_recallk_score}
        
        return log


    def _valid_epoch(self, epoch):
        """
        return recall score @k
        Validate after training an epoch
        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.model.eval()
        infer_list = []
        with torch.no_grad():
            for data in tqdm(self.valid_loader):
                data = data.to(self.device)
                output = self.model(data)
                prob = output.detach().cpu().numpy()[:, np.newaxis]
                info = data[:, :2].detach().cpu().numpy()
                infos = np.concatenate([info, prob], axis = 1)
                infer_list.append(infos)
        inference = np.concatenate(infer_list, axis = 0)
        print(inference.shape)
        inference = pd.DataFrame(inference, columns = ['user', 'item', 'prob'])
        inference = inference.sort_values(by = 'prob', ascending = False)

        return self.metric(inference, self.valid_target)

    def _valid_epoch_seq(self, epoch):
        """
        return recall score @k
        Validate after training an epoch
        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        #TODO: BERT4REC
        self.model.eval()
        infer_list = []
        with torch.no_grad():
            for user, tokens in tqdm(self.valid_loader):
                user = user.numpy()
                tokens = tokens.to(self.device)
                output = self.model(tokens)
                #output shape: [Batch_size, max_len, 6808] -> output[:, -1, :]
                output = output[:, -1, :]
                output = F.softmax(output, dim = -1)
                output = output.detach().cpu().numpy()
                for idx in range(self.valid_loader.batch_size):
                    user_num = int(user[idx].item())
                    user_probs = output[idx]
                    infos = []
                    for item_num in range(6808):
                        if item_num == 0:
                            continue
                        if (item_num - 1) in self.pos_items_dict[user_num]:
                            infos.append(np.array([user_num, item_num-1, -INF])[np.newaxis, :])
                        else:
                            infos.append(np.array([user_num, item_num-1, user_probs[item_num]])[np.newaxis, :])
                    temp = np.concatenate(infos, axis = 0)
                    infer_list.append(temp)
        inference = np.concatenate(infer_list, axis = 0)
        inference = pd.DataFrame(inference, columns = ['user', 'item', 'prob'])
        inference = inference.sort_values(by = 'prob', ascending = False)
        grouped = inference.groupby('user')
        top_10 = grouped.head(10)

        return self.metric(top_10, self.valid_target)

    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.data_loader, 'n_samples'):
            current = batch_idx * self.data_loader.batch_size
            total = self.data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)