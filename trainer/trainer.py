import pandas as pd
import numpy as np
import torch
from torchvision.utils import make_grid
from base import BaseTrainer
from tqdm import tqdm

class Trainer(BaseTrainer):
    """
    Trainer class
    """
    def __init__(self, model, criterion, metric_ftns, optimizer, config, device,
                 data_loader, valid_loader, valid_target, 
                 fold_num, lr_scheduler=None, len_epoch=None):
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
            data = data.view(-1, data.shape[-1])
            target = target.view(-1, 1).squeeze().float()
            data, target = data.to(self.device), target.to(self.device)

            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()
            
            current = batch_idx
            total = self.len_epoch
            epoch_iterator.set_description(
            "[FOLD - %s] Training (%d / %d Steps) (loss=%2.5f)" % (self.fold_num, current, total, loss.item())
            )
            
            train_loss = np.append(train_loss, loss.detach().cpu().numpy())

        if self.do_validation:
            val_recallk_score = self._valid_epoch(epoch)
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
        #TODO: BERT4REC
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

    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.data_loader, 'n_samples'):
            current = batch_idx * self.data_loader.batch_size
            total = self.data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)
