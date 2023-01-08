import os
import argparse
import collections
import numpy as np
from tqdm import tqdm
import torch
from pathlib import Path
import pickle
import pandas as pd


import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
from data_loader.context_data_loader import StaticDataset, StaticTestDataset
from data_loader.sequential_data_loader import SeqTrainDataset, SeqTestDataset
from torch.utils.data import DataLoader

from parse_config import ConfigParser
from trainer import Trainer
from utils import prepare_device
from sklearn.model_selection import KFold
from preprocess.preprocess import Preprocessor



# fix random seeds for reproducibility
SEED = 42
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)

def main(config):
    asset_dir = "/opt/ml/level2_movierecommendation_recsys-level3-recsys-06/saved/asset"
    """
    load pre-processed DataFrame
    """
    preprocessor = Preprocessor()
    interaction_df, title_df = preprocessor._preprocess_dataset()
    all_items = sorted(list(title_df['item'].unique()))

    logger = config.get_logger('train')

    with open(os.path.join(asset_dir, "item_dict.pkl"), 'rb') as f:
        item_dict = pickle.load(f)
    with open(os.path.join(asset_dir, "user_dict.pkl"), 'rb') as f:
        user_dict = pickle.load(f)

    with open(os.path.join(asset_dir, "item_popular.pkl"), 'rb') as f:
        neg_populars_dict = pickle.load(f)
    """
    popular top 200
    """
    for user in neg_populars_dict.keys():
        neg_populars_dict[user] = neg_populars_dict[user][:200]
        
    
    #TODO KFOLD Validation
    total_index = np.arange(len(interaction_df))
    neg_val_dict = collections.defaultdict(set)
    kf = KFold(n_splits = config['n_fold'], shuffle = True, random_state = SEED)
    for idx, (train_index, valid_index) in enumerate(kf.split(total_index)):
        train_df = interaction_df.iloc[train_index].reset_index(drop = True)
        valid_df = interaction_df.iloc[valid_index].reset_index(drop = True)

        print(f"[BEFORE CONCAT SHAPE] {train_df.shape}, {valid_df.shape}")

        valid_grouped = valid_df.groupby("user")
        valid_for_train_idx_list = set(valid_df.index)
        valid_for_test_idx_list = []
        for name, group in valid_grouped:
            if len(group) > 10:
                indices = np.random.choice(group.index, 10, replace = False)
            else:
                indices = group.index
            valid_for_test_idx_list.extend(list(indices))
        
        valid_for_test_idx_set = set(valid_for_test_idx_list)
        valid_for_train_idx_list = list(valid_for_train_idx_list - valid_for_test_idx_set)

        valid_for_train = valid_df.iloc[valid_for_train_idx_list]
        valid_df = valid_df.iloc[valid_for_test_idx_list]

        train_df = pd.concat([train_df, valid_for_train]).reset_index(drop = True)
        valid_df = valid_df.reset_index(drop = True)

        """
        calculate item_set for neg_val_dict
        """
        grouped = valid_df.groupby('user')
        for name, group in grouped:
            neg_val_dict[name].update(set(group['item']))

        valid_grouped2 = valid_df.groupby('user')
        cnt = 0
        for name, group in valid_grouped2:
            cnt += 1
        print(f"[CNT]: {cnt}")
        print(f"[AFTER CONCAT SHAPE] {train_df.shape}, {valid_df.shape}")

        pos_items_dict = collections.defaultdict(set)
        neg_items_dict = collections.defaultdict(set)
        neg_items_dict_for_valid = collections.defaultdict(set)

        grouped = train_df.groupby('user')
        
        for name, group in tqdm(grouped):
            pos_items_dict[name].update(set(list(group['item'])))

        for user in tqdm(train_df['user'].unique()):
            neg_items = set([x for x in all_items if x not in pos_items_dict[user]])
            neg_items_random_sampling = set(np.random.choice(list(neg_items), 800, replace = False))
            neg_popular_items = set(neg_populars_dict[user])
            neg_items_for_train = (neg_items & neg_popular_items) | neg_val_dict[user] # popular top 200 민주가 준 것이 유저가 안본 것 중에서 popular item을 순차적으로 뽑아낸 것을 줬기 때문에, 200개의 모든 샘플이 생기는 것이 맞다.
            neg_items_for_valid = neg_items_random_sampling | neg_popular_items | neg_val_dict[user]
            neg_items_dict[user].update(neg_items_for_train)
            neg_items_dict_for_valid[user].update(neg_items_for_valid)


        if config['name'] == "Bert4Rec":
            users = collections.defaultdict(list)
            for u, i in zip(train_df['user'], train_df['item']):
                users[u].append(i)

        if config['name'] == 'DeepFM':
            trainset = StaticDataset(train_df, neg_items_dict, user_dict, item_dict, config)
            validset = StaticTestDataset(neg_items_dict_for_valid, user_dict, item_dict, config)
        elif config['name'] == 'Bert4Rec':
            #TODO: Sequential Dataset으로 이름변경하기.
            trainset = SeqTrainDataset(users, 31360, 6807, config['arch']['args']['max_len'], config['mask_prob'])
            validset = SeqTestDataset(users, 31360, 6807, config['arch']['args']['max_len'], config['mask_prob'])

        train_loader = config.init_obj('data_loader', module_data, trainset, config)
        valid_loader = config.init_obj('data_loader', module_data, validset, config)

        train_batch = next(iter(train_loader))
        print(f"[TRAIN BATCH SHAPE]: {train_batch[0].shape}")
        valid_batch = next(iter(valid_loader))
        print(f"[VALID BATCH SHAPE]: {valid_batch.shape}")


        device, device_ids = prepare_device(config['n_gpu'])
        if config['name'] == 'DeepFM':
            model = config.init_obj('arch', module_arch)
        elif config['name'] == 'Bert4Rec':
            model = config.init_obj('arch', module_arch, device)
        model = model.to(device)

        # get function handles of loss and metrics
        criterion = getattr(module_loss, config['loss'])
        metrics = [getattr(module_metric, met) for met in config['metrics']]

        # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
        trainable_params = filter(lambda p: p.requires_grad, model.parameters())
        optimizer = config.init_obj('optimizer', torch.optim, trainable_params)
        lr_scheduler = config.init_obj('lr_scheduler', torch.optim.lr_scheduler, optimizer)

        trainer = Trainer(model, criterion, metrics, optimizer,
                        config=config,
                        device=device,
                        data_loader=train_loader,
                        valid_loader = valid_loader,
                        valid_target = valid_df,
                        lr_scheduler=lr_scheduler,
                        fold_num = idx + 1,
                        pos_items_dict = pos_items_dict)

        trainer.train()



if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default='config_deepfm.json', type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['--lr', '--learning_rate'], type=float, target='optimizer;args;lr'),
        CustomArgs(['--bs', '--batch_size'], type=int, target='data_loader;args;batch_size')
    ]
    config = ConfigParser.from_args(args, options)
    main(config)

