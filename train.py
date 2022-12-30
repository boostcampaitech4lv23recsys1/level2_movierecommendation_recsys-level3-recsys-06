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
        
    
    #TODO KFOLD Validation
    total_index = np.arange(len(interaction_df))
    kf = KFold(n_splits = config['n_fold'], shuffle = True, random_state = SEED)
    for idx, (train_index, valid_index) in enumerate(kf.split(total_index)):
        train_df = interaction_df.iloc[train_index].reset_index(drop = True)
        valid_df = interaction_df.iloc[valid_index].reset_index(drop = True)

        print(train_df.head(20))
        print(valid_df.head(20))
        print(train_df.index[:20])
        print(valid_df.index[:20])

        print(f"[BEFORE CONCAT SHAPE] {train_df.shape}, {valid_df.shape}")
        """
        valid_for_train과 valid_for_test로 남기는 CODE
        어차피 그 valid의 기준이 애매해질 바에는 학습 데이터셋을 더 남기는 게 나을듯
        len(group) > 10 + config['neg_ratio']도 고려해봤지만, data loader에서 
        negative sampling을 할 때, 차라리 중복되는 케이스가 있다하더라도 class imbalnace 해결 method 중 over sampling 느낌으로 
        replace = True로 냅두는 게 낫겠다. metric이 정확한 게 나음. -> 사소한 차이
        """
        valid_grouped = valid_df.groupby("user")
        valid_for_train_idx_list = set(valid_df.index)
        valid_for_test_idx_list = []
        for name, group in valid_grouped:
            if len(group) > 10:
                indices = np.random.choice(group.index, 10, replace = False)
                valid_for_test_idx_list.extend(list(indices))
            else:
                indices = group.index
                valid_for_test_idx_list.extend(list(indices))
        
        valid_for_test_idx_set = set(valid_for_test_idx_list)
        valid_for_train_idx_list = list(valid_for_train_idx_list - valid_for_test_idx_set)

        valid_for_train = valid_df.iloc[valid_for_train_idx_list]
        valid_df = valid_df.iloc[valid_for_test_idx_list]

        train_df = pd.concat([train_df, valid_for_train]).reset_index(drop = True)
        valid_df = valid_df.reset_index(drop = True)

        print(train_df[:25])
        print(valid_df[:25])
        valid_grouped2 = valid_df.groupby('user')
        cnt = 0
        for name, group in valid_grouped2:
            cnt += 1
        print(f"[CNT]: {cnt}")
        print(f"[AFTER CONCAT SHAPE] {train_df.shape}, {valid_df.shape}")

        #TODO: negative items user
        pos_items_dict = collections.defaultdict(set)
        neg_items_dict = collections.defaultdict(set)
        grouped = train_df.groupby('user')
        
        for name, group in tqdm(grouped):
            pos_items_dict[name].update(set(list(group['item'])))


        for user in tqdm(train_df['user'].unique()):
            neg_items = set([x for x in all_items if x not in pos_items_dict[user]])
            neg_items_dict[user].update(neg_items)

        
        print(train_df.sample(20))
        #data, neg_items_dict, user_dict, item_dict, config
        trainset = StaticDataset(train_df.sample(len(train_df) // 1000), neg_items_dict, user_dict, item_dict, config)
        validset = StaticTestDataset(neg_items_dict, user_dict, item_dict, config)

        train_loader = config.init_obj('data_loader', module_data, trainset, config)
        # train_loader = DataLoader(trainset, batch_size=32, shuffle=True, num_workers=4)
        valid_loader = config.init_obj('data_loader', module_data, validset, config)

        train_batch = next(iter(train_loader))
        valid_batch = next(iter(valid_loader))
        print(train_batch[0].shape, train_batch[1].shape)
        print(valid_batch.shape)


        model = config.init_obj('arch', module_arch)
        device, device_ids = prepare_device(config['n_gpu'])
        model = model.to(device)

        # get function handles of loss and metrics
        criterion = getattr(module_loss, config['loss'])
        metrics = [getattr(module_metric, met) for met in config['metrics']]

        # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
        trainable_params = filter(lambda p: p.requires_grad, model.parameters())
        optimizer = config.init_obj('optimizer', torch.optim, trainable_params)
        lr_scheduler = config.init_obj('lr_scheduler', torch.optim.lr_scheduler, optimizer)


        #TODO: Kfold별로 Model weight data값이 다른 폴더에 저장되도록 하고, test inference에서 각 모델의 inference를 합쳐준다.
        #TODO: model의 결과값에 prob과 user_idx, item_idx 매핑해주기. 그 후, user별로 prob이 높은 순서로 정렬한 다음, 나오는 item idx를 LabelEncoder의 reverse transform해주기.
        #TODO: 테스트 inference할 때는 neg_items_dict를 새로 설정해주어야 한다.
        trainer = Trainer(model, criterion, metrics, optimizer,
                        config=config,
                        device=device,
                        data_loader=train_loader,
                        valid_loader = valid_loader,
                        valid_target = valid_df,
                        lr_scheduler=lr_scheduler,
                        fold_num = idx + 1)

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

