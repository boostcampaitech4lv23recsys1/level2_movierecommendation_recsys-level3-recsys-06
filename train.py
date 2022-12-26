import os
import argparse
import collections
import numpy as np
from tqdm import tqdm
import torch
from pathlib import Path
import pickle


import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
from data_loader.context_data_loader import StaticDataset
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
    print(interaction_df['item'].max(), max(all_items))

    logger = config.get_logger('train')
    # print(config['kfold'])

    # setup data_loader instances
    # data_loader = config.init_obj('data_loader', module_data)
    # valid_data_loader = data_loader.split_validation() #TODO: 우리가 kfold로 먼저 짜야함.!! ㅎ

    # build model architecture, then print to console
    # model = config.init_obj('arch', module_arch)
    # logger.info(model)

    with open(os.path.join(asset_dir, "item_dict.pkl"), 'rb') as f:
        item_dict = pickle.load(f)
    with open(os.path.join(asset_dir, "user_dict.pkl"), 'rb') as f:
        user_dict = pickle.load(f)
        
    
    #TODO KFOLD Validation
    total_index = np.arange(len(interaction_df))
    kf = KFold(n_splits = config['n_fold'], shuffle = True, random_state = SEED)
    for idx, (train_index, valid_index) in enumerate(kf.split(total_index)):
        train_df = interaction_df.iloc[train_index]
        valid_df = interaction_df.iloc[valid_index]

        #TODO: negative items user
        pos_items_dict = collections.defaultdict(set)
        neg_items_dict = collections.defaultdict(set)
        grouped = train_df.groupby('user')
        
        #TODO: negatvie item, positive item dict 
        # if os.path.isfile(os.path.join(asset_dir, "pos_items_dict.pkl")):
        #     with open(os.path.join(asset_dir, "pos_items_dict.pkl"), 'rb') as f:
        #         pos_items_dict = pickle.load(f)
        for name, group in tqdm(grouped):
            pos_items_dict[name].update(set(list(group['item'])))
        # with open(os.path.join(asset_dir, "pos_items_dict.pkl"), 'wb') as f:
        #     pickle.dump(pos_items_dict, f)

        # if os.path.isfile(os.path.join(asset_dir, "neg_items_dict.pkl")):
        #     with open(os.path.join(asset_dir, "neg_items_dict.pkl"), 'rb') as f:
        #         neg_items_dict = pickle.load(f)
        for user in tqdm(train_df['user'].unique()):
            neg_items = set([x for x in all_items if x not in pos_items_dict[user]])
            neg_items_dict[user].update(neg_items)
        # with open(os.path.join(asset_dir, "neg_items_dict.pkl"), 'wb') as f:
        #     pickle.dump(neg_items_dict, f)

        
        print(train_df.sample(20))
        #data, neg_items_dict, user_dict, item_dict, config
        trainset = StaticDataset(train_df, neg_items_dict, user_dict, item_dict, config)
        validset = StaticDataset(valid_df, neg_items_dict, user_dict, item_dict, config)

        train_loader = DataLoader(trainset, batch_size=32, shuffle=True, num_workers=4)
        # valid_loader = DataLoader(validset, batch_size=32, shuffle=False, num_workers=4)

        # one_batch = next(iter(train_loader))
        # print(one_batch.shape)
        #TODO: Validation은 negative sampling 과정이 없기 때문에 따로 짜기



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

        trainer = Trainer(model, criterion, metrics, optimizer,
                        config=config,
                        device=device,
                        data_loader=train_loader,
                        lr_scheduler=lr_scheduler)

        trainer.train()



if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default=None, type=str,
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

