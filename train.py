import argparse
import collections
import torch
import numpy as np
import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
from parse_config import ConfigParser
from trainer import GBDTTrainer
from utils import prepare_device
from preprocess.preprocess import Preprocessor
import os
import pandas as pd
import pickle

# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)

def main(config):
    asset_dir = "/opt/ml/level2_movierecommendation_recsys-level3-recsys-06/saved/asset"
    preprocessor = Preprocessor()

    print("==========interaction dataframe 생성===========")
    interaction_df, title_df = preprocessor._preprocess_dataset()
    # all_items = sorted(list(title_df['item'].unique()))


    print("==========item, user 피클 파일 불러오기===========") 
    with open(os.path.join(asset_dir, "fm_item_dict.pkl"), 'rb') as f:
        item_dict = pickle.load(f)
    with open(os.path.join(asset_dir, "fm_user_dict.pkl"), 'rb') as f:
        user_dict = pickle.load(f)
    
    print("==========side information 추가===========") #16G
    item_df, user_df, interaction_df = preprocessor._make_dataset(item_dict, user_dict, True)

    print("=====print shape=====")
    print(item_df.shape)
    print(user_df.shape)
    print(interaction_df.shape)
    

    print("==========test_df 생성===========") #20G
    test_df = preprocessor._make_test_dataset()
    print("test_df.shape", test_df.shape)

    # if config['name'] == "GBDT":
    trainer = GBDTTrainer(config, interaction_df, test_df,item_df, user_df,len(user_df))
    trainer._train_epoch(5)



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
