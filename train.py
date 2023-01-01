import argparse
import collections
import torch
import numpy as np
import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
from parse_config import ConfigParser
from trainer import Trainer
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

    interaction_df, title_df = preprocessor._preprocess_dataset()
    all_items = sorted(list(title_df['item'].unique()))

    logger = config.get_logger('train')

    with open(os.path.join(asset_dir, "item_dict.pkl"), 'rb') as f:
        item_dict = pickle.load(f)
    with open(os.path.join(asset_dir, "user_dict.pkl"), 'rb') as f:
        user_dict = pickle.load(f)
        
    item_df, user_df, interaction_df = preprocessor._make_dataset(item_dict, user_dict, True)

    breakpoint()


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
