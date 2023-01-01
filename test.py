import os
import argparse
import torch
import numpy as np
import pandas as pd
import pickle
from tqdm import tqdm

import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
from parse_config import ConfigParser
from data_loader.context_data_loader import StaticDataset, StaticTestDataset

from collections import defaultdict
from pathlib import Path
from preprocess.preprocess import Preprocessor
def main(config):
    asset_dir = "/opt/ml/level2_movierecommendation_recsys-level3-recsys-06/saved/asset"
    save_dir = "/opt/ml/level2_movierecommendation_recsys-level3-recsys-06/saved/output"
    preprocessor = Preprocessor()
    interaction_df, title_df, user_encoder, item_encoder = preprocessor._preprocess_testset()
    all_items = sorted(list(title_df['item'].unique()))

    with open(os.path.join(asset_dir, "item_dict.pkl"), 'rb') as f:
        item_dict = pickle.load(f)
    with open(os.path.join(asset_dir, "user_dict.pkl"), 'rb') as f:
        user_dict = pickle.load(f)

    train_df = interaction_df

    pos_items_dict = defaultdict(set)
    neg_items_dict = defaultdict(set)
    grouped = train_df.groupby('user')
    
    for name, group in tqdm(grouped):
        pos_items_dict[name].update(set(list(group['item'])))


    for user in tqdm(train_df['user'].unique()):
        neg_items = set([x for x in all_items if x not in pos_items_dict[user]])
        neg_items_dict[user].update(neg_items)
    """
    Debugging Mode
    """
    # for user in neg_items_dict.keys():
    #     neg_items_dict[user] = set(list(neg_items_dict[user])[:20])

    """
    End
    """
    
    for fold_num in range(1, config['n_fold']+1):
        total_length = [len(neg_items_dict[user]) for user in neg_items_dict.keys()]
        total_length = sum(total_length)

        testset = StaticTestDataset(neg_items_dict, user_dict, item_dict, config)
        test_loader = config.init_obj('data_loader', module_data, testset, config)
        #FOLD별로 모델을 load하여 inference
        model = config.init_obj('arch', module_arch)
        cpath = os.path.join(config['trainer']['save_dir'], 'models', config['name'], f"FOLD-{fold_num}", f"{config['name']}-best_model.pth")
        checkpoint = torch.load(cpath)
        state_dict = checkpoint['state_dict']
        
        if config['n_gpu'] > 1:
            model = torch.nn.DataParallel(model)
        model.load_state_dict(state_dict)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        model.eval()

        infer_list = []
        nfold_probs = np.zeros((total_length, config['n_fold']))
        with torch.no_grad():
            for data in tqdm(test_loader):
                data = data.to(device)
                output = model(data)
                prob = output.detach().cpu().numpy()[:, np.newaxis]
                info = data[:, :2].detach().cpu().numpy()
                infos = np.concatenate([info, prob], axis = 1)
                infer_list.append(infos)

        inference = np.concatenate(infer_list, axis = 0)
        probs = inference[:, 2]
        nfold_probs[:, fold_num-1] = probs

        if fold_num == config['n_fold']:
            nfold_prob = nfold_probs.mean(axis = 1)
            inference[:, 2] = nfold_prob
            inference = pd.DataFrame(inference, columns = ['user', 'item', 'prob'])
            inference = inference.sort_values(by = 'prob', ascending = False)

    grouped = inference.groupby('user')
    top_10 = grouped.head(10)
    top_10 = top_10.sort_values(by=['user', 'prob'], ascending=[True, False])
    top_10 = top_10[['user', 'item']]
    top_10 = top_10.astype('int')
    top_10['user'] = user_encoder.inverse_transform(top_10['user'])
    top_10['item'] = item_encoder.inverse_transform(top_10['item'])

    opath = Path(os.path.join(save_dir, "output.csv"))
    opath.parent.mkdir(parents=True, exist_ok=True)
    top_10.to_csv(str(opath), index = False)


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    config = ConfigParser.from_args(args)
    main(config)
