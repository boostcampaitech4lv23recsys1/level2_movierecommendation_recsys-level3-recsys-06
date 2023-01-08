import os
import argparse
import torch
import torch.nn.functional as F
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
from data_loader.sequential_data_loader import SeqTrainDataset, SeqTestDataset

from collections import defaultdict
from pathlib import Path
from preprocess.preprocess import Preprocessor

INF = int(1e9)

def recommendk(save_dir, grouped, user_encoder, item_encoder, k = 10):
    top = grouped.head(k)
    top = top.sort_values(by=['user', 'prob'], ascending=[True, False])
    top = top[['user', 'item']]
    top = top.astype('int')
    top['user'] = user_encoder.inverse_transform(top['user'])
    top['item'] = item_encoder.inverse_transform(top['item'])

    opath = Path(os.path.join(save_dir, f"output_{k}.csv"))
    opath.parent.mkdir(parents=True, exist_ok=True)
    top.to_csv(str(opath), index = False)

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

    with open(os.path.join(asset_dir, "item_popular.pkl"), 'rb') as f:
        neg_populars_dict = pickle.load(f)

    train_df = interaction_df

    pos_items_dict = defaultdict(set)
    neg_items_dict = defaultdict(set)
    grouped = train_df.groupby('user')
    
    for name, group in tqdm(grouped):
        pos_items_dict[name].update(set(list(group['item'])))

    for user in neg_populars_dict.keys():
        neg_populars_dict[user] = neg_populars_dict[user][:1000]

    for user in tqdm(train_df['user'].unique()):
        neg_items = set([x for x in all_items if x not in pos_items_dict[user]])
        neg_items_dict[user].update(neg_items & set(neg_populars_dict[user]))

    total_length = [len(neg_items_dict[user]) for user in neg_items_dict.keys()]
    total_length = sum(total_length)
    nfold_probs = np.zeros((total_length, config['n_fold']))
    for fold_num in range(1, config['n_fold']+1):
        
        if config['name'] == 'DeepFM':
            testset = StaticTestDataset(neg_items_dict, user_dict, item_dict, config)
        elif config['name'] == 'Bert4Rec':
            users = defaultdict(list)
            for u, i in zip(train_df['user'], train_df['item']):
                users[u].append(i)
            testset = SeqTestDataset(users, 31360, 6807, config['arch']['args']['max_len'], config['mask_prob'])

        test_loader = config.init_obj('data_loader', module_data, testset, config)
        #FOLD별로 모델을 load하여 inference
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if config['name'] == 'DeepFM':
            model = config.init_obj('arch', module_arch)
        elif config['name'] == 'Bert4Rec':
            model = config.init_obj('arch', module_arch, device)

        cpath = os.path.join(config['trainer']['save_dir'], 'models', config['name'], f"FOLD-{fold_num}", f"{config['name']}-best_model.pth")
        checkpoint = torch.load(cpath)
        state_dict = checkpoint['state_dict']
        
        if config['n_gpu'] > 1:
            model = torch.nn.DataParallel(model)
        model.load_state_dict(state_dict)
        model = model.to(device)
        model.eval()

        if config['name'] == 'DeepFM':
            infer_list = []
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
                
        elif config['name'] == 'Bert4Rec':
            infer_list = []
            nfold_probs = np.zeros((total_length, config['n_fold']))
            with torch.no_grad():
                for user, tokens in tqdm(test_loader):
                    user = user.numpy()
                    tokens = tokens.to(device)
                    output = model(tokens)

                    output = output[:, -1, :]
                    output = F.softmax(output, dim = -1)
                    output = output.detach().cpu().numpy()

                    for idx in range(test_loader.batch_size):
                        user_num = int(user[idx].item())
                        user_probs = output[idx]
                        infos = []
                        for item_num in range(6808):
                            if item_num == 0:
                                continue
                            if (item_num - 1) in pos_items_dict[user_num]:
                                infos.append(np.array([user_num, item_num-1, -INF])[np.newaxis, :])
                            else:
                                infos.append(np.array([user_num, item_num-1, user_probs[item_num]])[np.newaxis, :])
                        temp = np.concatenate(infos, axis = 0)
                        infer_list.append(temp)
            inference = np.concatenate(infer_list, axis = 0)
            indices = np.where(inference[:, 2] > 0)[0]
            inference = inference[indices]
            probs = inference[:, 2]
            nfold_probs[:, fold_num-1] = probs

            if fold_num == config['n_fold']:
                nfold_prob = nfold_probs.mean(axis = 1)
                inference[:, 2] = nfold_prob
                inference = pd.DataFrame(inference, columns = ['user', 'item', 'prob'])
                inference = inference.sort_values(by = 'prob', ascending = False)


    grouped = inference.groupby('user')
    recommendk(save_dir, grouped, user_encoder, item_encoder, k = 30)
    recommendk(save_dir, grouped, user_encoder, item_encoder, k = 10)


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
