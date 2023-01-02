import pickle
import pandas as pd
import numpy as np
from scipy import sparse

import torch
import csv
from tqdm import tqdm

from data_loader.ae_dataloader import AETrainDataSet, AETestDataSet
from data_loader.data_loaders import AEDataLoader


def make_prediction_file(output_path, inference_results, config, total_recall_at_k, user_label, item_label):
    model_name = config['model_name']
    if model_name == 'EASE':
        EASE_lambda = config['EASE_lambda']

        with open(output_path + f'/mat_{model_name}_lambda{EASE_lambda}.pkl', "wb") as file:
            pickle.dump(inference_results, file)

    else:    
        lr, n_epochs, dropout_rate, batch_size = config['lr'], config['n_epochs'], config['dropout_rate'], config['batch_size']

        with open(output_path + f'/mat_{model_name}_{round(lr,4)}_epoch{n_epochs}_{total_recall_at_k}_dropout{dropout_rate}_batch_{batch_size}.pkl', "wb") as file:
            pickle.dump(inference_results, file)


def make_inference_data_and_mark(config, root_data, user_label, item_label):
     # inference에서 쓸 rating 마련하기
    n_users, n_items = config['n_users'], config['n_items'] 

    ratings = pd.read_csv(root_data+'train_ratings.csv')[['user', 'item']]
    temp_rows, temp_cols = ratings['user'].apply(lambda x : user_label[x]), ratings['item'].apply(lambda x: item_label[x])
    raw_data = sparse.csr_matrix((np.ones_like(temp_rows), (temp_rows, temp_cols)), dtype='float64', shape=(n_users, n_items)).toarray()
    train_mark=raw_data.nonzero()  # 최종 인퍼런스 때 필터링해줄 마스크]

    return torch.Tensor(raw_data), train_mark  # 인퍼런스에 쓰기 위해 Tensor로 바꿔줌


def write_submission_file(output_path, final_10, config, total_recall_at_k, user_label, item_label):
    model_name = config['model_name']
    label_to_user = {v: k for k, v in user_label.items()}
    label_to_item = {v: k for k, v in item_label.items()}

    if model_name == 'EASE':
        EASE_lambda = config['EASE_lambda']
        with open(output_path + f'/mat_{model_name}_lambda{EASE_lambda}.pkl', "w") as csvfile:
            writer = csv.writer(csvfile)
            write_csv(writer, final_10, label_to_user, label_to_item)

    else:
        lr, n_epochs, dropout_rate, batch_size = config['lr'], config['n_epochs'], config['dropout_rate'], config['batch_size']
        with open(output_path + f'/sub_{model_name}_{lr}_epoch{n_epochs}_{total_recall_at_k}_dropout{dropout_rate}_batch_{batch_size}.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            write_csv(writer, final_10, label_to_user, label_to_item)
    

def write_csv(writer, final_10, label_to_user, label_to_item):
    writer.writerow(['user', 'item'])
    
    # Write the data rows
    print("Creating submission file: 31360 users")
    for i, row in tqdm(enumerate(final_10)):
        u_n = label_to_user[i]
        for j in row:
            writer.writerow([u_n, label_to_item[j]])


def get_loaders(tr_data, te_data, config):
    batch_size, num_workers = config['batch_size'], config['num_workers']
    
    trainset = AETrainDataSet(tr_data)
    validset = AETestDataSet(tr_data, te_data)

    train_loader = AEDataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    valid_loader = AEDataLoader(validset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, valid_loader