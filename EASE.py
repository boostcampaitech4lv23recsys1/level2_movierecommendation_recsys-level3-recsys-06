import time
import os

import torch

import numpy as np
import bottleneck as bn

from parse_config import ConfigParser
from utils import prepare_device
from utils.ae_util import make_prediction_file, make_inference_data_and_mark, write_submission_file, get_loaders

from model.metric import recall_at_k_batch
from model.model import EASE

from data_loader.ae_dataloader import AETrainDataSet, AETestDataSet, ae_data_load, get_labels


# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)


def EASE_train_inference(config):
    start_time = time.time()
    root_data = config['root_data']
    data_dir = config['data_dir']
    model_name = config['model_name']
    output_path = config['output_path']
    n_users = config['n_users']
    n_items = config['n_items']
    EASE_lambda = config['EASE_lambda']

    n_gpu_use = torch.cuda.device_count() 
    device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')

    # 파일을 저장할 디렉토리 설정
    if not os.path.exists(output_path):
        os.mkdir(output_path)

    user_label, item_label = get_labels(data_dir)
    raw_data, train_mark = make_inference_data_and_mark(config, root_data, user_label, item_label)

    raw_data = raw_data.numpy()
    model = EASE(EASE_lambda)
    model.train(raw_data)
    inference_results = model.forward(raw_data)
    inference_results[raw_data.nonzero()]=-np.inf

    final_10 = bn.argpartition(-inference_results, 10, axis=1)[:, :10]  # 10개만 남겨둠

    total_recall_at_k = 0.2
    # 예측 파일을 저장함
    make_prediction_file(output_path, inference_results, config, total_recall_at_k, user_label, item_label)
    
    #제출 파일을 저장함
    write_submission_file(output_path, final_10, config, total_recall_at_k, user_label, item_label)



if __name__ == "__main__":
    config = {
        "root_data": './data/train/' ,
        "data_dir": './data/train/ae_data',
        "num_workers": 1,
        "model_name": 'EASE',
        "output_path": './output/auto_encoder',

        "n_users": 31360,
        "n_items": 6807,

        'EASE_lambda': 1500
    }
    EASE_train_inference(config)
    
