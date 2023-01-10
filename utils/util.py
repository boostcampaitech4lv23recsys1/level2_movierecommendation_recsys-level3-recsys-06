import json
import torch
import pandas as pd
from pathlib import Path
from itertools import repeat
from collections import OrderedDict
from collections import defaultdict
import math
import numpy as np
from tqdm import tqdm

def ensure_dir(dirname):
    dirname = Path(dirname)
    if not dirname.is_dir():
        dirname.mkdir(parents=True, exist_ok=False)

def read_json(fname):
    fname = Path(fname)
    with fname.open('rt') as handle:
        return json.load(handle, object_hook=OrderedDict)

def write_json(content, fname):
    fname = Path(fname)
    with fname.open('wt') as handle:
        json.dump(content, handle, indent=4, sort_keys=False)

def inf_loop(data_loader):
    ''' wrapper function for endless data loader. '''
    for loader in repeat(data_loader):
        yield from loader

def prepare_device(n_gpu_use):
    """
    setup GPU device if available. get gpu device indices which are used for DataParallel
    """
    n_gpu = torch.cuda.device_count()
    if n_gpu_use > 0 and n_gpu == 0:
        print("Warning: There\'s no GPU available on this machine,"
              "training will be performed on CPU.")
        n_gpu_use = 0
    if n_gpu_use > n_gpu:
        print(f"Warning: The number of GPU\'s configured to use is {n_gpu_use}, but only {n_gpu} are "
              "available on this machine.")
        n_gpu_use = n_gpu
    device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')
    list_ids = list(range(n_gpu_use))
    return device, list_ids

class MetricTracker:
    def __init__(self, *keys, writer=None):
        self.writer = writer
        self._data = pd.DataFrame(index=keys, columns=['total', 'counts', 'average'])
        self.reset()

    def reset(self):
        for col in self._data.columns:
            self._data[col].values[:] = 0

    def update(self, key, value, n=1):
        if self.writer is not None:
            self.writer.add_scalar(key, value)
        self._data.total[key] += value * n
        self._data.counts[key] += n
        self._data.average[key] = self._data.total[key] / self._data.counts[key]

    def avg(self, key):
        return self._data.average[key]

    def result(self):
        return dict(self._data.average)



class Ensemble:
    def __init__(self, filenames:str, filepath:str):
        self.filenames = filenames
        self.output_path = [filepath+filename+'.csv' for filename in filenames]
        
    def simple_weighted(self,weight:list):
        if not len(self.output_path)==len(weight):
            raise ValueError("model과 weight의 길이가 일치하지 않습니다.")
        if np.sum(weight)!=1:
            raise ValueError("weight의 합이 1이 되도록 입력해 주세요.")
        input_frame = []
        total_list=[]
        for file_path in self.output_path:
            file = pd.read_csv(file_path)
            grouped = file.groupby("user")["item"].apply(list)
            input_frame.append(grouped)
        for key in tqdm(input_frame[0].keys()):
            vote_list = defaultdict(float)
            for i in range(len(input_frame)):
                output = input_frame[i]
                for content in output[key]:
                    vote_list[content] += weight[i]
            result=sorted(vote_list.items(),key=lambda x:-x[1])
            for index,value in result[:10]:
                total_list.append([key,index])
        total = pd.DataFrame(total_list,columns=['user','item'])   
        return total
    
    def complicated_weighted(self,weight:list,ranked_weight:float):
        if not len(self.output_path)==len(weight):
            raise ValueError("model과 weight의 길이가 일치하지 않습니다.")
        if np.sum(weight)!=1:
            raise ValueError("weight의 합이 1이 되도록 입력해 주세요.")
        input_frame = []
        total_list=[]
        for file_path in self.output_path:
            file = pd.read_csv(file_path)
            grouped = file.groupby("user")["item"].apply(list)
            input_frame.append(grouped)
        for key in tqdm(input_frame[0].keys()):
            vote_list = defaultdict(float)
            for i in range(len(input_frame)):
                output = input_frame[i]
                cnt=0
                for content in output[key]:
                    if cnt<10:
                        vote_list[content] += weight[i]
                    else:
                        vote_list[content] += (weight[i] * ranked_weight)
                    cnt+=1
            result=sorted(vote_list.items(),key=lambda x:-x[1])
            for index,value in result[:10]:
                total_list.append([key,index])
        total = pd.DataFrame(total_list,columns=['user','item'])   
        return total
    
    def rank_weighted(self,weight:list):
        if not len(self.output_path)==len(weight):
            raise ValueError("model과 weight의 길이가 일치하지 않습니다.")
        if np.sum(weight)!=1:
            raise ValueError("weight의 합이 1이 되도록 입력해 주세요.")
        input_frame = []
        total_list=[]
        for file_path in self.output_path:
            file = pd.read_csv(file_path)
            grouped = file.groupby("user")["item"].apply(list)
            input_frame.append(grouped)
        for key in tqdm(input_frame[0].keys()):
            vote_list = defaultdict(float)
            for i in range(len(input_frame)):
                output = input_frame[i]
                rank = 1
                for content in output[key]:
                    vote_list[content] += (weight[i] / math.log2(rank+1))
                    rank+=1
            result=sorted(vote_list.items(),key=lambda x:-x[1])
            for index,value in result[:10]:
                total_list.append([key,index])
        total = pd.DataFrame(total_list,columns=['user','item'])   
        return total
