import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
import math

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