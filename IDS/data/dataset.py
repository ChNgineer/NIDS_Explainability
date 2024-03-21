import os
from pathlib import *
import random

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

class CICDDoS2019Dataset(Dataset):
    def __init__(self, dir_path:str, seq_length:int=1, transform=None, 
                 transform_args:list[any]=None, filters:list[str]=None, 
                 balanced:bool=False, sample_sizes:int|float|dict[int]|dict[float]=None, random_seed:int=None):
        
        self.transform = transform
        self.t_args = transform_args
        self.sequence_features = True if seq_length > 1 else False

        if random_seed:
            np.random.seed(random_seed)

        size_map = {}
        for f_path in tqdm(Path(dir_path).iterdir(), desc='allocating...'):
            if os.path.isfile(f_path):
                num_lines = sum(1 for _ in open(f_path, "rb"))
                size_map[str(f_path)] = num_lines - 1
        
        if sample_sizes:
            match sample_sizes:
                case int():
                    self.sample_sizes = dict.fromkeys(size_map.keys(),sample_sizes)
                case float():
                    self.sample_sizes = {key: int(value * sample_sizes) for key, value in size_map.items()}
                case dict():
                    assert set(sample_sizes.keys()) == set(size_map.keys())
                    if all(isinstance(value, int) for value in sample_sizes.values()):
                        self.sample_sizes = sample_sizes
                    elif all(isinstance(value, float) for value in sample_sizes.values()):
                        self.sample_sizes = {key: int(value * sample_sizes[key]) for key, value in size_map.items()}
                    else:
                        TypeError('invalid sample_size dict')
                case _:
                    TypeError('sample_sizes needs to be of type int, float, dict[int], or dict[float]')
            for key in self.sample_sizes.keys():
                if self.sample_sizes[key] > size_map[key]:
                    self.sample_sizes[key] = size_map[key]
                assert self.sample_sizes[key] <= size_map[key]
        
        if balanced and not sample_sizes:
            min_size = min(size_map.values())
            self.sample_sizes = dict.fromkeys(size_map.keys(),min_size)
        
        if self.sequence_features:
            for key in self.sample_sizes.keys():
                if self.sample_sizes[key] % seq_length != 0:
                    self.sample_sizes[key] -= self.sample_sizes[key] % seq_length
                assert self.sample_sizes[key] % seq_length == 0
            
        dfs = []
        for f_path in tqdm(Path(dir_path).iterdir(), desc='collecting data...'):
            if os.path.isfile(f_path):
                negative_sample = size_map[str(f_path)] - self.sample_sizes[str(f_path)]
                skip_index = random.sample(range(1, size_map[str(f_path)]), negative_sample) if negative_sample > 0 else None
                dfs.append(pd.read_csv(f_path, skiprows=skip_index, usecols=filters))
                if self.sequence_features:
                    assert(len(dfs[-1]) % seq_length == 0)
        df = pd.concat(dfs)

        data = torch.tensor(df.iloc[:,:-1].values, dtype=torch.float32)
        labels = torch.tensor(df.iloc[:,-1].values, dtype=torch.long)
            
        if self.transform:
            if self.t_args:
                data = self.transform(data,*self.t_args)
            data = self.transform(data)

        if self.sequence_features:
            data = torch.reshape(data, (-1,seq_length, data.shape[1])) # n, _ , features
            self.data = torch.permute(data,(0,2,1)) # n, sequence_length , features -> n, channels(features), sequence_length | conv over sequence
            labels = torch.reshape(labels, (-1,seq_length))

            for sequence_labels in tqdm(labels[:], desc='verifying sequence labels...'):
                same = torch.all(torch.eq(sequence_labels, sequence_labels[0]))
                assert(same)
            self.labels = labels[:,0]
        else:
            self.data = data.unsqueeze(1) # n, channel(1), feature | conv over packet features
            self.labels = labels
        
        
    def __len__(self):
        return len(self.data) 
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]
    
def min_max_normalize(data:torch.Tensor, dim:int=1, new_min:float=0, new_max:float=1):
    normed = data
    normed -= normed.min(dim, keepdim=True)[0]
    normed /= normed.max(dim, keepdim=True)[0]
    normed *= (new_max - new_min)
    normed += new_min
    return normed
    
        