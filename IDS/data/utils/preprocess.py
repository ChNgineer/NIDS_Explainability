from pathlib import *
import os

import json
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
with open('mapping.json', 'r') as f:
    mapping = json.load(f)
with open('filters.json', 'r') as f:
    filters = json.load(f)

def extract_data(dir_paths, output_dir, filter_toggle=False):
    for dir in tqdm(dir_paths, desc='dirs'):
        for f in tqdm(dir.iterdir(), desc='files', leave=False):
            dataset = pd.read_csv(f, dtype={"SimillarHTTP":"string"})
            dataset.rename(columns=lambda x: x.strip(), inplace=True)
            dataset.drop(columns=['Unnamed: 0'], inplace=True)
            if filter_toggle:
                dataset = dataset[filters]
            dataset["Label"] = dataset["Label"].map(mapping)
            grouped = dataset.groupby('Label')
            for name, df in grouped:
                df = df.select_dtypes(include=['int', 'float', 'bool'])
                if os.path.isfile(f'{output_dir}/{name}.csv'):
                    df.to_csv(f'{output_dir}/{name}.csv', mode='a', header=False, index=False)
                else:
                    df.to_csv(f'{output_dir}/{name}.csv', index=False)

if __name__ == '__main__':
    dir_paths = [Path("../../../DDoS Evaluation Dataset (CIC-DDoS2019)/01-12/"), Path("../../../DDoS Evaluation Dataset (CIC-DDoS2019)/03-11/")]
    output_dir = '../Processed Data'
    extract_data(dir_paths, output_dir)