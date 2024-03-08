import pandas as pd
from torch.utils.data import Dataset
from tqdm import tqdm

class CICDDoS2019Dataset(Dataset):
    def __init__(self, dirs):
        dfs = {}
        for dir in tqdm(dirs, desc='dirs'):
            for f in tqdm(dir.iterdir(), desc='files', leave=False):
                dfs[f.name] = pd.read_csv(f, dtype={"SimillarHTTP":"string"})
                dfs[f.name].rename(columns=lambda x: x.strip(), inplace=True)
                dfs[f.name].drop(columns=["Unnamed: 0"], inplace=True)
        self.data = pd.concat(dfs.values(), ignore_index=True)
        self.labels = self.data['Label']
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data.iloc[idx]
    
        