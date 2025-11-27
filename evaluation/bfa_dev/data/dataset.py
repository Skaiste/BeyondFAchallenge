import os
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import pandas as pd


class BFADataset(Dataset):
    def __init__(self, csv_fpath, json_dir, task):
        self.df = pd.read_csv(csv_fpath)
        self.json_dir = json_dir
        self.task = task

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        json_fname = row['json']
        # Direct path to JSON file in flat output directory structure
        json_fpath = os.path.join(self.json_dir, json_fname)

        with open(json_fpath, 'r') as f:
            data = json.load(f)
        feature = torch.tensor(data, dtype=torch.float32)

        if self.task == 'age':
            age = torch.tensor(row['age'], dtype=torch.float32)
            return feature, age
        elif self.task == 'sex':
            sex = torch.tensor(row['sex'], dtype=torch.float32)
            return feature, sex
        elif self.task == 'cognitive_status':
            cog = torch.tensor(row['cognitive_status'], dtype=torch.long)
            return feature, cog
    
        else:
            raise ValueError(f'Unknown task : {self.task}. ' + \
                            'Choose one from "age", "sex", or "cognitive_status".')
