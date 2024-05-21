from typing import Callable, Optional
import pandas as pd
import yaml
import os
import torch
import pickle
import torch_geometric
from torch_geometric.data import Dataset,Data
from itertools import product

class mubench_Dataset(Dataset):
    def __init__(self, root_dir, edge_dropping_ratio = 0.0):

        self.root_dir = root_dir
        self.file_names = [ os.path.join(root_dir,f) for f in os.listdir(self.root_dir) if f.endswith('pt')]
        self.edge_dropping_ratio = edge_dropping_ratio


    def __len__(self):
        return len(self.file_names)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        file_name  = self.file_names[idx]
        #pt_path = os.path.join(self.root_dir,file_name)
        try:
            sample = torch.load(file_name)
            sample.latent_edge_index = torch.tensor([[i,j] for (i,j) in list((product([_ for _ in range(sample.num_nodes)],[_ for _ in range(sample.num_nodes)]))) if i!=j]).T
            if self.edge_dropping_ratio!=0:
                resource_edge_num = sample.resource_edge_index.size(-1)
                invoke_edge_num = sample.invoke_edge_index.size(-1)
                left_ratio = 1 - self.edge_dropping_ratio

                random_columns = torch.randperm(resource_edge_num)[:int(resource_edge_num*left_ratio)]
                sample.resource_edge_index = sample.resource_edge_index[:,random_columns]

                random_columns = torch.randperm(invoke_edge_num)[:int(invoke_edge_num*left_ratio)]
                sample.invoke_edge_index = sample.invoke_edge_index[:,random_columns]
                sample.invoke_edge_attr = sample.invoke_edge_attr[random_columns,:]

        except KeyboardInterrupt:
            raise KeyboardInterrupt
        except Exception as e:
            print(e)
            print(f"Couldn't load {file_name}")

        return sample