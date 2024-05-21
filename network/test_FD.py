import pandas as pd
from torch.backends.cudnn import is_available
import yaml
import os
import torch
import pickle,random
import torch_geometric
from torch_geometric.data import Dataset,Data
from torch_geometric.loader import DataLoader
import argparse
from utils.mubench_Dataset import mubench_Dataset
import numpy as np
import yaml
from FD_model import *
from utils import subgraph

parser = argparse.ArgumentParser(description='Training parameters of anomaly detection network.')

parser.add_argument("--random_seed", default=174, type=int)
parser.add_argument("--ns", default='app18', type=str)
parser.add_argument("--num_nodes", default=172, type=int)
parser.add_argument("--root_dir_normal", default='', type=str)   # dir of normal data
parser.add_argument("--root_dir_anomaly", default='', type=str)   # dir of anomalous data
parser.add_argument("--ratio", default=0.2, type=float)
parser.add_argument("--model", default='./model/app18/FD_model.pth', type=str)  # model name

args = parser.parse_args()

def main():    
    ns = args.ns
    num_nodes = args.num_nodes
    ratio = args.ratio
    normal_set = mubench_Dataset(root_dir=args.root_dir_normal)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    seed = args.random_seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    train_val_dataset, test_dataset = torch.utils.data.random_split(normal_set, [int(len(normal_set)*0.8), len(normal_set) - int(len(normal_set)*0.8)])

    fd_model = torch.load(args.model).to(device)
    fd_model.eval()
    test_loader = DataLoader(dataset=test_dataset,batch_size=1,shuffle=True,num_workers = 8)
 
    abnormal_set = mubench_Dataset(root_dir=args.root_dir_anomaly)
    threshold_abnormal_set, abnormal_set = torch.utils.data.random_split(abnormal_set, [int(len(abnormal_set)*0.2), len(abnormal_set) - int(len(abnormal_set)*0.2)])

    thre_anomaly_loader = DataLoader(dataset=threshold_abnormal_set,batch_size=1,shuffle=True,num_workers = 8)
    thre_normal_loader = DataLoader(dataset=train_val_dataset,batch_size=1,shuffle=True,num_workers = 8)

    thre_anomaly = []
    thre_normal = []

    for batch in thre_anomaly_loader:
        batch = batch.to(device)
        thre_anomaly.append(fd_model.anomaly_detection(batch).min(dim=1)[0].topk(k=int(num_nodes*ratio),largest=False)[0].mean()) 
    thre_anomaly = torch.tensor(thre_anomaly)

    for batch in thre_normal_loader:
        batch = batch.to(device)
        thre_normal.append(fd_model.anomaly_detection(batch).min(dim=1)[0].topk(k=int(num_nodes*ratio),largest=False)[0].mean()) 
    thre_normal = torch.tensor(thre_normal)

    
    # Searching for the best threshold, you can modify the search upper/lower bound and step size
    max_f1= 0
    best_thre = 0
    for thre in np.arange(thre_anomaly.median().item() - 10,np.median(thre_normal) + 30 ,0.01):  
        a = (thre_anomaly<thre).sum().item()
        b = (thre_normal<thre).sum().item()
        c = (thre_anomaly>=thre).sum().item()
        d = (thre_normal>=thre).sum().item()

        p = a / (a + b)
        r = a / (a +c )

        F1 = 2 * p * r / (p + r)

        if F1 > max_f1:
            max_f1 = F1
            best_thre = thre
    
    # Model performance test
    for batch in test_loader:
        batch = batch.to(device)
        x.append(fd_model.anomaly_detection(batch).min(dim=1)[0].topk(k=int(num_nodes*ratio),largest=False)[0].mean())
    x = torch.tensor(x)
    
    loader = DataLoader(dataset=abnormal_set,batch_size=1,shuffle=True,num_workers = 8)
    y = []
    for batch in loader:
        batch = batch.to(device)
        y.append(fd_model.anomaly_detection(batch).min(dim=1)[0].topk(k=int(num_nodes*ratio),largest=False)[0].mean()) 
    y = torch.tensor(y)

    thre = best_thre
    a = (y<thre).sum().item()
    b = (x<thre).sum().item()
    c = (y>=thre).sum().item()
    d = (x>=thre).sum().item()

    p = a / (a + b)
    r = a / (a + c )

    F1 = 2 * p * r / (p + r)

    print(thre,p,r,F1)

if __name__ == "main":
    main()