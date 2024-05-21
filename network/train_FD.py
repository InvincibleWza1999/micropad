from ast import parse
from wsgiref import validate
import pandas as pd
from torch.backends.cudnn import is_available
import yaml
import os
import torch
import pickle
import torch_geometric
from torch_geometric.data import Dataset,Data
from torch_geometric.loader import DataLoader
import argparse
from utils.mubench_Dataset import mubench_Dataset
import numpy as np
import yaml
import matplotlib.pyplot as plt
from FD_model import *
import time,random
import argparse
  

parser = argparse.ArgumentParser(description='Training parameters of anomaly detection network.')

parser.add_argument("--random_seed", default=174, type=int)
parser.add_argument("--ns", default='app18', type=str)
parser.add_argument("--num_nodes", default=172, type=int)
parser.add_argument("--root_dir", default='', type=str)   # dir of normal data
parser.add_argument("--batch_size", default=32, type=int)


parser.add_argument("--epoch_num", default=200, type=int)
parser.add_argument("--lr", default=1e-4, type=float)
parser.add_argument("--dp", default=0.1, type=float)
parser.add_argument("--h_size", default=128, type=int)
parser.add_argument("--ggnn_num_layers", default=5, type=int)
parser.add_argument("--topk", default=0.5, type=float)
parser.add_argument("--conv1d_kernel_size", default=7, type=int)
parser.add_argument("--conv_out_channels", default=32, type=int)
parser.add_argument("--ggnn_out_channels", default=128, type=int)
parser.add_argument("--embedding_dim", default=64, type=int)

args = parser.parse_args()

def main():
    #torch.autograd.set_detect_anomaly(True)
    ns = args.ns 
    num_nodes = args.num_nodes
    normal_set = mubench_Dataset(root_dir=args.root_dir)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = FD_model(num_nodes=num_nodes,
                     h_size= args.h_size,
                     ggnn_num_layers= args.ggnn_num_layers,
                     topk= args.topk,
                     conv1d_kernel_size= args.conv1d_kernel_size,
                     conv_out_channels= args.conv_out_channels,
                     ggnn_out_channels=args.ggnn_out_channels,
                     embedding_dim=args.embedding_dim,
                     device=device
                     ).to(device)
    optimizer = torch.optim.Adam(lr=args.lr,params=model.parameters(),weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[40,70,100], gamma=0.2)
    min_loss = np.inf
    count = 0
    train_loss = []
    validate_loss = []
    
    seed = args.random_seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    train_val_dataset, test_dataset = torch.utils.data.random_split(normal_set, [int(len(normal_set)*0.8), len(normal_set) - int(len(normal_set)*0.8)])  #6:2:2
    start = time.time()

    for epoch in range(args.epoch_num):
        total_loss = 0
        train_l = 0
        

        train_dataset, valid_dataset = torch.utils.data.random_split(train_val_dataset, [int(len(train_val_dataset)*6/8), len(train_val_dataset) - int(len(train_val_dataset)*6/8)])
        train_loader = DataLoader(dataset=train_dataset,batch_size=args.batch_size,shuffle=True,num_workers = 16)
        valid_loader = DataLoader(dataset=valid_dataset,batch_size=args.batch_size,shuffle=True,num_workers = 8)
        
        model.train()
        for i,batch in enumerate(train_loader):
            optimizer.zero_grad()
            batch = batch.to(device)
            KL_loss,rec_loss= model(batch)
            loss = - rec_loss + KL_loss
    

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()

            print('rec_loss:{0},KL_loss:{1}'.format(-rec_loss,KL_loss))
            train_l += loss.item()
        
        train_loss.append(train_l /(i+1))
        print(f'average train loss at epoch {epoch+1}: {train_l / (i+1)}')
        scheduler.step()
        

        model.eval()
        for i,batch in enumerate(valid_loader):
            batch = batch.to(device)
            with torch.no_grad():
                KL_loss,rec_loss= model(batch)
            loss = - rec_loss + KL_loss
            total_loss += loss.item()
        
        total_loss = total_loss / (i+1)
        print(f'average valid loss at epoch {epoch+1}: {total_loss}')

        if total_loss<min_loss:
            min_loss = total_loss
            torch.save(model,f'./model/{ns}/FD_model.pth')
            count = 0
        else:
            count += 1
        
        if count >= 5:
            print("validation loss raises, training ends")
            break
        else:
            validate_loss.append(total_loss)

    end = time.time()
    print("training time:", end - start)

if __name__ == '__main__':
    main()