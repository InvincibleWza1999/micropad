from wsgiref import validate
import pandas as pd
from torch.backends.cudnn import is_available
import yaml
import os
import torch,random
import pickle,time
import torch_geometric
from torch_geometric.data import Dataset,Data
from torch_geometric.loader import DataLoader
import argparse
from utils.mubench_Dataset import mubench_Dataset
import numpy as np
import yaml
import matplotlib.pyplot as plt
from FD_model import *
from RCL_model import *
from utils.subgraph import get_subgraph

import argparse
  

parser = argparse.ArgumentParser(description='Training parameters of anomaly detection network.')

parser.add_argument("--random_seed", default=588, type=int)
parser.add_argument("--ns", default='app18', type=str)
parser.add_argument("--root_dir", default='', type=str)   # dir of anomalous data
parser.add_argument("--batch_size", default=16, type=int)
parser.add_argument("--ratio", default=0.2, type=float)
parser.add_argument("--iteration", default=3, type=int)
parser.add_argument("--out_channels", default=128, type=int)

parser.add_argument("--epoch_num", default=200, type=int)
parser.add_argument("--lr", default=1e-4, type=float)
parser.add_argument("--dp", default=0.2, type=float)
parser.add_argument("--model", default='./model/app18/FD_model.pth', type=str)  # FD model
parser.add_argument("--tolerance", default=10, type=int)

args = parser.parse_args()
  
  

def main():
    seed = args.random_seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    fd_model = torch.load(args.model).to(device)
    bcn1d = torch.nn.BatchNorm1d(30, eps=1e-05, momentum=0.1, affine=False, track_running_stats=True).to(device)
    ratio = args.ratio

    rcl_model = RCL_model(out_channels=args.out_channels,
                          iteration= args.iteration,
                          dropout=args.dp).to(device)
    optimizer = torch.optim.Adam(lr=args.lr,params=rcl_model.parameters(),weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size = 10,gamma = 0.8)
    abnormal_set = mubench_Dataset(root_dir=args.root_dir)

    rcl_set = []
    for i,data in enumerate(abnormal_set):
        with torch.no_grad():
            data.batch = torch.zeros(data.num_nodes).to(torch.int32)
            data = data.to(device)
            data = fd_model.get_rcl_data(data)
            subset = data.ac.min(dim=1)[0].topk(k=max(int(data.ac.size(0)*ratio),20),largest=False)[1]
            data_t = get_subgraph(data,subset.to(torch.long))
            data_t.ac = bcn1d(data_t.ac)
            data_t.x = torch.concat([data_t.p,data_t.ac],dim=1)
            #data_t.x = data_t.p
            del data_t.batch
            data_t= data_t.to("cpu")
            rcl_set.append(data_t)
    
    train_val_dataset, test_dataset = torch.utils.data.random_split(rcl_set, [int(len(rcl_set)*0.8), len(rcl_set) - int(len(rcl_set)*0.8) ])  #7:1:2
    min_loss = np.inf
    min_train_loss = np.inf
    count = 0
    train_loss = []
    validate_loss = []
    best_model = None
    st = int(time.time())
    for epoch in range(args.epoch_num):
        total_loss = 0
        train_l = 0
        train_dataset, valid_dataset = torch.utils.data.random_split(train_val_dataset, [int(len(train_val_dataset)*(7/8)), len(train_val_dataset) - int(len(train_val_dataset)*(7/8)) ])
        train_loader = DataLoader(dataset=train_dataset,batch_size=args.batch_size,shuffle=True,num_workers = 8)
        valid_loader = DataLoader(dataset=valid_dataset,batch_size=args.batch_size,shuffle=True,num_workers = 8)
        
        rcl_model.train()
        for i,batch in enumerate(train_loader):
            optimizer.zero_grad()
            batch = batch[0].to(device)
            
            predict = rcl_model(batch)
            y = batch.y

            weight = torch.ones_like(y).to(torch.float32)
            weight[torch.where(y==1)] = ((y==1).sum()).to(torch.float32)
            loss = -torch.mean(weight*(y*torch.log(predict + torch.tensor(1e-7)) + (1-y)*torch.log(1-predict+ torch.tensor(1e-7))))
            loss.backward()

            nn.utils.clip_grad_norm_(rcl_model.parameters(), 5)
            optimizer.step()

            #print('loss:{0}'.format(loss)) 
            train_l += loss.item()

        train_loss.append(train_l / (i+1))
        if min_train_loss > train_l / (i+1) :
            # best_model = rcl_model
            #torch.save(rcl_model,'./model/RCL_model.pth')
            min_train_loss = train_l / (i+1)
        print(f'average train loss at epoch {epoch+1}: {train_l / (i+1)}')
        scheduler.step()
        

        rcl_model.eval()
        for i,batch in enumerate(valid_loader):
            batch = batch[0].to(device)
            with torch.no_grad():
            
                predict = rcl_model(batch)
                y = batch.y

                weight = torch.ones_like(y).to(torch.float32)
                weight[torch.where(y==1)] = ((y==1).sum()).to(torch.float32)
                loss = -torch.mean(weight*(y*torch.log(predict + torch.tensor(1e-7)) + (1-y)*torch.log(1-predict+ torch.tensor(1e-7))))
                total_loss += loss.item()

        total_loss = total_loss / (i+1)
        print(f'average valid loss at epoch {epoch+1}: {total_loss}, count = {count}')

        if total_loss<min_loss:
            min_loss = total_loss
            best_model = rcl_model
            count = 0
        else:
            if total_loss >= min_loss*1.02:
                count += 1
        
        if count >= args.tolerance:
            print("validation loss raises, training ends")
            break
        else:
            validate_loss.append(total_loss)

    duration = int(time.time()) - st
    print(duration)

    rcl_model = best_model
    test_loader = DataLoader(dataset=test_dataset,batch_size=1,shuffle=True,num_workers = 8)
    ac_1 = 0
    ac_2 = 0
    ac_3 = 0
    ac_4 = 0
    ac_5 = 0
    total = 0
    test_st = int(time.time())
    for i,batch in enumerate(test_loader):
        batch = batch[0].to(device)
        with torch.no_grad():    
            result = rcl_model(batch)
            predict_1,predict_2,predict_3,predict_4,predict_5 = torch.topk(result,k=1)[1],torch.topk(result,k=2)[1],torch.topk(result,k=3)[1],torch.topk(result,k=4)[1],torch.topk(result,k=5)[1]         
            y = torch.where(batch.y==1)[0]

            if len(y)==0:
                total += 1
                continue
            else:
                y = y.item()
                ac_1 = ac_1 + 1 if y in predict_1 else ac_1
                ac_2 = ac_2 + 1 if y in predict_1 else ac_2
                ac_3 = ac_3 + 1 if y in predict_3 else ac_3
                ac_4 = ac_4 + 1 if y in predict_4 else ac_4
                ac_5 = ac_5 + 1 if y in predict_5 else ac_5
                total += 1

    infer_time = (int(time.time()) - test_st) / total
    ac_1 = ac_1 / total
    ac_2 = ac_2 / total
    ac_3 = ac_3 / total
    ac_4 = ac_4 / total
    ac_5 = ac_5 / total
    avg_5 = (ac_1+ac_2+ac_3 + ac_4+ac_5)/5 
    print(ac_1,ac_2,ac_3,ac_4,ac_5,avg_5,infer_time)
        


if __name__ == '__main__':
    main()