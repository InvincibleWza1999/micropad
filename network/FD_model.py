from re import sub
import torch
from torch import Tensor
import torch.nn as nn
import torch_geometric
from torch.nn import Linear, Parameter,Dropout
from torch_geometric.utils import add_self_loops, degree
from torch_geometric.nn.conv import GatedGraphConv
from utils.TCN import TCN
from utils.structure_learning import structure_learning,drop_duplicate_edges
from utils.attention_merge import attention_net
from utils.CVAE import *
from torch.distributions import Normal, kl_divergence
import numpy as np
from utils.subgraph import get_subgraph

class FD_encoder(torch.nn.Module):
    def __init__(self,dropout=0.1,conv_out_channels=32,num_nodes=172,ggnn_out_channels=128,ggnn_num_layers=3,
                 conv1d_kernel_size=7,topk=0.5,h_size=128,embedding_dim=64,device='cuda:0',input_length=30,
                 cpu_dim = 5,mem_dim= 7,fs_dim=5, net_dim=4, mub_dim=5, I_dim=3, trace_dim=4):
        super().__init__()
        self.num_nodes = num_nodes
        self.embedding_dim = embedding_dim
        self.embedd = torch.nn.Embedding(num_embeddings=self.num_nodes,embedding_dim=self.embedding_dim)
        self.topk = topk
        self.input_length = input_length
        self.z_size = embedding_dim
        self.device = device

        self.cpu_tcn = TCN(cpu_dim + I_dim + trace_dim,conv_out_channels,conv1d_kernel_size,dropout=dropout)
        self.mem_tcn = TCN(mem_dim + I_dim + trace_dim,conv_out_channels,conv1d_kernel_size,dropout=dropout)
        self.net_tcn = TCN(net_dim + I_dim + trace_dim,conv_out_channels,conv1d_kernel_size,dropout=dropout)
        self.fs_tcn = TCN(fs_dim + I_dim + trace_dim,conv_out_channels,conv1d_kernel_size,dropout=dropout)
        self.mub_tcn = TCN(mub_dim + I_dim + trace_dim,conv_out_channels,conv1d_kernel_size,dropout=dropout)
  
        self.invoke_ggnn = GatedGraphConv(ggnn_out_channels,ggnn_num_layers,aggr="mean")
        self.internal_ggnn = GatedGraphConv(ggnn_out_channels,ggnn_num_layers,aggr="mean")
        self.resource_ggnn = GatedGraphConv(ggnn_out_channels,ggnn_num_layers,aggr="mean")
        
        self.attention_net = attention_net(ggnn_out_channels,h_size)
        self.prior_net = prior_network(input_length*3,self.z_size)
        self.encoder = VAE_encoder(h_size,self.z_size)

    def forward(self,batch):       #I: [batch_num, 3*input_length]
        #I = batch.sys_qps.reshape(self.batch_num,-1).to(torch.float32) 
        cpu_p = self.cpu_tcn(batch.cpu_x)
        mem_p = self.mem_tcn(batch.mem_x)
        net_p = self.net_tcn(batch.net_x)
        fs_p = self.fs_tcn(batch.fs_x)
        mub_p = self.mub_tcn(batch.mub_x)

        p = torch.cat((cpu_p,mem_p,net_p,fs_p,mub_p),dim=0)
        node_index = torch.cat((batch.cpu_node_index,batch.mem_node_index,batch.net_node_index,batch.fs_node_index,batch.mub_node_index),dim=0)
        p = p[torch.argsort(node_index)]

        batch_num = len(batch.batch.unique())
        u = self.embedd(torch.arange(0,self.num_nodes).to(self.device)).repeat(batch_num,1).to(self.device)     #[num_nodes,embedding_dim]
        d = torch.cat((p,u),dim=1)

        learned_invoke_edge_index, _ = structure_learning(d,batch.invoke_edge_index,batch.invoke_edge_attr,k=self.topk,device=self.device)
        h_invoke = self.invoke_ggnn(d,learned_invoke_edge_index)
        h_invoke = h_invoke.unsqueeze(1)

        learned_internal_edge_index, _ = structure_learning(d,batch.internal_edge_index,None,k=self.topk,device=self.device)
        h_internal = self.internal_ggnn(d,learned_internal_edge_index)
        h_internal = h_internal.unsqueeze(1)       
        
        learned_resource_edge_index, _ = structure_learning(d,batch.resource_edge_index,None,k=self.topk,device=self.device)
        h_resource = self.resource_ggnn(d,learned_resource_edge_index)
        h_resource = h_resource.unsqueeze(1)     

        learned_latent_edge_index, _ = structure_learning(d,batch.latent_edge_index,None,k=self.topk,device=self.device)
        h_latent = self.resource_ggnn(d,learned_latent_edge_index)
        h_latent = h_latent.unsqueeze(1)     
         

        h = torch.concat((h_internal,h_internal,h_resource,h_latent),dim=1)
        #h = torch.concat((h_invoke,h_internal),dim=1)
        h = self.attention_net(h)           #[num_nodes, h_size]
        
        z_mu, z_var = self.encoder(h)
        #z_var = z_var + torch.tensor([1e-7])

        return z_mu, z_var 
    
    def get_p(self,batch):
        with torch.no_grad():
            cpu_p = self.cpu_tcn(batch.cpu_x)
            mem_p = self.mem_tcn(batch.mem_x)
            net_p = self.net_tcn(batch.net_x)
            fs_p = self.fs_tcn(batch.fs_x)
            mub_p = self.mub_tcn(batch.mub_x)

            p = torch.cat((cpu_p,mem_p,net_p,fs_p,mub_p),dim=0)
            node_index = torch.cat((batch.cpu_node_index,batch.mem_node_index,batch.net_node_index,batch.fs_node_index,batch.mub_node_index),dim=0)
            p = p[torch.argsort(node_index)]

            return p
    
    def get_learned_edge(self,batch):
        with torch.no_grad():
            cpu_p = self.cpu_tcn(batch.cpu_x)
            mem_p = self.mem_tcn(batch.mem_x)
            net_p = self.net_tcn(batch.net_x)
            fs_p = self.fs_tcn(batch.fs_x)
            mub_p = self.mub_tcn(batch.mub_x)

            p = torch.cat((cpu_p,mem_p,net_p,fs_p,mub_p),dim=0)
            node_index = torch.cat((batch.cpu_node_index,batch.mem_node_index,batch.net_node_index,batch.fs_node_index,batch.mub_node_index),dim=0)
            p = p[torch.argsort(node_index)]

            batch_num = len(batch.batch.unique())
            u = self.embedd(torch.arange(0,self.num_nodes).to(self.device)).repeat(batch_num,1).to(self.device)     #[num_nodes,embedding_dim]
            d = torch.cat((p,u),dim=1)

            learned_invoke_edge_index, learned_invoke_edge_attr = structure_learning(d,batch.invoke_edge_index,batch.invoke_edge_attr,k=self.topk,device=self.device)
            learned_internal_edge_index, _ = structure_learning(d,batch.internal_edge_index,None,k=self.topk,device=self.device)
            learned_resource_edge_index, _ = structure_learning(d,batch.resource_edge_index,None,k=self.topk,device=self.device)
            learned_latent_edge_index, _ = structure_learning(d,batch.latent_edge_index,None,k=self.topk,device=self.device)

            return learned_invoke_edge_index, learned_internal_edge_index, learned_resource_edge_index, learned_invoke_edge_attr, learned_latent_edge_index 

class FD_model(torch.nn.Module):
    def __init__(self,dropout=0.1,conv_out_channels=32,num_nodes=172,ggnn_out_channels=128,ggnn_num_layers=5,
                 conv1d_kernel_size=7,topk=0.7,h_size=128,embedding_dim=64,device='cuda:0',input_length=30,
                 cpu_dim = 5,mem_dim= 7,fs_dim=5, net_dim=4, mub_dim=5, I_dim=3, trace_dim=4):
        super().__init__()
        self.num_nodes = num_nodes
        self.topk = topk
        self.input_length = input_length
        self.device = device
        self.z_size = embedding_dim

        self.prior_net = prior_network(input_length*3,self.z_size)
        self.encoder = FD_encoder(dropout,conv_out_channels,num_nodes,ggnn_out_channels,ggnn_num_layers,
                 conv1d_kernel_size,topk,h_size,embedding_dim,device,input_length,cpu_dim,mem_dim,fs_dim, net_dim, mub_dim, I_dim, trace_dim)
        
        self.cpu_decoder = VAE_decoder(3*input_length+embedding_dim,cpu_dim + I_dim + trace_dim)
        self.mem_decoder = VAE_decoder(3*input_length+embedding_dim,mem_dim + I_dim + trace_dim)
        self.net_decoder = VAE_decoder(3*input_length+embedding_dim,net_dim + I_dim + trace_dim)
        self.fs_decoder = VAE_decoder(3*input_length+embedding_dim,fs_dim + I_dim + trace_dim)
        self.mub_decoder = VAE_decoder(3*input_length+embedding_dim,mub_dim + I_dim + trace_dim)   

        self.cpu_dim = cpu_dim
        self.mem_dim = mem_dim
        self.fs_dim = fs_dim
        self.net_dim = net_dim
        self.mub_dim = mub_dim

    def feature_concat(self,batch_raw,with_trace=True):
        batch = batch_raw.clone()
        batch_num = len(batch.batch.unique())
        if len(batch.trace_series.size()) != 4:
            batch.trace_series = batch.trace_series.reshape(batch_num,-1,batch.trace_series.size(1),batch.trace_series.size(-1))
        if len(batch.sys_qps.size()) != 3:
            batch.sys_qps = batch.sys_qps.reshape(batch_num,-1,batch.sys_qps.size(-1))

        batch_num_list = batch.batch[batch.cpu_node_index]
        trace = torch.index_select(batch.trace_series,0,batch_num_list)
        trace = trace[torch.arange(trace.size(0)),batch.cpu_pod_idx]
        intensity = torch.index_select(batch.sys_qps,0,batch_num_list)
        if with_trace:
            batch.cpu_x =  torch.cat((batch.cpu_x,trace,intensity),dim=1)
        else:
            batch.cpu_x =  torch.cat((batch.cpu_x,intensity),dim=1)

        batch_num_list = batch.batch[batch.mem_node_index]
        trace = torch.index_select(batch.trace_series,0,batch_num_list)
        trace = trace[torch.arange(trace.size(0)),batch.mem_pod_idx]
        intensity = torch.index_select(batch.sys_qps,0,batch_num_list)
        if with_trace:
            batch.mem_x =  torch.cat((batch.mem_x,trace,intensity),dim=1)
        else:
            batch.mem_x =  torch.cat((batch.mem_x,intensity),dim=1)
        #batch.mem_x =  torch.cat((batch.mem_x,trace,intensity),dim=1)

        batch_num_list = batch.batch[batch.fs_node_index]
        trace = torch.index_select(batch.trace_series,0,batch_num_list)
        trace = trace[torch.arange(trace.size(0)),batch.fs_pod_idx]
        intensity = torch.index_select(batch.sys_qps,0,batch_num_list)
        if with_trace:
            batch.fs_x =  torch.cat((batch.fs_x,trace,intensity),dim=1)
        else:
            batch.fs_x =  torch.cat((batch.fs_x,intensity),dim=1)
        #batch.fs_x =  torch.cat((batch.fs_x,trace,intensity),dim=1)

        batch_num_list = batch.batch[batch.net_node_index]
        trace = torch.index_select(batch.trace_series,0,batch_num_list)
        trace = trace[torch.arange(trace.size(0)),batch.net_pod_idx]
        intensity = torch.index_select(batch.sys_qps,0,batch_num_list)
        if with_trace:
            batch.net_x =  torch.cat((batch.net_x,trace,intensity),dim=1)
        else:
            batch.net_x =  torch.cat((batch.net_x,intensity),dim=1)
        #batch.net_x =  torch.cat((batch.net_x,trace,intensity),dim=1)

        batch_num_list = batch.batch[batch.mub_node_index]
        trace = torch.index_select(batch.trace_series,0,batch_num_list)
        trace = trace[torch.arange(trace.size(0)),batch.mub_pod_idx]
        intensity = torch.index_select(batch.sys_qps,0,batch_num_list)
        if with_trace:
            batch.mub_x =  torch.cat((batch.mub_x,trace,intensity),dim=1)
        else:
            batch.mub_x =  torch.cat((batch.mub_x,intensity),dim=1)
        #batch.mub_x =  torch.cat((batch.mub_x,trace,intensity),dim=1)

        return batch
    

    def forward(self,batch_raw):
        batch_num = len(batch_raw.batch.unique())
        I = batch_raw.sys_qps.reshape(batch_num,-1).to(torch.float32) 
        batch = self.feature_concat(batch_raw)

        z_prior_mu, z_prior_var = self.prior_net(I)           # z_prior_x: [batch_num, z_size]
        z_prior_mu = z_prior_mu.unsqueeze(1).repeat(1,self.num_nodes,1).reshape(-1,self.z_size)      #[num_nodes, z_size]
        z_prior_var = z_prior_var.unsqueeze(1).repeat(1,self.num_nodes,1).reshape(-1,self.z_size)  #[num_nodes, z_size]
        p_z_I = Normal(z_prior_mu, z_prior_var + torch.tensor(1e-7).to(self.device))    #[num_nodes, z_size]
        
        z_mu, z_var = self.encoder(batch)
    
        q_z_xI = Normal(z_mu,z_var + torch.tensor(1e-7).to(self.device))    #[num_nodes, z_size]
        
        z = q_z_xI.rsample()   #[ num_nodes, z_size]
        u = self.get_u(batch_raw)    #[num_nodes, embedding_dim]
        z_concat = torch.mul(z,u)
        
        I = I.unsqueeze(1).repeat(1,self.num_nodes,1).reshape(-1,I.size(-1))     #[num_nodes, 3*input_length]
        z_concat = torch.cat((z_concat,I),dim=-1)     #[num_nodes, z_size + 3 * input_size]
        z_concat = z_concat.unsqueeze(1)

        z_cpu = z_concat[batch.cpu_node_index]
        z_mu_cpu, z_var_cpu = self.cpu_decoder(z_cpu)   #[xxx_nodes, cpu_size, input_length]
        # cpu_loss =  torch.square(z_mu_cpu - batch.cpu_x) / (2 * torch.square(z_var_cpu)) + torch.log(z_var_cpu) 
        # cpu_loss = cpu_loss.mean(dim=1)
        cpu_loss = Normal(z_mu_cpu,z_var_cpu).log_prob(batch.cpu_x).mean(dim=1)
        #cpu_loss = torch.square(z_mu_cpu - z_cpu).mean(dim=1)

        z_mem = z_concat[batch.mem_node_index]
        z_mu_mem, z_var_mem = self.mem_decoder(z_mem)
        # mem_loss =  torch.square(z_mu_mem - batch.mem_x) / (2 * torch.square(z_var_mem)) + torch.log(z_var_mem) 
        # mem_loss = mem_loss.mean(dim=1)
        mem_loss = Normal(z_mu_mem,z_var_mem).log_prob(batch.mem_x).mean(dim=1)
        #mem_loss = torch.square(z_mu_mem - z_mem).mean(dim=1)

        z_net = z_concat[batch.net_node_index]
        z_mu_net, z_var_net = self.net_decoder(z_net)
        # net_loss =  torch.square(z_mu_net - batch.net_x) / (2 * torch.square(z_var_net)) + torch.log(z_var_net) 
        # net_loss = net_loss.mean(dim=1)
        net_loss = Normal(z_mu_net,z_var_net).log_prob(batch.net_x).mean(dim=1)
        #net_loss = torch.square(z_mu_net - z_net).mean(dim=1)

        z_fs = z_concat[batch.fs_node_index]
        z_mu_fs, z_var_fs = self.fs_decoder(z_fs)
        # fs_loss =  torch.square(z_mu_fs - batch.fs_x) / (2 * torch.square(z_var_fs)) + torch.log(z_var_fs) 
        # fs_loss = fs_loss.mean(dim=1)
        fs_loss = Normal(z_mu_fs,z_var_fs).log_prob(batch.fs_x).mean(dim=1)
        #fs_loss = torch.square(z_mu_fs - z_fs).mean(dim=1)

        z_mub = z_concat[batch.mub_node_index]
        z_mu_mub, z_var_mub = self.mub_decoder(z_mub)
        # mub_loss =  torch.square(z_mu_mub - batch.mub_x) / (2 * torch.square(z_var_mub)) + torch.log(z_var_mub)
        # mub_loss = mub_loss.mean(dim=1)
        mub_loss = Normal(z_mu_mub,z_var_mub).log_prob(batch.mub_x).mean(dim=1)
        #mub_loss = torch.square(z_mu_mub - z_mub).mean(dim=1)

        KL_loss = kl_divergence(q_z_xI,p_z_I).sum(dim=1).mean()
        rec_loss = torch.cat((cpu_loss,mem_loss,net_loss,fs_loss,mub_loss),dim=0)  #[num_nodes, input_length]
        #print(rec_loss.shape)
        rec_loss = rec_loss.sum(dim=1).mean()

        #loss = KL_loss + rec_loss

        return KL_loss,rec_loss
                

    def anomaly_detection(self,batch_raw):
        with torch.no_grad():
            #print(batch_raw.cpu_x.shape)
            batch_num = len(batch_raw.batch.unique())
            I = batch_raw.sys_qps.reshape(batch_num,-1).to(torch.float32) 
            batch = self.feature_concat(batch_raw)
            z_prior_mu, z_prior_var = self.prior_net(I)           # z_prior_x: [batch_num, z_size]
            z_prior_mu = z_prior_mu.unsqueeze(1).repeat(1,self.num_nodes,1).reshape(-1,self.z_size)      #[num_nodes, z_size]
            z_prior_var = z_prior_var.unsqueeze(1).repeat(1,self.num_nodes,1).reshape(-1,self.z_size)     #[num_nodes, z_size]
            p_z_I = Normal(z_prior_mu, z_prior_var + torch.tensor(1e-7).to(self.device))    #[num_nodes, z_size]
            z = p_z_I.rsample()

            u = self.get_u(batch_raw)    #[num_nodes, embedding_dim]
            z_concat = torch.mul(z,u)

            I = I.unsqueeze(1).repeat(1,self.num_nodes,1).reshape(-1,I.size(-1))     #[num_nodes, 3*input_length]
            z_concat = torch.cat((z_concat,I),dim=-1)     #[num_nodes, z_size + 3 * input_size]

            z_concat = z_concat.unsqueeze(1)

            z_cpu = z_concat[batch.cpu_node_index]
            z_mu_cpu, z_var_cpu = self.cpu_decoder(z_cpu)
            #print(batch.cpu_x.shape,z_mu_cpu.shape)
            cpu_loss = Normal(z_mu_cpu[:,0:self.cpu_dim,:],z_var_cpu[:,0:self.cpu_dim,:] + torch.tensor(1e-7).to(self.device)).log_prob(batch.cpu_x[:,0:self.cpu_dim,:]).mean(dim=1)

            z_mem = z_concat[batch.mem_node_index]
            z_mu_mem, z_var_mem = self.mem_decoder(z_mem)
            mem_loss = Normal(z_mu_mem[:,0:self.mem_dim,:],z_var_mem[:,0:self.mem_dim,:] + torch.tensor(1e-7).to(self.device)).log_prob(batch.mem_x[:,0:self.mem_dim,:]).mean(dim=1)

            z_net = z_concat[batch.net_node_index]
            z_mu_net, z_var_net = self.net_decoder(z_net)
            net_loss = Normal(z_mu_net[:,0:self.net_dim,:],z_var_net[:,0:self.net_dim,:] + torch.tensor(1e-7).to(self.device)).log_prob(batch.net_x[:,0:self.net_dim,:]).mean(dim=1)

            z_fs = z_concat[batch.fs_node_index]
            z_mu_fs, z_var_fs = self.fs_decoder(z_fs)
            fs_loss = Normal(z_mu_fs[:,0:self.fs_dim,:],z_var_fs[:,0:self.fs_dim,:] + torch.tensor(1e-7).to(self.device)).log_prob(batch.fs_x[:,0:self.fs_dim,:]).mean(dim=1)

            z_mub = z_concat[batch.mub_node_index]
            z_mu_mub, z_var_mub = self.mub_decoder(z_mub)
            mub_loss = Normal(z_mu_mub[:,0:self.mub_dim,:],z_var_mub[:,0:self.mub_dim,:] + torch.tensor(1e-7).to(self.device)).log_prob(batch.mub_x[:,0:self.mub_dim,:]).mean(dim=1)

            anomaly_score = torch.cat((cpu_loss,mem_loss,net_loss,fs_loss,mub_loss),dim=0)
            node_index = torch.cat((batch.cpu_node_index,batch.mem_node_index,batch.net_node_index,batch.fs_node_index,batch.mub_node_index),dim=0)
            anomaly_score = anomaly_score[torch.argsort(node_index)]

            return anomaly_score   #[num_nodes, input_length]

        

    def get_rcl_data(self,batch_raw,use_learned_edge = True):
        with torch.no_grad():
            batch = self.feature_concat(batch_raw)
            result = torch_geometric.data.Data()
            result.num_nodes = batch.num_nodes
            result.embedding = self.get_u(batch_raw)
            if use_learned_edge:
                result.invoke_edge_index, result.internal_edge_index, result.resource_edge_index, result.invoke_edge_attr, result.latent_edge_index = self.encoder.get_learned_edge(batch)
            else:
                result.invoke_edge_index, result.internal_edge_index, result.resource_edge_index, result.invoke_edge_attr, result.latent_edge_index  = \
                    batch.invoke_edge_index, batch.internal_edge_index, batch.resource_edge_index, batch.invoke_edge_attr, batch.latent_edge_index
            try:
                result.batch = batch.batch
            except:
                pass
            result.p = self.encoder.get_p(batch)
            result.ac = self.anomaly_detection(batch_raw)
            result.y = batch.y

            return result               

    def get_rcl_data_v2(self,batch_raw):
        with torch.no_grad():
            batch = self.feature_concat(batch_raw,with_trace=False)
            
            batch.ac = self.anomaly_detection(batch_raw)
            batch.embedding = self.get_u(batch_raw)
            batch.invoke_edge_index, batch.internal_edge_index, batch.resource_edge_index, batch.invoke_edge_attr = self.encoder.get_learned_edge(self.feature_concat(batch_raw))
            
            del batch.cpu_pod_idx
            del batch.fs_pod_idx
            del batch.mem_pod_idx
            del batch.mub_pod_idx
            del batch.net_pod_idx
            del batch.trace_series
            del batch.sys_qps
            del batch.batch

            return batch              

    def get_p(self,batch_raw):
        with torch.no_grad():
            batch = self.feature_concat(batch_raw)
            p = self.encoder.get_p(batch)
            return p
    
    def get_u(self,batch_raw):
        with torch.no_grad():
            batch_num = len(batch_raw.batch.unique())
            u = self.encoder.embedd(torch.arange(0,self.num_nodes).to(self.device)).repeat(batch_num,1)
            return u
            
    def get_rcl_data_all(self,batch_raw):
        with torch.no_grad():
            batch = self.feature_concat(batch_raw)
            result = torch_geometric.data.Data()
            result.num_nodes = batch.num_nodes
            result.embedding = self.get_u(batch_raw)
            result.invoke_edge_index_learned, result.internal_edge_index_learned, result.resource_edge_index_learned, result.invoke_edge_attr_learned = self.encoder.get_learned_edge(batch)
            result.invoke_edge_index, result.internal_edge_index, result.resource_edge_index, result.invoke_edge_attr = \
                 batch.invoke_edge_index, batch.internal_edge_index, batch.resource_edge_index, batch.invoke_edge_attr
            try:
                result.batch = batch.batch
            except:
                pass
            result.p = self.encoder.get_p(batch)
            result.ac = self.anomaly_detection(batch_raw)
            result.y = batch.y

            return result            