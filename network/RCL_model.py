import torch
import torch.nn as nn
from torch.nn import Linear, Parameter,Dropout
from torch_geometric.utils import add_self_loops, degree,softmax
from torch_geometric.nn.conv import GATv2Conv,GATConv
from utils.GAT_embed_Conv import GAT_embed_Conv
from utils.structure_learning import structure_learning,drop_duplicate_edges
from utils.attention_merge import attention_net
import numpy as np

class RCL_model(torch.nn.Module):
    def __init__(self,input_length=30+24,out_channels=32,edge_dim=11,gat_heads=1,dropout=0.3,iteration = 2):
        super().__init__()
        self.input_length = input_length
        self.out_channels = out_channels
        self.iteration = iteration
        self.invoke_gat = nn.ModuleList(
            [ GAT_embed_Conv(input_length, out_channels, edge_dim,gat_heads,dropout=dropout,bias=False,require_edge=True) ]  + 
            [ GAT_embed_Conv(out_channels, out_channels, edge_dim,gat_heads,dropout=dropout,bias=False,require_edge=True) for _ in range(iteration-1) ]
        )

        self.internal_gat = nn.ModuleList(
            [ GAT_embed_Conv(input_length, out_channels, edge_dim,gat_heads,dropout=dropout,bias=False,require_edge=False) ]  + 
            [ GAT_embed_Conv(out_channels, out_channels, edge_dim,gat_heads,dropout=dropout,bias=False,require_edge=False) for _ in range(iteration-1) ]
        )
        
        self.resource_gat = nn.ModuleList(
            [ GAT_embed_Conv(input_length, out_channels, edge_dim,gat_heads,dropout=dropout,bias=False,require_edge=False) ]  + 
            [ GAT_embed_Conv(out_channels, out_channels, edge_dim,gat_heads,dropout=dropout,bias=False,require_edge=False) for _ in range(iteration-1) ]
        )

        self.latent_gat = nn.ModuleList(
            [ GAT_embed_Conv(input_length, out_channels, edge_dim,gat_heads,dropout=dropout,bias=False,require_edge=False) ]  + 
            [ GAT_embed_Conv(out_channels, out_channels, edge_dim,gat_heads,dropout=dropout,bias=False,require_edge=False) for _ in range(iteration-1) ]
        )

        self.attention_net = attention_net(out_channels, out_channels)
        self.dense_layer = nn.Sequential(
            nn.Linear(out_channels, 32),
            nn.GELU(),
            nn.Linear(32,1),
            nn.Sigmoid()
        )
        
    def forward(self,batch):
        '''
        batch:  embedding, x, invoke_edge_index, internal_edge_index, resource_edge_index, invoke_edge_attr, 
        x: [number_nodes, input_length]       input_length = 30(anomaly score) + 24(feature)
        embedding: [number_nodes, embedding_dim]
        '''
        x = batch.x
        embedding = batch.embedding

        g_invoke = x
        g_internal = x
        g_resource = x
        g_latent = x

        for i in range(self.iteration):
            g_invoke_new = self.invoke_gat[i](g_invoke,batch.invoke_edge_index,embedding,edge_attr = batch.invoke_edge_attr)
            g_internal_new = self.internal_gat[i](g_internal,batch.internal_edge_index,embedding,edge_attr = None)       
            g_resource_new = self.resource_gat[i](g_resource,batch.resource_edge_index,embedding,edge_attr = None)
            g_latent_new = self.latent_gat[i](g_latent,batch.latent_edge_index,embedding,edge_attr = None)
            
            if i>=1:
                g_invoke =  (g_invoke +  g_invoke_new) 
                g_internal =  (g_internal +  g_internal_new) 
                g_resource =  (g_resource +  g_resource_new) 
                g_latent =  (g_latent +  g_latent_new)
            else:
                g_invoke = g_invoke_new 
                g_internal = g_internal_new
                g_resource = g_resource_new
                g_latent = g_latent_new
            
       
        g_invoke = g_invoke.unsqueeze(1)
        g_internal = g_internal.unsqueeze(1)
        g_resource = g_resource.unsqueeze(1)
        g_latent = g_latent.unsqueeze(1)

        g = torch.concat((g_invoke,g_internal,g_resource,g_latent),dim=1)
        #g = torch.concat((g_invoke,g_internal),dim=1)
        g = self.attention_net(g)      

        g = self.dense_layer(g)   #[number_nodes,1]
        g = g.squeeze()  #[number_nodes]
        #g = softmax(g,index=batch.batch)

        return g



        
