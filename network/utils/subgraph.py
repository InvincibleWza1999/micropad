from typing import List, Optional, Tuple, Union
import torch
from torch import Tensor
import copy
import torch_geometric
from torch_geometric.utils import subgraph

def get_subgraph(batch, subset):

    #data = torch_geometric.data.Data()
    data = copy.copy(batch)
    if subset.dtype == torch.bool:
        num_nodes = int(subset.sum())
    else:
        num_nodes = subset.size(0)
    
    edge_mask = None
    if 'edge_index' in batch.keys:
        edge_index, _, edge_mask = subgraph(subset,batch['edge_index'],num_nodes=batch.num_nodes,return_edge_mask=True,relabel_nodes=True)

    for key, value in batch:
        if key == 'invoke_edge_index':
            data[key], data['invoke_edge_attr'] = subgraph(subset,batch[key],batch['invoke_edge_attr'],num_nodes=batch.num_nodes,relabel_nodes=True)
        elif key == 'edge_index':
            data.edge_index = edge_index
        elif 'index' in key:
            data[key], _= subgraph(subset,batch[key],num_nodes=batch.num_nodes,return_edge_mask=False,relabel_nodes=True)    
        elif key == 'invoke_edge_attr':
            pass
        elif key == 'num_nodes':
            data.num_nodes = num_nodes
        elif isinstance(value, Tensor):
            if batch.is_node_attr(key):
                data[key] = value[subset]
            elif batch.is_edge_attr(key):
                data[key] = value[edge_mask]

    return data

def get_subgraph_v2(batch, subset):

    #data = torch_geometric.data.Data()
    data = copy.copy(batch)
    if subset.dtype == torch.bool:
        num_nodes = int(subset.sum())
    else:
        num_nodes = subset.size(0)
    
    edge_mask = None
    if 'edge_index' in batch.keys:
        edge_index, _, edge_mask = subgraph(subset,batch['edge_index'],num_nodes=batch.num_nodes,return_edge_mask=True,relabel_nodes=True)

    for key, value in batch:
        if key == 'invoke_edge_index':
            data[key], data['invoke_edge_attr'] = subgraph(subset,batch[key],batch['invoke_edge_attr'],num_nodes=batch.num_nodes,relabel_nodes=True)
        elif key == 'edge_index':
            data.edge_index = edge_index
        elif 'edge_index' in key:
            data[key], _= subgraph(subset,batch[key],num_nodes=batch.num_nodes,return_edge_mask=False,relabel_nodes=True)    
        elif key == 'invoke_edge_attr':
            pass
        elif key == 'num_nodes':
            data.num_nodes = num_nodes
        elif '_x' in key:
            type_x = key.split('_')[0]
            data[key] = data[key][torch.isin(batch[type_x + '_node_index'],subset),:,:]
        elif 'node_index' in key:
            data[key] = torch.masked_select(batch[key],torch.isin(batch[key],subset))
        elif isinstance(value, Tensor):
            if batch.is_node_attr(key):
                data[key] = value[subset]
            elif batch.is_edge_attr(key):
                data[key] = value[edge_mask]

    return data

    