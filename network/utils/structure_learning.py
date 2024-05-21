from torch_scatter import scatter_max
import torch
from torch.sparse import *

def drop_duplicate_edges(edge_index,edge_attr):

    '''
    edge_index: [2,number_edges]
    edge_attr: [number_edges,edge_feature]
    '''

    index_list = edge_index.T.tolist()  #[number_edges,2]
    idx = torch.tensor(list(map(lambda x:index_list.index(x),torch.unique(edge_index.T, dim=0).tolist()))) 
    edge_index = edge_index[:,idx]
    if not edge_attr is None: 
        edge_attr = edge_attr[idx,:]

    return edge_index,edge_attr

def structure_learning(input,edge_index,edge_attr,k,device):

    '''
    input: [number_nodes, feature_dim]
    edge_index: [2,number_edges]
    edge_attr: [number_edges, edge_attr_dim]
    k: int/float
    '''

    weight = input.detach().clone()   
    E =  torch.matmul(weight,weight.T) / torch.matmul(weight.norm(dim=1,keepdim=True),weight.norm(dim=1,keepdim=True).T)
    #adj_matrix = torch.sparse_coo_tensor(edge_index,size=(weight.size(0),weight.size(0)),values=torch.ones(weight.size(1))).to_dense()

    #edge_index,edge_attr = drop_duplicate_edges(edge_index,edge_attr)

    E = torch.index_select(E,0,edge_index[0])   #[number_edges,number_nodes]
    E = E[torch.arange(E.size(0)),edge_index[1]]    #[number_edges]
    learned_edge_index = torch.tensor([],dtype=torch.long).to(device)
    if not edge_attr is None:
        learned_edge_attr = torch.tensor([],dtype=torch.float32).to(device)
    else:
        learned_edge_attr = None

    for item in edge_index[1].unique():
        index_of_item = torch.nonzero(edge_index[1] == item).squeeze()
        if index_of_item.shape == torch.Size([]):
            len_i = 1
        else:
            len_i = len(index_of_item)
        if isinstance(k,float):
            assert 0<k<=1,"float topk larger than 1"
            _,max_edge_index = E[index_of_item].topk(k=max(int(k*len_i),1))
        elif isinstance(k,int):
            _,max_edge_index = E[index_of_item].topk(k=min(k,len_i))

        if len_i==1:
            max_edge = edge_index[:,index_of_item].unsqueeze(0).T
        else:
            max_edge = edge_index[:,index_of_item][:,max_edge_index]

        if not edge_attr is None:
            if len_i==1:
                max_edge_attr = edge_attr[index_of_item].unsqueeze(0)
            else:
                max_edge_attr = edge_attr[index_of_item][max_edge_index]
            learned_edge_attr = torch.concat((learned_edge_attr,max_edge_attr.to(torch.float32)),dim = 0)
        learned_edge_index = torch.concat((learned_edge_index,max_edge),dim = 1)
            
    
    return learned_edge_index, learned_edge_attr