import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Parameter
from torch_geometric.data.lightning_datamodule import kwargs_repr
from torch_sparse import SparseTensor, set_diag

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear

from torch_geometric.utils import add_self_loops, remove_self_loops, softmax

from torch_geometric.nn.inits import glorot, zeros

class GAT_embed_Conv(MessagePassing):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        edge_dim: int,
        heads: int = 1,
        concat: bool = False,
        negative_slope: float = 0.2,
        dropout: float = 0.0,
        bias: bool = True,
        require_edge: bool = True,
        embed_dim:int=64,
        **kwargs,
    ):
        super(GAT_embed_Conv,self).__init__(aggr='add', node_dim=0,  **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.add_self_loops = add_self_loops
        self.edge_dim = edge_dim
        self.embed_dim = embed_dim
        self.require_edge = require_edge
        self.relu = torch.nn.ReLU()
        self.fill_value = 'mean'
        self._alpha = None

        self.lin = Linear(in_channels, heads * out_channels, bias=False, weight_initializer='glorot')
        self.att = Parameter(torch.empty(1, heads, out_channels + embed_dim))
        

        if require_edge:
            self.lin_edge = Linear(self.edge_dim, heads * (out_channels + embed_dim), bias=False,
                                   weight_initializer='glorot')
    
        else:
            self.lin_edge = None
            

        if bias and concat:
            self.bias = Parameter(torch.Tensor(heads * out_channels))
        elif bias and not concat:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin.reset_parameters()
        if self.lin_edge is not None:
            self.lin_edge.reset_parameters()
        glorot(self.att)
        zeros(self.bias)

    def forward(self, x, edge_index, embedding: Tensor, return_attention_weights = False, edge_attr = None):

        if isinstance(x, Tensor):
            assert x.dim() == 2, "Static graphs not supported in 'GATConv'"
            x = self.lin(x)
            x = (x, x)
        else:  # Tuple of source and target node features:
            x = (self.lin(x[0]), self.lin(x[1]))
        
        edge_index, edge_attr = remove_self_loops(
                    edge_index, edge_attr)
        edge_index, edge_attr = add_self_loops(
                    edge_index, edge_attr, fill_value=self.fill_value,num_nodes=x[0].size(0))

        out = self.propagate(edge_index = edge_index, x=x, embedding=embedding, edge_attr = edge_attr)

        if self.concat:
            out = out.view(-1, self.heads * (self.embed_dim + self.out_channels))    #out:[num_nodes,heads,out_channels]
        else:
            out = out.mean(dim=1)

        if self.bias is not None:
            out += self.bias

        out = self.relu(out)

        if return_attention_weights:
            return out, (edge_index, self._alpha)
        else:
            return out
       

    def message(self, x_i: Tensor,x_j: Tensor, embedding_i, embedding_j, edge_attr, edge_index_i) -> Tensor:    #x_i: [E, self.heads * self.out_channels]
        x_i = x_i.view(-1, self.heads, self.out_channels)
        x_j = x_j.view(-1, self.heads, self.out_channels)

        embedding_i = embedding_i.unsqueeze(1).repeat(1,self.heads,1)   #[E, self.heads, embed_dim]
        embedding_j = embedding_j.unsqueeze(1).repeat(1,self.heads,1)

        cat_x_i = torch.cat((x_i, embedding_i), dim=-1)  #[E, self.heads, embed_dim + self.out_channels]
        cat_x_j = torch.cat((x_j, embedding_j), dim=-1)

        x = cat_x_i + cat_x_j


        if edge_attr is not None:
            edge_attr = self.lin_edge(edge_attr)
            edge_attr = edge_attr.view(-1, self.heads, self.out_channels + self.embed_dim)
            x = x + edge_attr

        x = F.leaky_relu(x, self.negative_slope)  #[E, self.heads, embed_dim + self.out_channels]
        alpha = (x * self.att).sum(dim=-1)    #att: [E, self.heads, embed_dim + self.out_channels]
        alpha = softmax(alpha, index = edge_index_i)
        self._alpha = alpha
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        return alpha.unsqueeze(-1) * x_j

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, heads={self.heads})')
