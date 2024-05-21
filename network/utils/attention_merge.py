import torch
import torch.nn as nn

class attention_net(torch.nn.Module):
    def __init__(self,input_dim,middle_dim):
        super().__init__()
        self.liner = nn.Linear(input_dim,middle_dim,bias=True)
        self.U = nn.Linear(middle_dim,1,bias=False)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)

    def forward(self,node_feature):
        '''
        node_feature: [number_nodes, len_args, ggnn_out_channels]  len_args:邻接矩阵个数
        '''

        weight = self.liner(node_feature)
        weight = self.tanh(weight)
        weight = self.U(weight)
        weight = self.softmax(weight)

        return torch.mul(node_feature,weight).sum(dim=1)

        


    
