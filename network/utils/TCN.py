import torch
import torch.nn as nn 
from torch.nn import GLU,Conv1d

class TCN(nn.Module):
    def __init__(self,in_channels = 12,out_channels = 32,kernel_size = 7,dropout = 0.2, input_len = 30):
        super(TCN,self).__init__()
        self.in_channels = in_channels
        self.out_channels= out_channels
        self.kernel_size = kernel_size
        self.dropout = dropout
        self.input_len = input_len
    
        self.conv_1 = Conv1d(in_channels= self.in_channels,out_channels=self.out_channels,kernel_size=self.kernel_size,padding=0)    ##输出给GLU单元[batch_num,out_channels,input_len-kernel_size+1]
        self.conv_2 = Conv1d(in_channels= self.in_channels,out_channels=self.out_channels,kernel_size=self.kernel_size,padding=0)
        self.glu = GLU(dim=1) 
        self.dropout = nn.Dropout(self.dropout)
        self.liner = nn.Linear(self.out_channels, 1)
        self.relu = nn.ReLU()


    def forward(self,data):   #data: [batch_num,in_channels,input_len]
        h_1 = self.conv_1(data.to(torch.float32))  #[batch_num,out_channels,input_len-kernel_size+1]
        h_2 = self.conv_2(data.to(torch.float32))  #[batch_num,out_channels,input_len-kernel_size+1]

        h = torch.cat((h_1,h_2),dim = 1)   #[batch_num,2 * out_channels,input_len-kernel_size+1]
        h = self.dropout(h)
        h = self.glu(h)   #[batch_num, out_channels,input_len-kernel_size+1]

        
        h = h.permute(0,2,1)  #[batch_num,input_len-kernel_size+1,out_channels]
        h = self.liner(h)     #[batch_num,input_len-kernel_size+1,1]
        h = self.relu(h)
        h = h.squeeze(dim=-1)  #[batch_num,input_len-kernel_size+1]

        return h