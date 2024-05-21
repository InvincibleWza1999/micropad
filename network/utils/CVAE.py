import torch
import torch.nn as nn

class prior_network(torch.nn.Module):
    def __init__(self,input_size:int,latent_size:int,dropout = 0.1):
        super().__init__()
        self.input_size = input_size
        self.latent_size = latent_size
        self.prior_z_mu = nn.Sequential(
            nn.Linear(self.input_size,128),
            nn.ReLU(),
            nn.Linear(128,128),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(128,self.latent_size)
        )
        self.prior_z_var = nn.Sequential(
            nn.Linear(self.input_size,128),
            nn.ReLU(),
            nn.Linear(128,128),
            nn.ReLU(),
            nn.Linear(128,self.latent_size),
            nn.Dropout(p=dropout),
            nn.Softplus()
        )
    
    def forward(self,x):
        mu,var = self.prior_z_mu(x),self.prior_z_var(x)
        return mu,var
    

class VAE_encoder(nn.Module):
    def __init__(self,input_size:int,latent_size:int,dropout = 0.1) -> None:
        super().__init__()
        self.input_size = input_size
        self.latent_size = latent_size
        self.encode_mu = nn.Sequential(
            nn.Linear(self.input_size,256),  
            nn.ReLU(),
            nn.Linear(256,256),
            nn.ReLU(),
            nn.Linear(256,self.latent_size),
            nn.Dropout(p=dropout),
            nn.ReLU()
        )

        self.encode_var = nn.Sequential(
            nn.Linear(self.input_size,256),
            nn.ReLU(),
            nn.Linear(256,256),
            nn.ReLU(),
            nn.Linear(256,self.latent_size),
            nn.Dropout(p=dropout),
            nn.Softplus()
        )

    def forward(self,x):
        mu,var = self.encode_mu(x),self.encode_var(x)
        return mu,var

class VAE_decoder(nn.Module):
    def __init__(self,input_length,output_dim,output_length = 30):
        super().__init__()
        self.input_length = input_length
        self.output_dim = output_dim
        self.decoder = nn.Conv1d(
            in_channels = 1,
            out_channels = output_dim * 2,
            kernel_size =  input_length - output_length + 1
        )
        self.Softplus = nn.Softplus()
        self.bn_mu = nn.BatchNorm1d(num_features=output_dim)
        #print(input_length,output_dim,input_length - output_length + 1)
        #self.bn_var = nn.BatchNorm1d(num_features=output_dim)

    def forward(self,x,with_bn=False):      #x: [num_samples * num_nodes, 1, input_size]
        rec_mu,rec_var = self.decoder(x).chunk(2,dim=1)
        rec_var = self.Softplus(rec_var)
        if with_bn:
            rec_mu = self.bn_mu(rec_mu)
            #rec_var = self.bn_var(rec_var)
        #print(rec_mu.shape,rec_var.shape)
        return rec_mu,rec_var

class AE_encoder(nn.Module):
    def __init__(self,input_size:int,latent_size:int,dropout = 0.1) -> None:
        super().__init__()
        self.input_size = input_size
        self.latent_size = latent_size
        self.encode_mu = nn.Sequential(
            nn.Linear(self.input_size,256),  
            nn.ReLU(),
            nn.Linear(256,256),
            nn.ReLU(),
            nn.Linear(256,self.latent_size),
            nn.Dropout(p=dropout),
            nn.ReLU()
        )

    def forward(self,x):
        z = self.encode_mu(x)
        return z
    

class AE_decoder(nn.Module):
    def __init__(self,input_length,output_dim,output_length = 30):
        super().__init__()
        self.input_length = input_length
        self.output_dim = output_dim
        self.decoder = nn.Conv1d(
            in_channels = 1,
            out_channels = output_dim,
            kernel_size =  input_length - output_length + 1
        )
        self.Softplus = nn.Softplus()
        self.bn = nn.BatchNorm1d(num_features=output_dim)

    def forward(self,x, with_bn= False):      #x: [num_samples * num_nodes, 1, input_size]
        rec = self.decoder(x)
        if with_bn:
            rec = self.bn(rec)
        return rec