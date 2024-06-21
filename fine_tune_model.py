import torch.nn as nn
from GIN_encoder import GIN
import torch.nn.functional as F



class Task_specific_model(nn.Module):
    """
    Build a task_specific model.
    """
    def __init__(self, GIN_input_dim,GIN_hidden_dim,GIN_num_layers,dim, pred_dim):
        """
        dim: feature dimension (default: 64)
        pred_dim: hidden dimension of the predictor (default: 32) 
        """
        super(Task_specific_model, self).__init__()

        # create the GIN encoder
        self.encoder=GIN(input_dim=GIN_input_dim, 
                         hidden_dim=GIN_hidden_dim, 
                         num_layers=GIN_num_layers,
                         cls_token='sum',
                         readout='mean')
    
        
        # build a 2-layer projector
        self.projector = nn.Sequential()
        self.projector.add_module('L1',nn.Linear(dim, pred_dim,bias=False)),
        self.projector.add_module('BN',nn.BatchNorm1d(pred_dim)),
        self.projector.add_module('RL',nn.ReLU(inplace=True)), 
        self.projector.add_module('L2',nn.Linear(pred_dim, 2)) 
        
    def forward(self, X,E):
        """
        Input:
            X: input node feature  size:(bs,1,n,n)
            E: input adjacency matrix size:(bs,1,n,n)
            No augmentation required.
        Output:
            feature: node-level feature  size:(bs,n,n)
            z: GIN encoder output  size:(bs,64)
            output: task_specific_model output  size:(bs,2)
        """

        # compute features 
        feature,z = self. encoder(X,E) 

        z = z.view(-1,64)

        p = self.projector(z)
       
        output = F.softmax(p, dim=1)
    
        return feature,z,output