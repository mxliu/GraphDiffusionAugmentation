import torch
import numpy as np
import torch.nn as nn
from einops import rearrange

class LayerGIN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, epsilon=True):
        super().__init__()
        if epsilon: self.epsilon = nn.Parameter(torch.Tensor([[0.0]])) # assumes that the adjacency matrix includes self-loop 
        else: self.epsilon = 0.0
        self.mlp = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, output_dim), nn.BatchNorm1d(output_dim), nn.ReLU())


    def forward(self, v, a):
        v_aggregate = torch.sparse.mm(a, v) 
        v_aggregate += self.epsilon * v # assumes that the adjacency matrix includes self-loop
        v_combine = self.mlp(v_aggregate)
        return v_combine


#three readout operations
class ModuleMeanReadout(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x, node_axis=1):
        return x.mean(node_axis), torch.zeros(size=[1,1,1], dtype=torch.float32) 


class ModuleSERO(nn.Module):
    def __init__(self, hidden_dim, input_dim, dropout=0.1, upscale=1.0):
        super().__init__()
        self.embed = nn.Sequential(nn.Linear(hidden_dim, round(upscale*hidden_dim)), nn.BatchNorm1d(round(upscale*hidden_dim)), nn.GELU()) 
        self.attend = nn.Linear(round(upscale*hidden_dim), input_dim) 
        self.dropout = nn.Dropout(dropout) 


    def forward(self, x, node_axis=1):
        # assumes shape [... x node x ... x feature]
        x_readout = x.mean(node_axis)
        x_shape = x_readout.shape
        x_embed = self.embed(x_readout.reshape(-1,x_shape[-1])) 
        x_graphattention = torch.sigmoid(self.attend(x_embed)).view(*x_shape[:-1],-1) #
        permute_idx = list(range(node_axis))+[len(x_graphattention.shape)-1]+list(range(node_axis,len(x_graphattention.shape)-1))
        x_graphattention = x_graphattention.permute(permute_idx) 
        return (x * self.dropout(x_graphattention.unsqueeze(-1))).mean(node_axis), x_graphattention.permute(1,0,2)


class ModuleGARO(nn.Module):
    def __init__(self, hidden_dim, dropout=0.1, upscale=1.0, **kwargs):
        super().__init__()
        self.embed_query = nn.Linear(hidden_dim, round(upscale*hidden_dim))
        self.embed_key = nn.Linear(hidden_dim, round(upscale*hidden_dim))
        self.dropout = nn.Dropout(dropout)


    def forward(self, x, node_axis=1):
        # assumes shape [... x node x ... x feature]
        x_q = self.embed_query(x.mean(node_axis, keepdims=True))
        x_k = self.embed_key(x)
        x_graphattention = torch.sigmoid(torch.matmul(x_q, rearrange(x_k, 't b n c -> t b c n'))/np.sqrt(x_q.shape[-1])).squeeze(2)
        return (x * self.dropout(x_graphattention.unsqueeze(-1))).mean(node_axis), x_graphattention.permute(1,0,2)





class GIN(nn.Module):
    def __init__(self, input_dim, hidden_dim,  num_layers, cls_token='sum', readout='mean'):
        super().__init__()
        assert cls_token in ['sum', 'mean', 'param']
        if cls_token=='sum': self.cls_token = lambda x: x.sum(0) 
        elif cls_token=='mean': self.cls_token = lambda x: x.mean(0)
        elif cls_token=='param': self.cls_token = lambda x: x[-1]
        else: raise
        if readout=='garo': readout_module = ModuleGARO
        elif readout=='sero': readout_module = ModuleSERO
        elif readout=='mean': readout_module = ModuleMeanReadout
        else: raise
    

        # define modules
       
        self.initial_linear = nn.Linear(input_dim, hidden_dim)
        self.gnn_layers = nn.ModuleList()
        self.readout_modules = nn.ModuleList()
        
        
        for i in range(num_layers):
            self.gnn_layers.append(LayerGIN(hidden_dim, hidden_dim, hidden_dim))
            self.readout_modules.append(readout_module(hidden_dim=hidden_dim, input_dim=input_dim, dropout=0.1))
            

    def _collate_adjacency(self, a):
           i_list = []
           v_list = []
           for sample, _dyn_a in enumerate(a):
               for timepoint, _a in enumerate(_dyn_a):
                   #parameter is set to 0: sparsity is not required
                   thresholded_a = (_a > np.percentile(_a.detach().cpu().numpy(),0))
                   _i = thresholded_a.nonzero(as_tuple=False)
                   _v = torch.ones(len(_i))
                   _i += sample * a.shape[1] * a.shape[2] + timepoint * a.shape[2]
                   i_list.append(_i)
                   v_list.append(_v)
           _i = torch.cat(i_list).T.to(a.device)
           _v = torch.cat(v_list).to(a.device)

           return torch.sparse.FloatTensor(_i, _v, (a.shape[0]*a.shape[1]*a.shape[2], a.shape[0]*a.shape[1]*a.shape[3]))
   
    def forward(self, v1, a1):
    
        minibatch_size, num_timepoints, num_nodes = a1.shape[:3]
        
        h = v1
        h = rearrange(h, 'b t n c -> (b t n) c')
        h = self.initial_linear(h)
        a1 = self._collate_adjacency(a1)
        for layer, (G, R) in enumerate(zip(self.gnn_layers, self.readout_modules)):
            h = G(h, a1)
            
            h_bridge = rearrange(h, '(b t n) c -> t b n c', t=num_timepoints, b=minibatch_size, n=num_nodes)
            #node-level feature
            feature=torch.squeeze(h_bridge)
            h_readout, node_attn = R(h_bridge, node_axis=2)
            
           
        return feature,h_readout

