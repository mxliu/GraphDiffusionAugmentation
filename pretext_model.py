import torch.nn as nn
from diffusion_model import DiscreteDenoisingDiffusion
from GIN_encoder import GIN
import torch
import diffusion_utils
from noise_schedule import DiscreteUniformTransition, PredefinedNoiseScheduleDiscrete 
import torch.nn.functional as F
from einops import repeat


class Pretext_model(nn.Module):
    """
    Build a pretext model.
    """
    def __init__(self, device, T, diffusion_hidden_mlp_dims, diffusion_hidden_dims, 
                       diffusion_output_dims, diffusion_num_layers, 
                       GIN_input_dim, GIN_hidden_dim, GIN_num_layers, 
                       projector_input_dim, projector_hidden_dim):
        
        super(Pretext_model, self).__init__()
        self.device=device
        self.T=T
        
        #Used for graph augmentation
        self.diffusion_model=DiscreteDenoisingDiffusion(device, diffusion_steps=T,
                                                        n_layers=diffusion_num_layers, 
                                                        hidden_mlp_dims=diffusion_hidden_mlp_dims,
                                                        hidden_dims=diffusion_hidden_dims,
                                                        diffusion_noise_schedule='cosine'  )
        
        #Used for feature extraction
        self.base_encoder=GIN(input_dim=GIN_input_dim, 
                         hidden_dim=GIN_hidden_dim, 
                         num_layers=GIN_num_layers,
                         cls_token='sum',
                         readout='mean')
    
       
        # MLP projector
        self.projector = nn.Sequential()
        self.projector.add_module('L1',nn.Linear(projector_input_dim, projector_hidden_dim,bias=False)),
        self.projector.add_module('BN',nn.BatchNorm1d(projector_hidden_dim)),
        self.projector.add_module('RL',nn.ReLU(inplace=True)),
        self.projector.add_module('L2',  nn.Linear(projector_hidden_dim, projector_input_dim)) # output layer
        
        
        
        
        
   
        Xdim_output = diffusion_output_dims['X']
        self.Edim_output = diffusion_output_dims['E']
        ydim_output = diffusion_output_dims['y']
        
        
        
        self.noise_schedule = PredefinedNoiseScheduleDiscrete(noise_schedule='cosine',timesteps=T)
        self.transition_model = DiscreteUniformTransition(x_D=Xdim_output, e_D=self.Edim_output,
                                                     y_D=ydim_output)
    
    
    def sample_p_zs_given_zt(self, s, t, X_t, E_t, y_t):
       """Samples from zs ~ p(zs | zt). Only used during sampling.
          if last_step, return the graph prediction as well"""
       bs, n, dxs = X_t.shape
       beta_t = self.noise_schedule(t_normalized=t)  # (bs, 1)
       alpha_s_bar = self.noise_schedule.get_alpha_bar(t_normalized=s)
       alpha_t_bar = self.noise_schedule.get_alpha_bar(t_normalized=t)
    
       # Retrieve transitions matrix
       Qtb = self.transition_model.get_Qt_bar(alpha_t_bar, X_t.device)
       Qsb = self.transition_model.get_Qt_bar(alpha_s_bar, X_t.device)
       Qt = self.transition_model.get_Qt(beta_t, X_t.device)
    
       # Neural net predictions
       noisy_data = {'X_t': X_t, 'E_t': E_t, 'y_t': y_t, 't': t}
       extra_data = self.diffusion_model.compute_extra_data(noisy_data)
       
       pred=self.diffusion_model(noisy_data,extra_data)
    
       # Normalize predictions
       pred_X = F.softmax(pred.X, dim=-1)               # bs, n, d0
       pred_E = F.softmax(pred.E, dim=-1)               # bs, n, n, d0 
       
       #Posterior probability
       p_s_and_t_given_0_X = diffusion_utils.compute_batched_over0_posterior_distribution(X_t=X_t,
                                                                                          Qt=Qt.X,
                                                                                          Qsb=Qsb.X,
                                                                                          Qtb=Qtb.X)
    
       p_s_and_t_given_0_E = diffusion_utils.compute_batched_over0_posterior_distribution(X_t=E_t,
                                                                                          Qt=Qt.E,
                                                                                          Qsb=Qsb.E,
                                                                                          Qtb=Qtb.E)
       # Dim of these two tensors: bs, N, d0, d_t-1
       
       #bs,n,d0,1  *   bs,n,d0,d_t-1
       weighted_X = pred_X.unsqueeze(-1) * p_s_and_t_given_0_X         # bs, n, d0, d_t-1
       unnormalized_prob_X = weighted_X.sum(dim=2)                     # bs, n, d0
       unnormalized_prob_X[torch.sum(unnormalized_prob_X, dim=-1) == 0] = 1e-5
       prob_X = unnormalized_prob_X / torch.sum(unnormalized_prob_X, dim=-1, keepdim=True)  
       
       
       pred_E = pred_E.reshape((bs, -1, pred_E.shape[-1]))  
       weighted_E = pred_E.unsqueeze(-1) * p_s_and_t_given_0_E        # bs, N, d0, d_t-1
       unnormalized_prob_E = weighted_E.sum(dim=-2)
       unnormalized_prob_E[torch.sum(unnormalized_prob_E, dim=-1) == 0] = 1e-5
       prob_E = unnormalized_prob_E / torch.sum(unnormalized_prob_E, dim=-1, keepdim=True)
       prob_E = prob_E.reshape(bs, n, n, pred_E.shape[-1])
    
       assert ((prob_X.sum(dim=-1) - 1).abs() < 1e-4).all()
       assert ((prob_E.sum(dim=-1) - 1).abs() < 1e-4).all()
    
       E_s = diffusion_utils.sample_discrete_features(prob_E)
       
       E_s = F.one_hot(E_s, num_classes=self.Edim_output).float()
    
       assert (prob_E == torch.transpose(prob_E, 1, 2)).all()
       assert (X_t.shape == prob_X.shape) and (E_t.shape == E_s.shape)
    
       generation_graph = diffusion_utils.PlaceHolder(X=prob_X, E=E_s, y=torch.zeros(y_t.shape[0], 0))
    
       return generation_graph

    
    
    def forward(self, X1,E1,y):
        """
        Input:
            X1, E1: Original graoh
            y:  global feature
        Output:
            pred: predicted graph in GDA module
            p1, p2, z1, z2: predictors and targets of the network
        """
        
        #Graph diffusion augmentation module
        
        #Calculate something for sample process
        output_dims = {'X': X1.size(-1),'E': E1.size(-1),'y': 0}
   
        Xdim_output = output_dims['X']
        Edim_output = output_dims['E']
        ydim_output = output_dims['y']
             
        x_limit = torch.ones(Xdim_output) / Xdim_output
        e_limit = torch.ones(Edim_output) / Edim_output
        y_limit = torch.ones(ydim_output) / ydim_output
        limit_dist1 = diffusion_utils.PlaceHolder(X=x_limit, E=e_limit, y=y_limit)
        
        
        
        
        
        #train
        X = X1.float()
        E = E1.float()
        y = y.float()
        noisy_data = self.diffusion_model.apply_noise(X, E, y)
        extra_data = self.diffusion_model.compute_extra_data(noisy_data)
    
        pred = self.diffusion_model(noisy_data,extra_data)
        
        #sample
        with torch.no_grad():   
            # Sample noise  -- z has size (n_samples, n_nodes, n_features)
    
            n_nodes = X1.size(1) * torch.ones(E1.size(0), device=self.device, dtype=torch.int)    
            n_max = torch.max(n_nodes).item()  
            # Build the masks
            arange = torch.arange(n_max,device=self.device).unsqueeze(0).expand(X1.size(0), -1)
            node_mask = arange < n_nodes.unsqueeze(1)  #tensor([[True],[True],...,[True]])
            z_T = diffusion_utils.sample_discrete_feature_noise(limit_dist1,node_mask)
            X2, E2, y2 = z_T.X, z_T.E, z_T.y
           
            
            assert (E == torch.transpose(E, 1, 2)).all()
            
            # Iteratively sample p(z_s | z_t) for t = 1, ..., T, with s = t - 1.
            for s_int in reversed(range(0, self.T)):    #T=5, s_int:4,3,2,1,0
                s_array = s_int * torch.ones((z_T.X.size(0), 1)).type_as(z_T.y)
                t_array = s_array + 1
                s_norm = s_array / self.T
                t_norm = t_array / self.T
            
                # Sample z_s
                sampled_s = self.sample_p_zs_given_zt(s_norm, t_norm, X2, E2, y2)
                X2, E2, y = sampled_s.X, sampled_s.E, sampled_s.y
                
            
                
        
        
        X2=repeat(X2,'b t c -> b s t c',s=1) #[bs,1,116,116]
        
        E2=E2[:,:,:,-1:]
        E2 =E2.squeeze(dim=-1)
        E2=repeat(E2,'b t c -> b s t c',s=1) #[bs,1,116,116]
        
        X1=repeat(X1,'b t c -> b s t c',s=1) #[bs,1,116,116]
        E1=E1[:,:,:,-1:]
        E1 =E1.squeeze(dim=-1)
        E1=repeat(E1,'b t c -> b s t c',s=1) #[bs,1,116,116]
        
        
        
        #Graph contrastive learning Module
        
        #Graph feature extraction
        feature,z1 = self. base_encoder(X1,E1) 
        feature,z2 = self. base_encoder(X2,E2) 

        z1 = z1.view(-1,64)
        z2 = z2.view(-1,64)
        
        #MLP projector
        p1 = self.projector(z1) 
        p2 = self.projector(z2) 

        return pred, p1, p2, z1.detach(), z2.detach()