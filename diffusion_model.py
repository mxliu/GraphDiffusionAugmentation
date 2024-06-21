import torch
import torch.nn as nn
import torch.nn.functional as F
from transformer_model import GraphTransformer
from noise_schedule import DiscreteUniformTransition, PredefinedNoiseScheduleDiscrete 
from extra_features import ExtraFeatures
from extra_features1 import ExtraFeatures1
from dataset import X1,E1,y
import diffusion_utils


class DiscreteDenoisingDiffusion(nn.Module):
    def __init__(self, device, diffusion_steps, n_layers, hidden_mlp_dims, hidden_dims, diffusion_noise_schedule):  
        super().__init__()
        
        self.T = diffusion_steps
        #ExtraFeatures1: specifically to calculate dimensions
        self.extra_features1 = ExtraFeatures1(extra_features_type='all')
        
        self.extra_features = ExtraFeatures(extra_features_type='all')


        self.input_dims = {'X': X1.size(-1),
                           'E': E1.size(-1),
                           'y': y.size(1) + 1}  # + 1 due to time conditioning                
        ex_extra_feat = self.extra_features1(X1,E1,y)
        self.input_dims['X'] += ex_extra_feat.X.size(-1)
        self.input_dims['E'] += ex_extra_feat.E.size(-1)
        self.input_dims['y'] += ex_extra_feat.y.size(-1)


        self.output_dims = {'X': X1.size(-1),
                            'E': E1.size(-1),
                            'y': 0}   
        self.Xdim_output = self.output_dims['X']
        self.Edim_output = self.output_dims['E']
        self.ydim_output = self.output_dims['y']
        
      
        
        #denoising neural network
        self.model = GraphTransformer(n_layers=n_layers,
                                      input_dims=self.input_dims,
                                      hidden_mlp_dims=hidden_mlp_dims,
                                      hidden_dims=hidden_dims,
                                      output_dims=self.output_dims,
                                      act_fn_in=nn.ReLU(),
                                      act_fn_out=nn.ReLU())
        
        #alpha_s_bar in transition matrix 
        self.noise_schedule = PredefinedNoiseScheduleDiscrete(diffusion_noise_schedule,
                                                              timesteps=diffusion_steps)
        #transition matrix in noise unit
        self.transition_model = DiscreteUniformTransition(x_D=self.Xdim_output, 
                                                          e_D=self.Edim_output,
                                                          y_D=self.ydim_output)

    

    def apply_noise(self, X, E, y):
        """ Sample noise and apply it to the data. """

        # Sample a timestep t.
        # When evaluating, the loss for t=0 is computed separately
        #lowest_t = 0 if self.training else 1
        lowest_t = 0 
        t_int = torch.randint(lowest_t, self.T + 1, size=(X.size(0), 1), device=X.device).float()  # (bs, 1)
        s_int = t_int - 1

        t_float = t_int / self.T
        s_float = s_int / self.T

        # beta_t and alpha_s_bar are used for denoising/loss computation
        beta_t = self.noise_schedule(t_normalized=t_float)                         # (bs, 1)
        alpha_s_bar = self.noise_schedule.get_alpha_bar(t_normalized=s_float)      # (bs, 1)
        alpha_t_bar = self.noise_schedule.get_alpha_bar(t_normalized=t_float)      # (bs, 1)

        Qtb = self.transition_model.get_Qt_bar(alpha_t_bar, device=X.device)       # (bs, dx_in, dx_out), (bs, de_in, de_out)
        
        assert (abs(Qtb.X.sum(dim=2) - 1.) < 1e-4).all(), Qtb.X.sum(dim=2) - 1
        assert (abs(Qtb.E.sum(dim=2) - 1.) < 1e-4).all()
       
        probX = X @ Qtb.X  # (bs, n, dx_out)
       
        
        probE = E @ Qtb.E.unsqueeze(1)  # (bs, n, n, de_out)
        probE[torch.sum(probE, dim=-1) == 0] = 1e-5
        
        
        E_t = diffusion_utils.sample_discrete_features(probE=probE)
        E_t = F.one_hot(E_t, num_classes=self.Edim_output)
        
        assert (X.shape == probX.shape) and (E.shape == E_t.shape)


        noisy_data = {'t_int': t_int, 't': t_float, 'beta_t': beta_t, 'alpha_s_bar': alpha_s_bar,
                      'alpha_t_bar': alpha_t_bar, 'X_t': probX, 'E_t':E_t , 'y_t': y}
        return noisy_data
      
    
    
    
    def compute_extra_data(self, noisy_data):
        """ At every training step (after adding noise) and step in sampling, compute extra information and append to
            the network input. """

        extra_features = self.extra_features(noisy_data)

        t = noisy_data['t']
        extra_y = torch.cat((extra_features.y, t), dim=1)

        return diffusion_utils.PlaceHolder(X=extra_features.X, E=extra_features.E, y=extra_y)
    
    

    
    
    def forward(self, noisy_data, extra_data):
        """ The extra_data  and the main features are concatenated as the input of the denoising neural network, 
            the pred is compared with the original data to obtain the loss for updating the denoising neural network, 
            the trained denoising neural network is used for graph generation. """
        
        X_e = torch.cat((noisy_data['X_t'], extra_data.X), dim=-1).float()
      
       
      
        E_e = torch.cat((noisy_data['E_t'], extra_data.E), dim=-1).float()
       
      
        y_e = torch.hstack((noisy_data['y_t'], extra_data.y)).float()
       
        pred=self.model(X_e, E_e, y_e)
        
        return pred
   
    
