import torch
import torch.nn as nn
import torch.nn.init as init
import torch.distributions as dist


class LSTM_Trajectory_Forecast(nn.Module):
    """LSTM module class for 3D trajectory forecasting.
    """
    
    def __init__(self, cfg):
        '''
        The forward method takes an input tensor x of shape
        [train_batch_size, sequence_length, lstm_input_shape].
        For the local step-to-step 3D drone case, lstm_input_shape is
        expected to be 3: [sideways_dx, forward_dy, delta_z].
        The output tensor has shape
        [train_batch_size, forecast_horizon, num_gaussians * 10].
        '''
        super(LSTM_Trajectory_Forecast, self).__init__()
        
        self.lstm_input_shape = cfg['lstm_input_shape']
        self.lstm_hidden_size = cfg['lstm_hidden_size']
        self.output_size = cfg['num_gaussians']*cfg['output_factor']
        self.forecast_horizon = cfg['forecast_horizon']
        self.lstm_num_layers = cfg['lstm_num_layers']
        
        # stacked LSTM layer
        self.lstm = nn.LSTM(input_size=self.lstm_input_shape, hidden_size=self.lstm_hidden_size, num_layers=self.lstm_num_layers, batch_first=True)
        
        # output layer
        self.fc = nn.Linear(in_features=self.lstm_hidden_size, out_features=self.output_size*self.forecast_horizon)
        
        # initialize the weights of the LSTM and fully connected layers
        for name, param in self.lstm.named_parameters():
            
            if 'weight' in name:
                
                init.xavier_uniform_(param)
                
            elif 'bias' in name:
                
                init.zeros_(param)
                
        init.xavier_uniform_(self.fc.weight)
        init.zeros_(self.fc.bias)
        
        
    def forward(self, x):
        """Forward path.
        """
        
        # x shape: [train_batch_size, sequence_length, lstm_input_shape]
        lstm_out, _ = self.lstm(x)
        
        # output shape: [train_batch_size, output_size*forecast_horizon]
        output = self.fc(lstm_out[:, -1, :])
        
        # reshape output: [train_batch_size, forecast_horizon, output_size]
        output = torch.reshape(output, (-1,self.forecast_horizon, self.output_size))
        
        return output
    
    
def NLL_MDN_loss(output, target, num_gaussians):
    """Negative log likelihood loss for a 3D mixture density network.
    
    output shape: [batch_size, n_horizons, num_gaussians * 10]
    target shape: [batch_size, n_horizons, 3]
    
    Per Gaussian:
    mu_x, mu_y, mu_z,
    sigma_x, sigma_y, sigma_z,
    rho_xy, rho_xz, rho_yz,
    alpha
    """
    
    train_batch_size = target.size(0)
    forecast_horizon = target.size(1)
    eps = 1e-4
    
    # split the output into the parameters for each Gaussian
    mu_x = output[:,:, :num_gaussians]
    mu_y = output[:,:, num_gaussians:2*num_gaussians]
    mu_z = output[:,:, 2*num_gaussians:3*num_gaussians]
    
    sigma_x = torch.exp(output[:,:, 3*num_gaussians:4*num_gaussians]) + eps
    sigma_y = torch.exp(output[:,:, 4*num_gaussians:5*num_gaussians]) + eps
    sigma_z = torch.exp(output[:,:, 5*num_gaussians:6*num_gaussians]) + eps
    
    rho_xy = torch.tanh(output[:,:, 6*num_gaussians:7*num_gaussians])
    rho_xz = torch.tanh(output[:,:, 7*num_gaussians:8*num_gaussians])
    rho_yz_partial = torch.tanh(output[:,:, 8*num_gaussians:9*num_gaussians])
    rho_yz = rho_xy * rho_xz + torch.sqrt((1 - rho_xy ** 2).clamp_min(eps)) * torch.sqrt((1 - rho_xz ** 2).clamp_min(eps)) * rho_yz_partial
    alpha = torch.softmax(output[:,:, 9*num_gaussians:], dim=-1)
    
    mixture = dist.Categorical(alpha)
    covs = torch.zeros(train_batch_size, forecast_horizon, num_gaussians, 3, 3).to(output.device)
    
    covs[:, :, :, 0, 0] = sigma_x ** 2
    covs[:, :, :, 1, 1] = sigma_y ** 2
    covs[:, :, :, 2, 2] = sigma_z ** 2
    covs[:, :, :, 0, 1] = rho_xy * sigma_x * sigma_y
    covs[:, :, :, 1, 0] = rho_xy * sigma_x * sigma_y
    covs[:, :, :, 0, 2] = rho_xz * sigma_x * sigma_z
    covs[:, :, :, 2, 0] = rho_xz * sigma_x * sigma_z
    covs[:, :, :, 1, 2] = rho_yz * sigma_y * sigma_z
    covs[:, :, :, 2, 1] = rho_yz * sigma_y * sigma_z
    
    # numerical stabilizer for early training covariance matrices
    eye = torch.eye(3, device=output.device)
    covs = covs + eps * eye
    
    try:
        
        gaussians = dist.MultivariateNormal(torch.stack([mu_x, mu_y, mu_z], dim=-1), covs)
        
    except:
        
        return None, True
        
    # compute the negative log likelihood loss
    mixture = dist.MixtureSameFamily(mixture, gaussians)
    params_loss = -mixture.log_prob(target).mean()
    loss = params_loss
    
    return loss, False
