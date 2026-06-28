import torch
import torch.nn as nn
import torch.nn.init as init
import torch.distributions as dist


class Transformer_Trajectory_Forecast(nn.Module):
    """Transformer encoder + MDN head for 3D trajectory forecasting.
    """
    
    def __init__(self, cfg):
        '''
        The forward method takes an input tensor x of shape
        [train_batch_size, sequence_length, transformer_input_shape].
        For the 3D drone case, transformer_input_shape is expected to be 6:
        [ego_x, ego_y, ego_z, v_ego_x, v_ego_y, v_ego_z].
        The output tensor has shape
        [train_batch_size, forecast_horizon, num_gaussians * 10].
        '''
        super(Transformer_Trajectory_Forecast, self).__init__()
        
        self.transformer_input_shape = cfg['transformer_input_shape']
        self.d_model = cfg['transformer_d_model']
        self.transformer_num_layers = cfg['transformer_num_layers']
        self.transformer_nhead = cfg.get('transformer_nhead', 4)
        self.transformer_dim_feedforward = cfg.get('transformer_dim_feedforward', self.d_model * 4)
        self.transformer_dropout = cfg.get('transformer_dropout', 0.1)
        self.max_seq_len = cfg.get('max_input_horizon', 64)
        self.output_size = cfg['num_gaussians']*cfg['output_factor']
        self.forecast_horizon = cfg['forecast_horizon']
        
        if self.d_model % self.transformer_nhead != 0:
            
            raise ValueError("transformer_d_model must be divisible by transformer_nhead")
        
        self.input_projection = nn.Linear(self.transformer_input_shape, self.d_model)
        self.pos_embedding = nn.Parameter(torch.zeros(1, self.max_seq_len, self.d_model))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.transformer_nhead,
            dim_feedforward=self.transformer_dim_feedforward,
            dropout=self.transformer_dropout,
            batch_first=True,
            norm_first=True
        )
        
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=self.transformer_num_layers
        )
        
        # output layer
        self.fc = nn.Linear(in_features=self.d_model, out_features=self.output_size*self.forecast_horizon)
        
        # initialize the weights of the projection, positional embedding, and output layer
        init.xavier_uniform_(self.input_projection.weight)
        init.zeros_(self.input_projection.bias)
        init.normal_(self.pos_embedding, mean=0.0, std=0.02)
        init.xavier_uniform_(self.fc.weight)
        init.zeros_(self.fc.bias)
        
        
    def forward(self, x):
        """Forward path.
        """
        
        # x shape: [train_batch_size, sequence_length, transformer_input_shape]
        sequence_length = x.size(1)
        
        if sequence_length > self.max_seq_len:
            
            raise ValueError("Input sequence length exceeds max_seq_len")
        
        transformer_out = self.input_projection(x)
        transformer_out = transformer_out + self.pos_embedding[:, :sequence_length, :]
        transformer_out = self.transformer_encoder(transformer_out)
        
        # output shape: [train_batch_size, output_size*forecast_horizon]
        output = self.fc(transformer_out[:, -1, :])
        
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
