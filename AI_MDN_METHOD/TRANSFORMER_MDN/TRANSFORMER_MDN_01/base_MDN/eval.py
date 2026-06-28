import os
import torch
import time
import numpy as np
import torch.distributions as dist

from vis import plot_aee_over_time, plot_ego_forecast, plot_world_forecast
from vis import plot_reliability_calibration, plot_sharpness_over_time
from utils.helper import ego2world

try:
    
    from termcolor import colored
    
except ModuleNotFoundError:
    
    def colored(text, color=None):
        
        return text


class MDN_Forecaster:
    """Inference and evaluation for 3D MDN.
    """
    
    def __init__(self, cfg, model, data_loader, type, logger, device='cpu'):
        """Init and setup.
        """
        
        self.cfg = cfg
        self.model = model.to(device)
        self.device = device
        self.type = type
        self.logger = logger
        self.train_params = cfg.train_params
        self.test_params = cfg.test_params
        self.model_params = cfg.model_params
        self.eval_metrics = cfg.eval_metrics
        self.num_gaussians = self.model_params['num_gaussians']
        self.dt = self.model_params['delta_t']
        self.forecast_horizon = self.model_params['forecast_horizon']
        self.batch_size = self.test_params['batch_size']
        self.n_samples = self.test_params['num_samples']
        self.eval_input_horizons = self.test_params['num_input_horizons']
        self.reliability_bins = cfg.reliability_bins
        
        # metrics
        self.with_ade_fde_k = self.eval_metrics['ade_fde_k']
        self.with_reliability = self.eval_metrics['reliability']
        self.with_sharpness = self.eval_metrics['sharpness']
        self.with_asaee = self.eval_metrics['asaee']
        
        # forecasting use case
        # eval while training
        if type == 'eval':
            
            self.plot_examples = self.train_params['plot_examples']
            self.plot_step = self.train_params['plot_step']
            self.plot_ego_dst_dir = cfg.eval_ego_examples_path
            self.plot_world_dst_dir = cfg.eval_world_examples_path
            self.data_X, self.data_y, self.data_ref, self.data_rot, self.data_src = data_loader.get_eval_data()
            self.dst_dir = cfg.evaluation_path
        
        # testing a trained model with test data
        elif type == 'testing':
            
            self.plot_examples = self.test_params['plot_examples']
            self.plot_step = self.test_params['plot_step']
            self.plot_ego_dst_dir = cfg.test_ego_examples_path
            self.plot_world_dst_dir = cfg.test_world_examples_path
            self.data_X, self.data_y, self.data_ref, self.data_rot, self.data_src = data_loader.get_test_data()
            self.dst_dir = cfg.testing_path
            
        return
        
        
    def evaluate(self, epoch=None):
        """Evaluate model on 3D trajectory prediction metrics.
        """
        
        self.model.eval()
        aee_sets = []
        min_ade_sets = []
        min_fde_sets = []
        confidence_sets = []
        sharpness_sets = []
        batch_run_time = []
        
        # create mesh grid for 3D ego coordinates
        grid = None
        if self.with_sharpness or self.with_asaee:
            
            grid = self.build_mesh_grid(
                mesh_range_x=self.test_params['mesh_range_x'],
                mesh_range_y=self.test_params['mesh_range_y'],
                mesh_range_z=self.test_params['mesh_range_z'],
                mesh_resolution=self.test_params['mesh_resolution']
            )
        
        with torch.no_grad():
            
            # batch processing
            for iteration in range(0, len(self.data_X), self.batch_size):
                
                # upload data and run model
                inputs = torch.tensor(data=self.data_X[iteration:(iteration+self.batch_size)][:,-self.eval_input_horizons:,:], dtype=torch.float32).to(self.device)
                targets = torch.tensor(data=self.data_y[iteration:(iteration+self.batch_size)], dtype=torch.float32).to(self.device)
                
                st = time.time()
                outputs = self.model(inputs)
                et = time.time()
                
                # measure batch model inference time
                batch_run_time.append((et-st)*1000)
                
                # only use user defined output steps
                filtered_outputs = torch.stack([outputs[:,k,:] for k in self.test_params['test_horizons']], dim=1)
                filtered_targets = torch.stack([targets[:,k,:] for k in self.test_params['test_horizons']], dim=1)
                
                if self.with_ade_fde_k:
                    
                    # get k samples from distributions with probabilities
                    k_samples, k_probs = self.sample_with_probs(filtered_outputs, num_gaussians=self.num_gaussians, n_samples=self.test_params['num_k_samples'])
                    
                    # sort k samples by probabilities
                    _, k_sorted_indices = torch.topk(k_probs, k=self.test_params['num_k_samples'], dim=0)
                    k_sorted_samples = k_samples.gather(dim=0, index=k_sorted_indices.unsqueeze(-1).expand(-1, -1, -1, 3))
                    
                    # get euclidean errors for k samples over defined future time steps of batch
                    euc_errors = torch.sqrt(torch.square(filtered_targets - k_sorted_samples).sum(dim=-1)).cpu().numpy()
                    
                    # get smallest errors of k samples over defined future time steps of batch
                    min_ade_sets.append(euc_errors.mean(axis=2).min(axis=0))
                    
                    # get smallest error of k trajectories over defined future time steps of batch
                    min_fde_sets.append(euc_errors[:,:,-1].min(axis=0))
                
                if self.with_reliability:
                    
                    confidence_sets.append(self.build_confidence_set_mdn(output=filtered_outputs, target=filtered_targets, num_gaussians=self.num_gaussians, n_samples=self.n_samples).cpu().numpy())
                
                if self.with_sharpness or self.with_asaee:
                    
                    sharpness_batch = []
                    modes_batch = []
                    
                    # do for every single sample one by one to keep memory bounded
                    for idx in range(0, filtered_outputs.shape[0]):
                        
                        conf_map = self.build_confidence_set_mdn(output=filtered_outputs[idx][None,...], target=grid, num_gaussians=self.num_gaussians, n_samples=self.n_samples)
                        
                        if self.with_sharpness:
                            
                            sharpness = []
                            
                            for k in self.test_params['confidence_levels']:
                                
                                sharpness.append(
                                    self.estimate_sharpness(confidence_map=conf_map, kappa=k)
                                    * (2 * self.test_params['mesh_range_x'])
                                    * (2 * self.test_params['mesh_range_y'])
                                    * (2 * self.test_params['mesh_range_z'])
                                )
                            
                            sharpness_batch.append(torch.stack(sharpness, 0))
                        
                        if self.with_asaee:
                            
                            # original repo style: mode is the most likely grid point
                            modes_batch.append(grid[torch.argmin(conf_map, dim=0, keepdim=True)][0,:,0,:])
                    
                    if self.with_sharpness:
                        
                        sharpness_sets.append(torch.stack(sharpness_batch, 0).cpu().numpy())
                    
                    if self.with_asaee:
                        
                        modes = torch.stack(modes_batch, 0)
                        aee_sets.append(torch.sqrt(torch.square(filtered_targets - modes).sum(dim=-1)).cpu().numpy())
            
            # average model inference time
            if self.cfg.with_print: print(colored(f"====================", 'magenta'))
            if self.cfg.with_print: print(colored(f" Avg model inference time: {str(round(np.mean(batch_run_time), 2))} ms", 'magenta'))
            if self.cfg.with_log: self.logger.info(f"====================")
            if self.cfg.with_log: self.logger.info(f" Avg model inference time: {str(round(np.mean(batch_run_time), 2))} ms")
            
            if self.with_asaee:
                
                ASAEE = plot_aee_over_time(
                    data=np.vstack(aee_sets),
                    dst_dir=self.dst_dir,
                    epoch=epoch,
                    dt=self.dt,
                    steps=self.test_params['test_horizons'],
                    num_steps=self.forecast_horizon
                )
                ASAEE = round(float(ASAEE), 3)
                
                if self.cfg.with_print: print(colored(f"====================", 'magenta'))
                if self.cfg.with_log: self.logger.info(f"====================")
                if self.cfg.with_print: print(colored(f"ASAEE: {ASAEE:.3f} m", 'magenta'))
                if self.cfg.with_log: self.logger.info(f"ASAEE: {ASAEE:.3f} m")
                
                with open(os.path.join(self.dst_dir, 'asaee.txt'), "w") as f:
                    
                    f.write(f"ASAEE: {ASAEE:.3f} m")
            
            if self.with_reliability:
                
                mean_RLS, min_RLS, RLS_bins = plot_reliability_calibration(
                    confidence_sets=np.vstack(confidence_sets),
                    bins=self.reliability_bins,
                    dst_dir=self.dst_dir,
                    epoch=epoch,
                    dt=self.dt,
                    steps=self.test_params['test_horizons']
                )
                
                if self.cfg.with_print: print(colored(f"====================", 'magenta'))
                if self.cfg.with_log: self.logger.info(f"====================")
                if self.cfg.with_print: print(colored(f"Reliability Scores:", 'magenta'))
                if self.cfg.with_log: self.logger.info(f"Reliability Scores: ")
                
                if self.cfg.with_print:
                    for idx, step in enumerate(self.test_params['test_horizons']): print(colored(f" avg. RLS @ {round((step+1)*self.dt, 1)} sec: {(1 - np.mean(RLS_bins[idx]))*100:.2f} %", 'magenta'))
                if self.cfg.with_log:
                    for idx, step in enumerate(self.test_params['test_horizons']): self.logger.info(f" avg. RLS @ {round((step+1)*self.dt, 1)} sec: {(1 - np.mean(RLS_bins[idx]))*100:.2f} %")
                
                if self.cfg.with_print: print(colored(f"--------------------", 'magenta'))
                if self.cfg.with_log: self.logger.info(f"--------------------")
                if self.cfg.with_print: print(colored(f" RLS: avg: {mean_RLS:.2f} %, min: {min_RLS:.2f} %", 'magenta'))
                if self.cfg.with_log: self.logger.info(f" RLS: avg: {mean_RLS:.2f} %, min: {min_RLS:.2f} %")
                
                with open(os.path.join(self.dst_dir, 'rls.txt'), "w") as f:
                    
                    f.write("Reliability Scores:")
                    f.write(f"\n--------------------:")
                    for idx, step in enumerate(self.test_params['test_horizons']): f.write(f"\n avg. RLS @ {round((step+1)*self.dt, 1)} sec: {(1 - np.mean(RLS_bins[idx]))*100:.2f} %")
                    f.write(f"\n--------------------:")
                    f.write(f"\n RLS: avg: {mean_RLS:.2f} %, min: {min_RLS:.2f} %")
            
            if self.with_sharpness:
                
                SS = plot_sharpness_over_time(
                    data=np.vstack(sharpness_sets),
                    dst_dir=self.dst_dir,
                    epoch=epoch,
                    dt=self.dt,
                    confidence_levels=self.test_params['confidence_levels'],
                    steps=self.test_params['test_horizons'],
                    num_steps=self.forecast_horizon
                )
                
                if self.cfg.with_print: print(colored(f"====================", 'magenta'))
                if self.cfg.with_log: self.logger.info(f"====================")
                if self.cfg.with_print: print(colored(f"Sharpness Score:", 'magenta'))
                if self.cfg.with_log: self.logger.info(f"Sharpness Score: ")
                
                if self.cfg.with_print:
                    for idx, kappa in enumerate(self.test_params['confidence_levels']): print(colored(f" SS @ {kappa} %: {SS[idx]:.2f} m³/s", 'magenta'))
                if self.cfg.with_log:
                    for idx, kappa in enumerate(self.test_params['confidence_levels']): self.logger.info(f" SS @ {kappa} %: {SS[idx]:.2f} m³/s")
                
                with open(os.path.join(self.dst_dir, 'ss.txt'), "w") as f:
                    
                    f.write("Sharpness Score:")
                    f.write(f"\n--------------------:")
                    for idx, kappa in enumerate(self.test_params['confidence_levels']): f.write(f"\n SS @ {kappa} %: {SS[idx]:.2f} m³/s")
                
            if self.with_ade_fde_k:
                
                # min ade of k samples
                min_ade_k = round(float(np.hstack(min_ade_sets).mean()), 3)
                min_fde_k = round(float(np.hstack(min_fde_sets).mean()), 3)
                
                if self.cfg.with_print: print(colored(f"====================", 'magenta'))
                if self.cfg.with_log: self.logger.info(f"====================")
                if self.cfg.with_print: print(colored(f"Best of K ({self.test_params['num_k_samples']}) min ADE/FDE ", 'magenta'))
                if self.cfg.with_log: self.logger.info(f"Best of K ({self.test_params['num_k_samples']}) min ADE/FDE: ")
                
                if self.cfg.with_print: print(colored(f"--------------------", 'magenta'))
                if self.cfg.with_log: self.logger.info(f"--------------------")
                if self.cfg.with_print: print(colored(f" min ADE K: {str(round(min_ade_k, 3))} m", 'magenta'))
                if self.cfg.with_log: self.logger.info(f" min ADE K: {str(round(min_ade_k, 3))} m")
                
                if self.cfg.with_print: print(colored(f" min FDE K: {str(round(min_fde_k, 3))} m", 'magenta'))
                if self.cfg.with_log: self.logger.info(f" min FDE K: {str(round(min_fde_k, 3))} m")
                
                with open(os.path.join(self.dst_dir, 'ade_fde_k.txt'), "w") as f:
                    
                    f.write(f"Best of K ({self.test_params['num_k_samples']}):")
                    f.write(f"\n--------------------:")
                    f.write(f"\nmin ADE: {str(round(min_ade_k, 3))} m \nmin FDE: {str(round(min_fde_k, 3))} m")
        
        return
        
        
    def save_examples(self, mesh_range_x, mesh_range_y, mesh_resolution, confidence_levels, n_samples, plot_ego, plot_map, epoch=None):
        """Plot 3D forecast examples.
        """
        
        self.model.eval()
        grid = self.build_mesh_grid(
            mesh_range_x=self.test_params['mesh_range_x'],
            mesh_range_y=self.test_params['mesh_range_y'],
            mesh_range_z=self.test_params['mesh_range_z'],
            mesh_resolution=self.test_params['mesh_resolution']
        )
        
        with torch.no_grad():
            
            for p in range(0, len(self.data_X), self.plot_step):
                
                input = torch.tensor(self.data_X[p:p+1][:,-self.eval_input_horizons:,:], dtype=torch.float32).to(self.device)
                target = torch.tensor(self.data_y[p:p+1], dtype=torch.float32).to(self.device)
                reference_pos = self.data_ref[p]
                rotation_angle = self.data_rot[p]
                src = self.data_src[p]
                
                # run model
                output = self.model(input)
                
                # only use user defined output steps
                filtered_output = torch.stack([output[:,k,:] for k in self.test_params['test_horizons']], dim=1)
                filtered_target = torch.stack([target[:,k,:] for k in self.test_params['test_horizons']], dim=1)
                conf_map = self.build_confidence_set_mdn(output=filtered_output, target=grid, num_gaussians=self.num_gaussians, n_samples=n_samples)
                modes = grid[torch.argmin(conf_map, dim=0, keepdim=True)][0,:,0,:]
                
                # get ade and fde
                EE = torch.sqrt(torch.square(filtered_target-modes).sum(dim=-1)).cpu().numpy()
                ADE = str(round(np.mean(EE.T),3))
                FDE = str(round(np.mean(EE.T[-1]),3))
                
                if plot_ego:
                    
                    plot_ego_forecast(
                        cfg=self.cfg,
                        X=input.cpu().numpy(),
                        y=filtered_target.cpu().numpy(),
                        forecasts=None,
                        modes=modes.cpu().numpy(),
                        dst_dir=self.plot_ego_dst_dir,
                        sample_id=p,
                        epoch=epoch,
                        confidence_levels=confidence_levels,
                        ade=ADE,
                        fde=FDE,
                        src=src
                    )
                
                if plot_map:
                    
                    input_ego = np.squeeze(input.cpu().numpy(), axis=0)[:,:3]
                    target_ego = np.squeeze(filtered_target.cpu().numpy(), axis=0)[:,:3]
                    modes_ego = modes.cpu().numpy()
                    
                    input_world = ego2world(X=input_ego, rotation_angle=rotation_angle, translation=reference_pos)
                    target_world = ego2world(X=target_ego, rotation_angle=rotation_angle, translation=reference_pos)
                    modes_world = ego2world(X=modes_ego, rotation_angle=rotation_angle, translation=reference_pos)
                    
                    plot_world_forecast(
                        cfg=self.cfg,
                        input_world=input_world,
                        target_world=target_world,
                        modes_world=modes_world,
                        dst_dir=self.plot_world_dst_dir,
                        sample_id=p,
                        epoch=epoch,
                        ade=ADE,
                        fde=FDE,
                        src=src
                    )
        
        return
    

    def build_mesh_grid(self, mesh_range_x, mesh_range_y, mesh_range_z, mesh_resolution):
    
        # build 3D grid
        nx = int((2 * mesh_range_x) / mesh_resolution) + 1
        ny = int((2 * mesh_range_y) / mesh_resolution) + 1
        nz = int((2 * mesh_range_z) / mesh_resolution) + 1

        xs = torch.linspace(-mesh_range_x, mesh_range_x, steps=nx)
        ys = torch.linspace(-mesh_range_y, mesh_range_y, steps=ny)
        zs = torch.linspace(-mesh_range_z, mesh_range_z, steps=nz)

        x, y, z = torch.meshgrid(xs, ys, zs, indexing='ij')

        grid = torch.stack([x, y, z], dim=-1)

        # shape: [num_grid_points, 1, 3]
        grid = grid.reshape((-1, 1, 3))

        grid = grid.to(self.device)

        return grid
    
    def build_distribution(self, output, num_gaussians):
        
        # output shape: [batch_size, n_horizons, num_gaussians * 10]
        mu_x = output[..., :num_gaussians]
        mu_y = output[..., num_gaussians:2*num_gaussians]
        mu_z = output[..., 2*num_gaussians:3*num_gaussians]
        
        sigma_x = torch.exp(output[..., 3*num_gaussians:4*num_gaussians]) + 1e-4
        sigma_y = torch.exp(output[..., 4*num_gaussians:5*num_gaussians]) + 1e-4
        sigma_z = torch.exp(output[..., 5*num_gaussians:6*num_gaussians]) + 1e-4
        
        eps = 1e-4
        rho_xy = torch.tanh(output[..., 6*num_gaussians:7*num_gaussians])
        rho_xz = torch.tanh(output[..., 7*num_gaussians:8*num_gaussians])
        rho_yz_partial = torch.tanh(output[..., 8*num_gaussians:9*num_gaussians])
        rho_yz = rho_xy * rho_xz + torch.sqrt((1 - rho_xy ** 2).clamp_min(eps)) * torch.sqrt((1 - rho_xz ** 2).clamp_min(eps)) * rho_yz_partial
        alpha = torch.softmax(output[..., 9*num_gaussians:], dim=-1)
        
        covs = torch.zeros((*mu_x.shape, 3, 3)).to(output.device)
        covs[..., 0, 0] = sigma_x ** 2
        covs[..., 1, 1] = sigma_y ** 2
        covs[..., 2, 2] = sigma_z ** 2
        covs[..., 0, 1] = rho_xy * sigma_x * sigma_y
        covs[..., 1, 0] = rho_xy * sigma_x * sigma_y
        covs[..., 0, 2] = rho_xz * sigma_x * sigma_z
        covs[..., 2, 0] = rho_xz * sigma_x * sigma_z
        covs[..., 1, 2] = rho_yz * sigma_y * sigma_z
        covs[..., 2, 1] = rho_yz * sigma_y * sigma_z
        covs = covs + eps * torch.eye(3, device=output.device)
        
        normal = dist.MultivariateNormal(torch.stack([mu_x, mu_y, mu_z], dim=-1), covs)
        mixture = dist.Categorical(alpha)
        gmm = dist.MixtureSameFamily(mixture, normal)
        return gmm
    
    
    def estimate_sharpness(self, confidence_map, kappa):
        """Estimate 3D confidence volume fraction.

        confidence_map shape: [n_grid_points, n_horizons]
        return: confidence volume fraction for each horizon [n_horizon]
        """

        volume = torch.where(confidence_map <= kappa, 1.0, 0.0)
        volume = volume.mean(dim=0)

        return volume
    
    
    def build_confidence_set_mdn(self, output, target, num_gaussians, n_samples):
        
        # output shape: [batch_size, n_horizons, num_gaussians * 10]
        # target shape: [batch_size or n_grid_points, n_horizons, 3]
        
        # build distribution model
        gmm = self.build_distribution(output=output, num_gaussians=num_gaussians)
        
        gt_log_prob = gmm.log_prob(target)
        samples = gmm.sample(sample_shape=torch.Size([n_samples]))
        samples_log_prob = gmm.log_prob(samples)
        idx_mask = (samples_log_prob > gt_log_prob).float()
        conf = torch.sum(idx_mask, 0)/samples.shape[0]
        return conf
    
    
    def sample_with_probs(self, output, num_gaussians, n_samples):
        
        # build distribution model
        gmm = self.build_distribution(output=output, num_gaussians=num_gaussians)
        
        # get samples and compute log probabilities
        tau = 1
        samples = gmm.sample(sample_shape=torch.Size([n_samples]))
        samples_log_prob = gmm.log_prob(samples)
        probs = torch.exp(samples_log_prob/tau) / torch.sum(torch.exp(samples_log_prob/tau), dim=0)
    
        return samples, probs
