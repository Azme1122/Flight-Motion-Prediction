import os
import numpy as np


try:
    
    import matplotlib.pyplot as plt
    
except ModuleNotFoundError:
    
    plt = None


def _ensure_dir(path):
    
    if not os.path.exists(path):
        
        os.makedirs(path)
    
    return


def _save_array_fallback(data, dst_dir, name):
    
    _ensure_dir(dst_dir)
    np.savetxt(os.path.join(dst_dir, name), np.asarray(data), delimiter=',')
    
    return


def plot_train_loss(train_loss_list, dst_dir, cfg_name, model_arch):
    """Plot train loss.
    """
    
    _ensure_dir(dst_dir)
    
    if plt is None:
        
        _save_array_fallback(train_loss_list, dst_dir, 'train_loss.txt')
        return
    
    plt.figure(figsize=(12,7))
    plt.plot(train_loss_list)
    plt.xlabel('epochs')
    plt.ylabel('train loss')
    plt.title(f'Train Loss over Epochs for {model_arch} with {cfg_name} config')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(dst_dir, 'train_loss.png'))
    plt.close()
    
    return


def plot_reliability_calibration(confidence_sets, dst_dir, epoch, dt, steps, bins):
    """Plot reliability calibration curve.
    """
    
    _ensure_dir(dst_dir)
    bin_data = np.digitize(confidence_sets, bins=bins)
    reliability_errors = []
    
    if plt is None:
        
        _save_array_fallback(confidence_sets, dst_dir, 'reliability_confidence_sets.txt')
        return 0.0, 0.0, []
    
    plt.figure(figsize=(12,7))
    plt.plot(bins, bins, 'k--', linewidth=2, label='ideal')
    coverage_rows = {0.68: [], 0.95: []}
    
    for idx in range(0, len(steps)):
        
        f0 = np.array(np.bincount(bin_data[:,idx], minlength=len(bins)+1)).T
        acc_f0 = np.cumsum(f0[1:],axis=0)/confidence_sets.shape[0]
        r = abs(acc_f0 - bins)
        reliability_errors.append(r)
        
        for level in coverage_rows:
            
            level_idx = int(np.argmin(np.abs(np.asarray(bins) - level)))
            coverage_rows[level].append(float(acc_f0[level_idx]))
        
        plt.plot(bins, acc_f0, linewidth=2, label=f"{round((steps[idx]+1)*dt, 1)} sec")
    
    reliability_avg_score = (1 - np.mean(reliability_errors))*100
    reliability_min_score = (1 - np.max(reliability_errors))*100
    
    plt.grid(True)
    plt.xlabel('confidence level')
    plt.ylabel('observed frequency')
    title = f'3D Reliability - Avg: {reliability_avg_score:.1f} % - Min: {reliability_min_score:.1f} %'
    
    if epoch:
        
        title += f' - Epoch {epoch}'
    
    plt.title(title)
    plt.legend(fontsize=9)
    plt.tight_layout()
    
    path = os.path.join(dst_dir, 'reliability')
    _ensure_dir(path)
    
    if epoch:
        
        plt.savefig(os.path.join(path, f'reliability_epoch_{str(epoch).zfill(4)}.png'))
        
    else:
        
        plt.savefig(os.path.join(path, 'reliability.png'))
    
    with open(os.path.join(path, 'coverage_summary.txt'), 'w') as f:
        
        for level, values in coverage_rows.items():
            
            f.write(f"Coverage_{int(level*100)}: {np.mean(values)*100:.3f} %\n")
    
    plt.close()
    return reliability_avg_score, reliability_min_score, reliability_errors


def plot_sharpness_over_time(data, dst_dir, epoch, dt, confidence_levels, steps, num_steps):
    """Plot 3D sharpness volumes over time.
    """
    
    _ensure_dir(dst_dir)
    x = [round((k+1) * dt, 2) for k in steps]
    sharpness_scores_list = []
    
    if plt is None:
        
        _save_array_fallback(data.reshape((data.shape[0], -1)), dst_dir, 'sharpness.txt')
        return [0.0 for _ in confidence_levels]
    
    plt.figure(figsize=(12,7))
    
    for idx, kappa in enumerate(confidence_levels):
        
        sample_data = data[:,idx,:]
        mean_volume = np.mean(sample_data, axis=0)
        sharpness_score = float(np.mean(mean_volume / np.array(x)))
        sharpness_scores_list.append(sharpness_score)
        plt.plot(x, mean_volume, marker='o', label=f"{kappa*100:.0f} %, SS: {sharpness_score:.2f} m³/s")
    
    plt.xlabel('prediction horizon [s]')
    plt.ylabel('confidence volume [m³]')
    title = '3D Sharpness over Time'
    
    if epoch:
        
        title += f' at epoch {epoch}'
    
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    
    path = os.path.join(dst_dir, 'sharpness')
    _ensure_dir(path)
    
    if epoch:
        
        plt.savefig(os.path.join(path, f'sharpness_epoch_{str(epoch).zfill(4)}.png'))
        
    else:
        
        plt.savefig(os.path.join(path, 'sharpness.png'))
    
    plt.close()
    
    return sharpness_scores_list


def plot_aee_over_time(data, dst_dir, epoch, dt, steps, num_steps):
    """Plot 3D average Euclidean error over time.
    """
    
    _ensure_dir(dst_dir)
    
    # data shape: [n_samples, n_horizons]
    mean_error = np.mean(data, axis=0)
    x = [round((step+1) * dt, 2) for step in steps]
    asaee = float(np.mean(mean_error))
    
    if plt is None:
        
        _save_array_fallback(np.stack([x, mean_error], axis=1), dst_dir, 'aee_over_time.txt')
        return asaee
    
    plt.figure(figsize=(12,7))
    plt.plot(x, mean_error, marker='o', label=f'ASAEE: {asaee:.3f} m')
    plt.xlabel('prediction horizon [s]')
    plt.ylabel('3D Euclidean error [m]')
    title = f'3D AEE over Time'
    
    if epoch:
        
        title += f' at epoch {epoch}'
    
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    
    if epoch:
        
        plt.savefig(os.path.join(dst_dir, f'aee_epoch_{str(epoch).zfill(4)}.png'))
        
    else:
        
        plt.savefig(os.path.join(dst_dir, 'aee.png'))
    
    plt.close()
    
    return asaee


def plot_ego_forecast(cfg, X, y, forecasts, modes, dst_dir, sample_id, epoch, confidence_levels, ade, fde, src):
    """Plot a 3D forecast example in ego coordinates.
    """
    
    _ensure_dir(dst_dir)
    
    if epoch:
        
        dst_dir = os.path.join(dst_dir, 'epoch_' + str(epoch).zfill(4))
        _ensure_dir(dst_dir)
    
    input_track = np.squeeze(X, axis=0)[..., :3]
    target_track = np.squeeze(y, axis=0)[..., :3]
    modes = np.asarray(modes)
    
    if plt is None:
        
        out = np.vstack([
            np.pad(input_track, ((0,0),(0,0)), mode='constant'),
            np.pad(target_track, ((0,0),(0,0)), mode='constant'),
            np.pad(modes, ((0,0),(0,0)), mode='constant')
        ])
        _save_array_fallback(out, dst_dir, f'sample_{str(sample_id).zfill(8)}.txt')
        return
    
    fig = plt.figure(figsize=(14,10))
    ax = fig.add_subplot(221, projection='3d')
    ax_xy = fig.add_subplot(222)
    ax_z = fig.add_subplot(223)
    ax_xz = fig.add_subplot(224)
    
    ax.plot(input_track[:,0], input_track[:,1], input_track[:,2], color='tab:blue', marker='o', label='input')
    ax.plot(target_track[:,0], target_track[:,1], target_track[:,2], color='black', marker='o', label='target')
    ax.plot(modes[:,0], modes[:,1], modes[:,2], color='tab:orange', marker='o', label='mode')
    ax.set_xlabel('ego x [m]')
    ax.set_ylabel('ego y [m]')
    ax.set_zlabel('ego z [m]')
    ax.legend()
    
    ax_xy.plot(input_track[:,0], input_track[:,1], color='tab:blue', marker='o', label='input')
    ax_xy.plot(target_track[:,0], target_track[:,1], color='black', marker='o', label='target')
    ax_xy.plot(modes[:,0], modes[:,1], color='tab:orange', marker='o', label='mode')
    ax_xy.set_xlabel('ego x [m]')
    ax_xy.set_ylabel('ego y [m]')
    ax_xy.set_title('x-y top view')
    ax_xy.grid(True)
    ax_xy.axis('equal')
    
    t_in = np.arange(input_track.shape[0])
    t_out = np.arange(input_track.shape[0], input_track.shape[0] + target_track.shape[0])
    ax_z.plot(t_in, input_track[:,2], color='tab:blue', marker='o', label='input z')
    ax_z.plot(t_out, target_track[:,2], color='black', marker='o', label='target z')
    ax_z.plot(t_out, modes[:,2], color='tab:orange', marker='o', label='mode z')
    ax_z.set_xlabel('timestep')
    ax_z.set_ylabel('ego z [m]')
    ax_z.set_title('z over time')
    ax_z.grid(True)
    ax_z.legend()
    
    ax_xz.plot(input_track[:,0], input_track[:,2], color='tab:blue', marker='o', label='input')
    ax_xz.plot(target_track[:,0], target_track[:,2], color='black', marker='o', label='target')
    ax_xz.plot(modes[:,0], modes[:,2], color='tab:orange', marker='o', label='mode')
    ax_xz.set_xlabel('ego x [m]')
    ax_xz.set_ylabel('ego z [m]')
    ax_xz.set_title('x-z side view')
    ax_xz.grid(True)
    
    fig.suptitle(f'{src} | ADE: {ade} m | FDE: {fde} m')
    fig.tight_layout()
    fig.savefig(os.path.join(dst_dir, f'sample_{str(sample_id).zfill(8)}.png'))
    plt.close(fig)
    
    return


def plot_world_forecast(cfg, input_world, target_world, modes_world, dst_dir, sample_id, epoch, ade, fde, src):
    """Plot a 3D forecast example in world coordinates.
    """
    
    _ensure_dir(dst_dir)
    
    if epoch:
        
        dst_dir = os.path.join(dst_dir, 'epoch_' + str(epoch).zfill(4))
        _ensure_dir(dst_dir)
    
    if plt is None:
        
        out = np.vstack([input_world, target_world, modes_world])
        _save_array_fallback(out, dst_dir, f'sample_{str(sample_id).zfill(8)}.txt')
        return
    
    fig = plt.figure(figsize=(14,10))
    ax = fig.add_subplot(221, projection='3d')
    ax_xy = fig.add_subplot(222)
    ax_z = fig.add_subplot(223)
    ax_yz = fig.add_subplot(224)
    
    ax.plot(input_world[:,0], input_world[:,1], input_world[:,2], color='tab:blue', marker='o', label='input')
    ax.plot(target_world[:,0], target_world[:,1], target_world[:,2], color='black', marker='o', label='target')
    ax.plot(modes_world[:,0], modes_world[:,1], modes_world[:,2], color='tab:orange', marker='o', label='mode')
    ax.set_xlabel('world x [m]')
    ax.set_ylabel('world y [m]')
    ax.set_zlabel('world z [m]')
    ax.legend()
    
    ax_xy.plot(input_world[:,0], input_world[:,1], color='tab:blue', marker='o', label='input')
    ax_xy.plot(target_world[:,0], target_world[:,1], color='black', marker='o', label='target')
    ax_xy.plot(modes_world[:,0], modes_world[:,1], color='tab:orange', marker='o', label='mode')
    ax_xy.set_xlabel('world x [m]')
    ax_xy.set_ylabel('world y [m]')
    ax_xy.set_title('x-y top view')
    ax_xy.grid(True)
    ax_xy.axis('equal')
    
    t_in = np.arange(input_world.shape[0])
    t_out = np.arange(input_world.shape[0], input_world.shape[0] + target_world.shape[0])
    ax_z.plot(t_in, input_world[:,2], color='tab:blue', marker='o', label='input z')
    ax_z.plot(t_out, target_world[:,2], color='black', marker='o', label='target z')
    ax_z.plot(t_out, modes_world[:,2], color='tab:orange', marker='o', label='mode z')
    ax_z.set_xlabel('timestep')
    ax_z.set_ylabel('world z [m]')
    ax_z.set_title('z over time')
    ax_z.grid(True)
    ax_z.legend()
    
    ax_yz.plot(input_world[:,1], input_world[:,2], color='tab:blue', marker='o', label='input')
    ax_yz.plot(target_world[:,1], target_world[:,2], color='black', marker='o', label='target')
    ax_yz.plot(modes_world[:,1], modes_world[:,2], color='tab:orange', marker='o', label='mode')
    ax_yz.set_xlabel('world y [m]')
    ax_yz.set_ylabel('world z [m]')
    ax_yz.set_title('y-z side view')
    ax_yz.grid(True)
    
    fig.suptitle(f'{src} | ADE: {ade} m | FDE: {fde} m')
    fig.tight_layout()
    fig.savefig(os.path.join(dst_dir, f'sample_{str(sample_id).zfill(8)}.png'))
    plt.close(fig)
    
    return
