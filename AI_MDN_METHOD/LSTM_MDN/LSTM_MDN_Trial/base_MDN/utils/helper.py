import argparse
import numpy as np
import random


def config_parser():
    """parse command line arguments
    """
    
    p = argparse.ArgumentParser()
    p.add_argument('-t', '--target', type=str, default='trial_data')
    p.add_argument('-c', '--configs', type=str, default='default_drone_trial.json')
    p.add_argument('-l', '--log', action='store_true')
    p.add_argument('-p', '--print', action='store_true')
    p.add_argument('-g', '--gpu', type=str, default='0')
    p.set_defaults(log=True)
    p.set_defaults(print=True)
    
    return p


def build_R(rotation_angle):
    """build 2D rotation matrix for x-y coordinates
    
    Args:
        rotation_angle (np.array): stack of x-y heading angles
        
    Returns:
        np.array: stack of 2D rotation matrixes
    """
    
    rotation_angle = np.asarray(rotation_angle)
    
    # get identity matrices
    R = np.zeros((*rotation_angle.shape, 2, 2))
    R[..., 0, 0] = np.cos(rotation_angle)
    R[..., 0, 1] = np.sin(rotation_angle)
    R[..., 1, 0] = -np.sin(rotation_angle)
    R[..., 1, 1] = np.cos(rotation_angle)
    
    return R


def build_R_inv(rotation_angle):
    """build inverse 2D rotation matrix for x-y coordinates
    
    Args:
        rotation_angle (np.array): stack of x-y heading angles
        
    Returns:
        np.array: stack of inverted 2D rotation matrixes
    """
    
    # get identity 
    rotation_angle = -np.asarray(rotation_angle)
    R_inv = np.zeros((*rotation_angle.shape, 2, 2))
    R_inv[..., 0, 0] = np.cos(rotation_angle)
    R_inv[..., 0, 1] = np.sin(rotation_angle)
    R_inv[..., 1, 0] = -np.sin(rotation_angle)
    R_inv[..., 1, 1] = np.cos(rotation_angle)
    
    return R_inv


def bat_mat_vec_mult(a, M):
    """simple helper for batched 2D matrix multiplication
    
    Args:
        a (np.array): stack of x-y points
        M (np.array): stack of 2D rotation matrixes
    """
    
    # batched matrix multiplication
    return np.einsum('...ij,...j->...i', M[..., np.newaxis, :, :], a, optimize=True)


def calc_velocities(t, s):
    """calc velocities for ego shifted 3D position data
    
    Args:
        t (np.array): sample data with [ego_x, ego_y, ego_z]
        s (int): sample rate
        
    Returns:
        np.array: track data with [ego_x, ego_y, ego_z, v_ego_x, v_ego_y, v_ego_z]
    """
    
    if t.shape[-1] < 3:
        
        raise ValueError('The trajectory needs at least x, y, z columns')
    
    positions = t[..., 0:3]
    velocities = np.zeros_like(positions)
    
    # finite differences between consecutive positions
    velocities[..., 1:, :] = (positions[..., 1:, :] - positions[..., :-1, :]) / (1/s)
    
    # match existing style: copy first valid velocity to first timestep
    if positions.shape[-2] > 1:
        
        velocities[..., 0, :] = velocities[..., 1, :]
    
    return np.concatenate((positions, velocities), axis=-1)


def crop_trajectory(data, win_size, fh, shift=1, dims=6):
    """partition a large 3D ego trajectory with a sliding window
    
    Args:
        data (np.array): full ego trajectory with [x, y, z, vx, vy, vz]
        win_size (int): total window size, e.g. obs_len + pred_len
        fh (int): forecast horizon / prediction length
        shift (int): shift size between two sliding window crops
        dims (int): dimensions from the trajectory to use
        
    Raises:
        ValueError: dimensionally error
        
    Returns:
        np.array: input windows X, target windows y, number of windows
    """
    
    if dims > data.shape[-1]:
        
        raise ValueError('The trajectory only has %d dimensions' % data.shape[-1])
    
    T = rolling_window(data, shape=(win_size, dims), shift=shift)
    X = T[:,0:-fh,:]
    y = T[:,-fh:,0:3]
    
    # Expected shapes for first 3D trial: raw [N, 3], ego pos [N, 3],
    # ego with velocities [N, 6], X [8, 6], y [12, 3].
    return X, y, T.shape[0]


def rolling_window(a, shape, shift):
    """sliding window over a trajectory
    
    Args:
        a (np.array): input trajectory
        shape (tuple): shape for the sliding window
        shift (int): shift size between to sliding window extractions
        
    Returns:
        np.array: rolling window
    """
    
    s = (a.shape[-2] - shape[-2] + 1,) + (a.shape[-1] - shape[-1] + 1,) + shape
    strides = a.strides + a.strides
    astrided_array = np.lib.stride_tricks.as_strided(a, shape=s, strides=strides).squeeze()
    
    if astrided_array.ndim < 3:
        
        astrided_array = astrided_array[None,...]
        
    elif astrided_array.ndim > 3:
        
        print ("Error too many dimensions in rolling window result...")
    
    # apply shift size and return
    return astrided_array[::shift]


def estimate_ego_transform(X):
    """estimate x-y ego heading and 3D reference position
    
    Args:
        X (np.array): stack of observed raw 3D track data
        
    Returns:
        np.array: rotation angles
        np.array: reference positions [x_ref, y_ref, z_ref]
    """
    
    X = np.asarray(X)
    
    if X.shape[-1] < 3:
        
        raise ValueError('The trajectory needs at least x, y, z columns')
    
    # get last observed point as 3D reference position
    reference_positions = X[...,-1:,0:3]
    
    # estimate heading only in the x-y plane
    dx = X[...,-1,0] - X[...,0,0]
    dy = X[...,-1,1] - X[...,0,1]
    motion_norm = np.sqrt(np.square(dx) + np.square(dy))
    rotation_angles = np.where(motion_norm > 1e-8, np.arctan2(dy, dx), 0.0)
    
    return rotation_angles, reference_positions


def world2ego(X, rotation_angle, translation, sample_rate):
    """transform from world coordinates to 3D ego coordinates
    
    Args:
        X (np.array): stack of raw 3D track data [x, y, z]
        rotation_angle (np.array): stack of x-y heading angles
        translation (np.array): stack of 3D reference positions
        
    Returns:
        np.array: stack of ego track data [ego_x, ego_y, ego_z, v_x, v_y, v_z]
    """
    
    X = np.asarray(X)
    
    if X.shape[-1] < 3:
        
        raise ValueError('The trajectory needs at least x, y, z columns')
    
    R = build_R(rotation_angle)
    X_xy_t = X[...,0:2] - translation[...,0:2]
    X_xy_Rt = bat_mat_vec_mult(X_xy_t,R)
    X_z_t = X[...,2:3] - translation[...,2:3]
    ego_positions = np.concatenate((X_xy_Rt, X_z_t), axis=-1)
    ego = calc_velocities(t=ego_positions, s=sample_rate)
    
    return ego


def ego2world(X, rotation_angle, translation):
    """transform from 3D ego coordinates to world coordinates
    
    Args:
        X (np.array): stack of ego 3D track data [ego_x, ego_y, ego_z]
        rotation_angle (np.array): stack of x-y heading angles
        translation (np.array): stack of 3D reference positions
        
    Returns:
        np.array: stack of track data in world coords [world_x, world_y, world_z]
    """
    
    X = np.asarray(X)
    
    if X.shape[-1] < 3:
        
        raise ValueError('The trajectory needs at least x, y, z columns')
    
    R_inv = build_R_inv(rotation_angle)
    X_xy_R = bat_mat_vec_mult(X[...,0:2],R_inv)
    X_xy_tR = X_xy_R + translation[...,0:2]
    X_z_t = X[...,2:3] + translation[...,2:3]
    
    return np.concatenate((X_xy_tR, X_z_t), axis=-1)


def count_model_parameters(model):
    """pytorch method to count a models parameter size
    """
    
    return sum(p.numel() for p in model.parameters())


def generate_unique_randoms(count, min_value, max_value):
    """generate unique random numbers within a given range
    """
    
    # create a set to store the generated random numbers
    generated_numbers = set()
    
    while len(generated_numbers) < count:
        
        # generate a new random number in the specified range
        new_number = random.randint(min_value, max_value)
        
        # if the number is not already in the set, add it to the set and continue generating numbers
        if new_number not in generated_numbers:
            
            generated_numbers.add(new_number)
            
    return list(generated_numbers)
