import argparse
import numpy as np
import random


def config_parser():
    """parse command line arguments
    """
    
    p = argparse.ArgumentParser()
    p.add_argument('-t', '--target', type=str, default='trial_local_step')
    p.add_argument('-c', '--configs', type=str, default='default_drone_local_step.json')
    p.add_argument('-l', '--log', action='store_true')
    p.add_argument('-p', '--print', action='store_true')
    p.add_argument('-g', '--gpu', type=str, default='0')
    p.set_defaults(log=True)
    p.set_defaults(print=True)
    
    return p


def normalize_local_step_displacements(trajectory, epsilon=1e-8):
    """Normalize raw 3D points into local step-to-step displacement features.
    
    For each middle point p_k, the previous horizontal movement p_k - p_{k-1}
    defines the local x-y frame. The next movement p_{k+1} - p_k is projected
    into that local frame, while z stays a vertical displacement.
    
    Returns:
        local_steps: [N-2, 3] with [sideways_dx, forward_dy, delta_z]
        current_positions: [N-2, 3] raw p_k positions
        next_positions: [N-2, 3] raw p_{k+1} positions
        heading_angles: [N-2] local x-y heading angles
    """
    
    points = np.asarray(trajectory, dtype=float)
    
    if points.ndim != 2 or points.shape[1] != 3:
        
        raise ValueError(f"Expected trajectory shape [N, 3], got {points.shape}")
    
    if len(points) < 3:
        
        raise ValueError("At least 3 points are required for local-step normalization")
    
    local_steps = []
    current_positions = []
    next_positions = []
    heading_angles = []
    last_valid_forward = None
    
    for k in range(1, len(points) - 1):
        
        previous_xy = points[k-1, :2]
        current_xy = points[k, :2]
        next_xy = points[k+1, :2]
        
        previous_motion = current_xy - previous_xy
        previous_motion_norm = float(np.linalg.norm(previous_motion))
        
        if previous_motion_norm > epsilon:
            
            e_y = previous_motion / previous_motion_norm
            last_valid_forward = e_y
            
        elif last_valid_forward is not None:
            
            e_y = last_valid_forward
            
        else:
            
            continue
        
        e_x = np.array([-e_y[1], e_y[0]], dtype=float)
        next_motion_xy = next_xy - current_xy
        
        sideways_step = float(np.dot(next_motion_xy, e_x))
        forward_step = float(np.dot(next_motion_xy, e_y))
        delta_z = float(points[k+1, 2] - points[k, 2])
        
        local_steps.append([sideways_step, forward_step, delta_z])
        current_positions.append(points[k])
        next_positions.append(points[k+1])
        heading_angles.append(float(np.arctan2(e_y[1], e_y[0])))
    
    return (
        np.asarray(local_steps, dtype=float).reshape(-1, 3),
        np.asarray(current_positions, dtype=float).reshape(-1, 3),
        np.asarray(next_positions, dtype=float).reshape(-1, 3),
        np.asarray(heading_angles, dtype=float).reshape(-1)
    )


def local_steps_to_world(local_steps, start_position, start_heading_angle, epsilon=1e-8):
    """Reconstruct world coordinates from local displacement predictions.
    
    Args:
        local_steps (np.array): [T, 3] local [sideways, forward, delta_z]
        start_position (np.array): world position before the first local step
        start_heading_angle (float): heading used for the first local step
        
    Returns:
        np.array: [T, 3] reconstructed world positions after each step
    """
    
    local_steps = np.asarray(local_steps, dtype=float)
    current_position = np.asarray(start_position, dtype=float).reshape(3).copy()
    heading = float(np.asarray(start_heading_angle))
    world_positions = []
    
    for step in local_steps:
        
        e_y = np.array([np.cos(heading), np.sin(heading)], dtype=float)
        e_x = np.array([-e_y[1], e_y[0]], dtype=float)
        delta_xy = step[0] * e_x + step[1] * e_y
        
        current_position = current_position.copy()
        current_position[:2] += delta_xy
        current_position[2] += step[2]
        world_positions.append(current_position.copy())
        
        motion_norm = float(np.linalg.norm(delta_xy))
        
        if motion_norm > epsilon:
            
            heading = float(np.arctan2(delta_xy[1], delta_xy[0]))
    
    return np.asarray(world_positions, dtype=float).reshape(-1, 3)


def count_model_parameters(model):
    """pytorch method to count a models parameter size
    """
    
    return sum(p.numel() for p in model.parameters())


def generate_unique_randoms(count, min_value, max_value):
    """generate unique random numbers within a given range
    """
    
    generated_numbers = set()
    
    while len(generated_numbers) < count:
        
        new_number = random.randint(min_value, max_value)
        
        if new_number not in generated_numbers:
            
            generated_numbers.add(new_number)
            
    return list(generated_numbers)
