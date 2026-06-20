from __future__ import print_function

import argparse
import os
from pathlib import Path
from six.moves import cPickle as pickle

import numpy as np
import pandas as pd

from utils.helper import estimate_ego_transform, world2ego


DEFAULT_EXCEL_PATH = "/home/rt_azme/Documents/flight-motion-prediction/AI_MDN_METHOD/LSTM_MDN/LSTM_MDN_Trial/base_MDN/raw_data/c1.xlsx"


def parse_args():
    """Parse preprocessing arguments.
    """
    
    p = argparse.ArgumentParser()
    p.add_argument('--excel-path', type=str, default=DEFAULT_EXCEL_PATH)
    p.add_argument('--sheet-name', type=str, default='flight_1')
    p.add_argument('--obs-len', type=int, default=8)
    p.add_argument('--pred-len', type=int, default=12)
    p.add_argument('--max-samples', type=int, default=100)
    p.add_argument('--shift', type=int, default=1)
    p.add_argument('--sample-rate', type=float, default=1.0)
    p.add_argument('--output-dir', type=str, default='mdn_track_data/trial_data')
    
    return p.parse_args()


def load_flight_xyz(excel_path, sheet_name):
    """Load one flight sheet as raw [x, y, z] points.
    """
    
    df = pd.read_excel(excel_path, sheet_name=sheet_name)
    required_cols = ['X', 'Y', 'Z']
    missing_cols = [c for c in required_cols if c not in df.columns]
    
    if missing_cols:
        
        raise ValueError(f"Missing required columns in {sheet_name}: {missing_cols}")
    
    xyz = df[required_cols].dropna().to_numpy(dtype=np.float64)
    
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        
        raise ValueError(f"Expected raw trajectory shape [N, 3], got {xyz.shape}")
    
    return xyz


def build_samples(raw_xyz, obs_len, pred_len, shift, max_samples, sample_rate, source_name):
    """Build 3D ego-coordinate samples from raw [x, y, z] trajectory points.
    """
    
    window_size = obs_len + pred_len
    
    if len(raw_xyz) < window_size:
        
        raise ValueError(f"Need at least {window_size} points, got {len(raw_xyz)}")
    
    samples = {}
    sample_id = 0
    
    for start in range(0, len(raw_xyz) - window_size + 1, shift):
        
        if sample_id >= max_samples:
            
            break
        
        raw_window = raw_xyz[start:start+window_size]
        raw_obs = raw_window[:obs_len]
        
        rotation_angle, reference_position = estimate_ego_transform(raw_obs)
        ego_window = world2ego(
            X=raw_window,
            rotation_angle=rotation_angle,
            translation=reference_position,
            sample_rate=sample_rate
        )
        
        X = ego_window[:obs_len]
        y = ego_window[obs_len:, :3]
        avg_velocity = float(np.linalg.norm(X[:, 3:6], axis=1).mean())
        
        samples[sample_id] = {
            'X': X,
            'y': y,
            'class': 0,
            'reference_position': reference_position,
            'rotation_angle': float(np.asarray(rotation_angle)),
            'movement_class': 'flight',
            'avg_velocity': avg_velocity,
            'shift': start,
            'id': sample_id,
            'source': f"{source_name}_{str(start).zfill(5)}"
        }
        
        sample_id += 1
    
    return samples


def split_samples(samples, train_ratio=0.70, eval_ratio=0.15):
    """Split samples chronologically into train/eval/test dictionaries.
    """
    
    sample_items = [(k, samples[k]) for k in sorted(samples.keys())]
    n_samples = len(sample_items)
    n_train = int(n_samples * train_ratio)
    n_eval = int(n_samples * eval_ratio)
    
    train_items = sample_items[:n_train]
    eval_items = sample_items[n_train:n_train+n_eval]
    test_items = sample_items[n_train+n_eval:]
    
    return {
        'train': {idx: item for idx, (_, item) in enumerate(train_items)},
        'eval': {idx: item for idx, (_, item) in enumerate(eval_items)},
        'test': {idx: item for idx, (_, item) in enumerate(test_items)}
    }


def save_split_samples(split_data, output_dir):
    """Save split sample dictionaries into train/eval/test folders.
    """
    
    output_dir = Path(output_dir)
    output_paths = {}
    
    for split_name, samples in split_data.items():
        
        split_dir = output_dir / split_name
        split_dir.mkdir(parents=True, exist_ok=True)
        output_path = split_dir / 'ego_samples.pkl'
        
        with output_path.open('wb') as f:
            
            pickle.dump(samples, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        output_paths[split_name] = output_path
    
    return output_paths


def main():
    """Create a trial 3D ego_samples.pkl file.
    """
    
    args = parse_args()
    raw_xyz = load_flight_xyz(excel_path=args.excel_path, sheet_name=args.sheet_name)
    samples = build_samples(
        raw_xyz=raw_xyz,
        obs_len=args.obs_len,
        pred_len=args.pred_len,
        shift=args.shift,
        max_samples=args.max_samples,
        sample_rate=args.sample_rate,
        source_name=args.sheet_name
    )
    
    split_data = split_samples(samples=samples)
    output_paths = save_split_samples(split_data=split_data, output_dir=args.output_dir)
    first_sample = samples[0]
    print(f"Loaded raw trajectory: {raw_xyz.shape}")
    print(f"Saved samples: {len(samples)}")
    print(f"First X shape: {first_sample['X'].shape}")
    print(f"First y shape: {first_sample['y'].shape}")
    
    for split_name in ['train', 'eval', 'test']:
        
        print(f"{split_name}: {len(split_data[split_name])} samples -> {os.path.abspath(output_paths[split_name])}")
    
    return


if __name__ == "__main__":
    
    main()
