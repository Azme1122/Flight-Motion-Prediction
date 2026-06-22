"""Prepare CV-KF train/test window data from raw 3D trajectories."""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from estimate_reference_trajectory import run_cv_kf_over_full_trajectory


DEFAULT_CONFIG = "configs/trial_data/default_cv_kf_trial.json"


def load_config(config_path):
    """Load a JSON config relative to base_CV_KF."""
    path = Path(config_path)
    if not path.is_absolute():
        path = Path(__file__).resolve().parent / path

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def resolve_path(path):
    """Resolve paths relative to base_CV_KF."""
    path = Path(path)
    if path.is_absolute():
        return path
    return Path(__file__).resolve().parent / path


def load_trajectory_xyz(path, max_points=None):
    """Load one raw trajectory and use the last three columns as X, Y, Z."""
    raw = np.loadtxt(path, dtype=float)

    if raw.ndim == 1:
        raw = raw.reshape(1, -1)

    if raw.shape[1] < 3:
        raise ValueError(f"Expected at least 3 columns in {path}. Got {raw.shape}.")

    xyz = raw[:, -3:]
    xyz = xyz[~np.isnan(xyz).any(axis=1)]

    if max_points is not None:
        xyz = xyz[:max_points]

    return xyz


def split_samples(samples, train_ratio):
    """Chronologically split generated windows into train/test dictionaries."""
    sample_items = [(key, samples[key]) for key in sorted(samples.keys())]
    n_samples = len(sample_items)
    n_train = int(n_samples * train_ratio)

    split_items = {
        "train": sample_items[:n_train],
        "test": sample_items[n_train:],
    }

    split_samples_by_name = {"train": {}, "test": {}}
    for split_name, items in split_items.items():
        for _, sample in items:
            sample = dict(sample)
            sample["split"] = split_name
            sample["source"] = sample["source"].replace("_all_", f"_{split_name}_")
            sample["id"] = len(split_samples_by_name[split_name])
            split_samples_by_name[split_name][sample["id"]] = sample

    return split_samples_by_name


def make_reference_table(dataset_name, split_name, measured, reference, dt):
    """Build a readable pseudo-reference table for one dataset split."""
    point_index = np.arange(len(measured))
    return pd.DataFrame({
        "dataset": dataset_name,
        "split": split_name,
        "point_index": point_index,
        "t": point_index * dt,
        "measured_X": measured[:, 0],
        "measured_Y": measured[:, 1],
        "measured_Z": measured[:, 2],
        "ref_X": reference[:, 0],
        "ref_Y": reference[:, 1],
        "ref_Z": reference[:, 2],
    })


def build_window_samples(measured, reference, obs_len, pred_len, shift, dataset_name, split_name, dt):
    """Create sliding-window CV-KF samples from measured and reference points."""
    window_size = obs_len + pred_len
    if len(measured) < window_size:
        return {}

    samples = {}
    sample_id = 0

    for start in range(0, len(measured) - window_size + 1, shift):
        obs_end = start + obs_len
        pred_end = obs_end + pred_len

        obs_measurements = measured[start:obs_end]
        future_reference = reference[obs_end:pred_end]
        measured_window = measured[start:pred_end]
        reference_window = reference[start:pred_end]

        samples[sample_id] = {
            "X": obs_measurements,
            "y": future_reference,
            "obs_measurements": obs_measurements,
            "future_reference": future_reference,
            "measured_window": measured_window,
            "reference_window": reference_window,
            "dataset": dataset_name,
            "split": split_name,
            "shift": start,
            "id": sample_id,
            "source": f"{dataset_name}_{split_name}_{start:05d}",
            "dt": dt,
        }
        sample_id += 1

    return samples


def merge_split_samples(split_samples_by_dataset):
    """Merge per-dataset sample dictionaries and reindex each split."""
    merged = {"train": {}, "test": {}}

    for split_samples in split_samples_by_dataset:
        for split_name in merged:
            for sample in split_samples[split_name].values():
                sample = dict(sample)
                sample["id"] = len(merged[split_name])
                merged[split_name][sample["id"]] = sample

    return merged


def save_split_samples(split_samples, output_dir):
    """Save CV-KF samples into train/test folders."""
    output_paths = {}
    output_dir = resolve_path(output_dir)

    for split_name, samples in split_samples.items():
        split_dir = output_dir / split_name
        split_dir.mkdir(parents=True, exist_ok=True)
        output_path = split_dir / "cv_kf_samples.pkl"

        with output_path.open("wb") as f:
            pickle.dump(samples, f, protocol=pickle.HIGHEST_PROTOCOL)

        output_paths[split_name] = output_path

    return output_paths


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    return parser.parse_args()


def main():
    """Create pseudo-reference trajectories and CV-KF window samples."""
    args = parse_args()
    cfg = load_config(args.config)

    paths = cfg["paths"]
    model_params = cfg["model_params"]
    split_params = cfg["split_params"]
    reference_params = cfg["reference_params"]

    dt = model_params["delta_t"]
    obs_len = model_params["max_input_horizon"]
    pred_len = model_params["forecast_horizon"]
    max_points = split_params.get("max_points_per_dataset")

    split_samples_by_dataset = []
    reference_tables = []

    for dataset_path in paths["dataset_paths"]:
        resolved_path = resolve_path(dataset_path)
        dataset_name = resolved_path.stem
        measured = load_trajectory_xyz(resolved_path, max_points=max_points)
        print(f"{dataset_name}: raw {measured.shape}")

        reference, _ = run_cv_kf_over_full_trajectory(
            measurements=measured,
            dt=dt,
            measurement_noise_std=reference_params["measurement_noise_std"],
            q_scale=reference_params["q_scale"],
        )
        reference_tables.append(
            make_reference_table(
                dataset_name=dataset_name,
                split_name="all",
                measured=measured,
                reference=reference,
                dt=dt,
            )
        )
        samples = build_window_samples(
            measured=measured,
            reference=reference,
            obs_len=obs_len,
            pred_len=pred_len,
            shift=split_params["shift"],
            dataset_name=dataset_name,
            split_name="all",
            dt=dt,
        )
        dataset_split_samples = split_samples(
            samples=samples,
            train_ratio=split_params["train_ratio"],
        )
        print(f"  all windows: {len(samples)}")
        for split_name in ["train", "test"]:
            print(f"  {split_name}: windows {len(dataset_split_samples[split_name])}")

        split_samples_by_dataset.append(dataset_split_samples)

    merged_split_samples = merge_split_samples(split_samples_by_dataset)
    output_paths = save_split_samples(merged_split_samples, paths["track_data_dir"])

    reference_table = pd.concat(reference_tables, ignore_index=True)
    reference_output = resolve_path(paths["track_data_dir"]) / "pseudo_reference.csv"
    reference_output.parent.mkdir(parents=True, exist_ok=True)
    reference_table.to_csv(reference_output, index=False)

    print("\nSaved pseudo-reference table:", reference_output)
    for split_name in ["train", "test"]:
        print(f"Saved {split_name}: {len(merged_split_samples[split_name])} samples -> {output_paths[split_name]}")


if __name__ == "__main__":
    main()
