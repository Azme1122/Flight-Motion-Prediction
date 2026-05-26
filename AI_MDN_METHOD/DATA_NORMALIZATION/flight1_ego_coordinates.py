from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


sys.path.insert(0, "mdn_trajectory_forecasting/base_mdn")
from utils.helper import estimate_ego_transform, world2ego  # noqa: E402


def get_flight1_ego_coordinates():
    xlsx_path = Path("c1.xlsx")
    data = pd.read_excel(xlsx_path, sheet_name="flight_1")

    world_xyz = data[["X", "Y", "Z"]].to_numpy(dtype=float)
    trajectory = world_xyz[np.newaxis, :, :]

    np.random.seed(7)
    rotation_angle, translation = estimate_ego_transform(trajectory)
    ego_xy = world2ego(
        trajectory,
        rotation_angle,
        translation,
        sample_rate=1,
    )[0, :, :2]

    ego_xyz = np.column_stack([ego_xy[:, 0], ego_xy[:, 1], world_xyz[:, 2]])
    return world_xyz, ego_xyz


def plot_xy(world_xyz, ego_xyz):
    plt.figure(figsize=(6, 5))
    plt.plot(world_xyz[:, 0], world_xyz[:, 1], marker="o")
    plt.scatter(world_xyz[0, 0], world_xyz[0, 1], color="green", s=80, label="start")
    plt.scatter(world_xyz[-1, 0], world_xyz[-1, 1], color="red", s=80, label="end")
    plt.title("World coordinate: X-Y")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.axis("equal")
    plt.grid(True)
    plt.legend()

    plt.figure(figsize=(6, 5))
    plt.plot(ego_xyz[:, 0], ego_xyz[:, 1], marker="o")
    plt.scatter(ego_xyz[0, 0], ego_xyz[0, 1], color="green", s=80, label="start")
    plt.scatter(ego_xyz[-1, 0], ego_xyz[-1, 1], color="red", s=80, label="end")
    plt.title("Ego coordinate: X-Y")
    plt.xlabel("ego X")
    plt.ylabel("ego Y")
    plt.axis("equal")
    plt.grid(True)
    plt.legend()

    plt.show()


if __name__ == "__main__":
    world_xyz, ego_xyz = get_flight1_ego_coordinates()

    ego_table = pd.DataFrame(ego_xyz, columns=["ego_x", "ego_y", "ego_z"])
    print(ego_table.to_string(index=False))

    plot_xy(world_xyz, ego_xyz)
