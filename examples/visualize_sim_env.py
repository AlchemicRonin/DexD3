from sapien.utils import Viewer
import argparse
import numpy as np

from dexart.env.sim_env.laptop_env import LaptopEnv
from dexart.env.sim_env.pen_env import PenEnv
from dexart.env.sim_env.pot_env import PotEnv


def main(task_name: str) -> None:
    if task_name == "laptop":
        env = LaptopEnv(index=-1)
    elif task_name == "pen":
        env = PenEnv(index=-1, friction=100000)
    elif task_name == "pot":
        env = PotEnv(index=-1)
    else:
        raise ValueError(f"Unknown task name: {task_name}")
    env.seed(0)

    viewer = Viewer(env.renderer)
    viewer.set_scene(env.scene)
    viewer.set_camera_xyz(x=0, y=0.5, z=0.5)
    viewer.set_camera_rpy(r=0, p=-0.5, y=np.pi / 2)
    env.viewer = viewer

    while not viewer.closed:
        env.simple_step()
        env.render()

    viewer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_name", type=str, required=True)
    args = parser.parse_args()
    task_name = args.task_name

    main(task_name)
