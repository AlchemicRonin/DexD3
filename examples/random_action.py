#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import argparse
import time

import numpy as np
from sapien.utils import Viewer
from dexart.env.create_env import create_env

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_name', type=str, default='faucet')
    args = parser.parse_args()
    task_name = args.task_name

    env = create_env(task_name=task_name, use_visual_obs=True, img_type='robot', use_gui=True, rand_pos=0.05)
    robot_dof = env.robot.dof
    env.seed(0)
    env.reset()

    # config the viewer
    viewer = Viewer(env.renderer)
    viewer.set_scene(env.scene)
    viewer.focus_camera(env.cameras['instance_1'])
    env.viewer = viewer

    action = np.random.random(robot_dof)
    while True:
        for i in range(10000):
            if i % 100 == 0:
                action = np.random.random(robot_dof)
            env.step(action)
            env.render()
        env.reset()

    viewer.close()