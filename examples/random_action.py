#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import argparse
import time

import numpy as np
from sapien.utils import Viewer
from dexart.env.create_env import create_env

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_name', type=str, default='laptop')
    args = parser.parse_args()
    task_name = args.task_name

    env = create_env(task_name=task_name, use_visual_obs=True, img_type='robot', use_gui=True, rand_pos=0.05)
    # robot_dof = env.robot.dof
    env.seed(0)
    env.reset()

    # config the viewer
    viewer = Viewer(env.renderer)
    viewer.set_scene(env.scene)
    viewer.focus_camera(env.cameras['instance_1'])
    env.viewer = viewer

    action = np.random.random(28)*2-1
    while not viewer.closed:
        qlimits = env.instance.get_qlimits()
        for i in range(1000):
            if i % 100 == 0:
                arm_action = np.random.random(12) * 2 - 1
                finger_action = (np.random.random(16) * 2 - 1) * np.pi
                action = np.concatenate((arm_action, finger_action), axis=0)
                qpos = np.random.uniform(qlimits[:, 0], qlimits[:, 1])
                # if i != 0:
                #     r_palm_pose_p = obs['oracle_state'][30+6:30+9]
                #     l_palm_pose_p = obs['oracle_state'][30+15:30+18]

            obs, reward, done, info = env.step(action)

            # robot_qpos_vec, 30
            # r_palm_v, 3
            # r_palm_w, 3
            # r_palm_pose.p, 3
            # l_palm_v, 3
            # l_palm_w, 3
            # l_palm_pose.p, 3 
            # [float(self.current_step) / float(self.horizon)] 3
            # env.instance.set_qpos(qpos)
            env.render()
        env.reset()