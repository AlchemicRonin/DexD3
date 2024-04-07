import json
import os
from functools import cached_property
from pathlib import Path
from typing import Optional, Dict
from enum import Enum, auto

import numpy as np
import sapien.core as sapien
import transforms3d

# from dexart.env.rl_env.base import BaseRLEnv
from dexart.env.rl_env.bibase import BaseBimanualRLEnv
from dexart.env.sim_env.constructor import add_default_scene_light
from dexart.env.sim_env.pen_env import PenEnv
from dexart.env import task_setting


class GraspState(Enum):
    REACHING = 1
    GRASPING = 2
    GRASPED = 3

class PenRLEnv(PenEnv, BaseBimanualRLEnv):
    def __init__(self, use_gui=False, frame_skip=5, robot_name=["allegro_hand_xarm6_right","allegro_hand_xarm6_left"], friction=5, index=0, rand_pos=0.0,
                 rand_orn=0.0, **renderer_kwargs):
        super().__init__(use_gui, frame_skip, friction=friction, index=index, **renderer_kwargs)
        # ============== status definition ==============
        self.instance_init_pos = None
        self.robot_init_pose = None
        self.robot_object_contact = None
        self.finger_tip_pos = None
        self.rand_pos = rand_pos
        self.rand_orn = rand_orn
        # ============== will not change during training and randomize instance ==============
        self.robot_name = robot_name
        self.setup(robot_name)
        #self.robot_init_pose_l = sapien.Pose(np.array([-0.5, 0, 0]), transforms3d.euler.euler2quat(0, 0, 0))
        #self.robot_l.set_pose(self.robot_init_pose_l)

        #self.robot_init_pose = sapien.Pose(np.array([0.5, 0, 0]), transforms3d.euler.euler2quat(0, 0, 0))
        #self.robot.set_pose(self.robot_init_pose)

if __name__ == "__main__":
    env = PenRLEnv(use_gui=True, index=-1)
    env.reset()
    env.render()

    while not env.viewer.closed:
        env.simple_step()
        env.render()

    env.viewer.close()



    env.arm_sim_step(
        np.array(
            [0,0,0,0,0,0]+[0]*16+\
            [0,0,0,0,0,0]+[0]*16
        )
    )
    env.simple_step()
    env.render()