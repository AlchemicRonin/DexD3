import json
import os
from functools import cached_property
from pathlib import Path
from typing import Optional, Dict
from enum import Enum, auto

import numpy as np
import sapien.core as sapien
import transforms3d

from dexart.env.rl_env.base import BaseRLEnv
from dexart.env.rl_env.bibase import BaseBimanualRLEnv
from dexart.env.sim_env.constructor import add_default_scene_light
from dexart.env.sim_env.pot_env import PotEnv
from dexart.env import task_setting


class GraspState(Enum):
    REACHING = 1
    GRASPING = 2
    GRASPED = 3

class PotRLEnv(PotEnv, BaseRLEnv):
    def __init__(self, use_gui=False, frame_skip=5, robot_name="atlas", friction=5, index=0, rand_pos=0.0,
                 rand_orn=0.0, **renderer_kwargs):
        super().__init__(use_gui, frame_skip, friction=friction, index=index, **renderer_kwargs)
        # ============== status definition ==============
        self.instance_init_pos = None
        self.robot_init_pose = None
        self.finger_object_contact = None
        self.finger_tip_pos = None
        self.rand_pos = rand_pos
        self.rand_orn = rand_orn
        # ============== will not change during training and randomize instance ==============
        self.robot_name = robot_name
        self.setup(robot_name)
        # self.robot_init_pose_l = sapien.Pose(np.array([-0.5, -0.3, 0]), transforms3d.euler.euler2quat(0, 0, 0))
        # self.robot_l.set_pose(self.robot_init_pose_l)
        
        # TODO: keep robot position the same for all env
        self.robot_init_pose = sapien.Pose(np.array([-0.9, 0, -1.2]), transforms3d.euler.euler2quat(0, 0, 0))
        self.robot.set_pose(self.robot_init_pose)
        # self.actor_builder = self.scene.create_actor_builder()
        # self.actor_builder.add_box_visual(half_size=[0.01, 0.01, 0.01], color=[1., 0., 0.])

        self.configure_robot_contact_reward()
        self.robot_annotation = self.setup_robot_annotation(robot_name)
        # ============== will change if randomize instance ==============
        self.reset()

    def update_cached_state(self):
        # pass
        
        # # right side (with allegro hand)
        # ## finger
        # for i, link in enumerate(self.finger_tip_links):
        #     self.finger_tip_pos[i] = self.finger_tip_links[i].get_pose().p
        # r_check_contact_links = self.finger_contact_links + [self.r_palm_link]
        # finger_contact_boolean = self.check_actor_pair_contacts(r_check_contact_links, self.body)
        # # NOTE: change the name (in BaseRLEnv): self.robot_object_contact -> self.finger_object_contact
        # self.finger_object_contact[:] = np.clip(np.bincount(self.finger_contact_ids, weights=finger_contact_boolean), 0, 1)
        # # any one finger or palm is contacting
        # self.loosen_contact_finger = np.sum(self.finger_object_contact[:-1]) >= 1 or self.finger_object_contact[-1]
        # # two fingers and palm is contacting
        # self.is_contact_finger = np.sum(self.finger_object_contact[:-1]) >= 2 and self.finger_object_contact[-1]
        
        ## palm
        self.r_palm_pose = self.r_palm_link.get_pose()
        self.r_palm_v = self.r_palm_link.get_velocity()
        self.r_palm_w = self.r_palm_link.get_angular_velocity()
        trans_matrix = self.r_palm_pose.to_transformation_matrix()
        self.r_palm_vector = trans_matrix[:3, :3] @ np.array([1, 0, 0])

        # left side (with ball)
        # l_check_contact_links = [self.l_palm_link]
        # ball_contact_boolean = self.check_actor_pair_contacts(l_check_contact_links, self.body)
        # self.ball_object_contact = np.clip(ball_contact_boolean, 0,1) # the clip is not necessary
        
        # palm 
        self.l_palm_pose = self.l_ball_link.get_pose() # use the virtual ball pose as the global pose of l_palm 
        self.l_palm_v = self.l_palm_link.get_velocity()
        self.l_palm_w = self.l_palm_link.get_angular_velocity()
        trans_matrix = self.l_palm_pose.to_transformation_matrix()
        self.l_palm_vector = trans_matrix[:3, :3] @ np.array([1, 0, 0])


        # # arm contact
        # arm_contact_boolean = self.check_actors_pair_contacts(self.arm_contact_links, self.instance_links)
        # l_arm_contact_boolean = self.check_actors_pair_contacts(self.l_arm_contact_links, self.instance_links)
        # r_arm_contact_boolean = self.check_actors_pair_contacts(self.r_arm_contact_links, self.instance_links)
        
        # self.is_arm_contact = np.sum(arm_contact_boolean)
        # self.l_is_arm_contact = np.sum(l_arm_contact_boolean)
        # self.r_is_arm_contact = np.sum(r_arm_contact_boolean)
        
        self.robot_qpos_vec = self.robot.get_qpos()

        # # object state
        # self.height = self.instance.get_pose().p[2]
        # openness = abs(self.instance.get_qpos()[0] - self.joint_limits_dict[str(self.index)]['middle'])
        # total = abs(self.joint_limits_dict[str(self.index)]['left'] - self.joint_limits_dict[str(self.index)]['middle']) - self.init_open_rad
        # self.progress = 1 - openness / total
        self.r_handle_pose, self.l_handle_pose = self.get_handle_global_pose()

        # # print("r_handle_pose:", self.r_handle_pose.p, "l_handle_pose:", self.l_handle_pose.p)

        # self.r_handle_in_palm = self.r_handle_pose.p - self.r_palm_pose.p
        # self.l_handle_in_palm = self.l_handle_pose.p - self.l_palm_pose.p

        
        # # box = self.actor_builder.build(name="box")
        # # pose = sapien.Pose(self.l_palm_pose.p)
        # # box.set_pose(pose)
        # if np.linalg.norm(self.r_handle_pose.p - self.l_palm_pose.p) < 0.10:
        #     print("left plam touch right handle")
        # if np.linalg.norm(self.l_handle_pose.p - self.l_palm_pose.p) < 0.10:
        #     print("left plam touch left handle")
        # # if np.sum(self.finger_object_contact) > 0:
        # #     print("right finger contacted")
        # # if self.ball_object_contact > 0:
        # #     print("left ball contacted")

        # if np.linalg.norm(self.r_handle_in_palm) > 0.1 or np.linalg.norm(self.l_handle_in_palm) > 0.1:  
        #     self.state = GraspState.REACHING

        # elif not self.is_contact_finger: # loose contact or close to the handle
        #     self.state = GraspState.GRASPING
        # else:
        #     self.state = GraspState.GRASPED
        # self.early_done = (self.progress > 0.95) and (self.state == 3)
        # self.is_eval_done = (self.progress > 0.95) and (self.state == 3)

        # # print("state:", self.state, "progress:", self.progress, "is_eval_done:", self.is_eval_done, "early_done:", self.early_done)

    def get_oracle_state(self):
        return self.get_robot_state()

    def get_robot_state(self):
        # 30+(3+3+3)+(3+3+3)+1 = 49
        return np.concatenate([
            self.robot_qpos_vec, 
            self.r_palm_v, 
            self.r_palm_w, 
            self.r_palm_pose.p, 
            self.r_palm_vector[-1:],
            self.l_palm_v, 
            self.l_palm_w, 
            self.l_palm_pose.p, 
            self.l_palm_vector[-1:],
            [float(self.current_step) / float(self.horizon)]
        ])

    def get_reward(self, action):
        reward = 0
        # if self.state == GraspState.REACHING:
        reward = -0.1 * (min(np.linalg.norm(self.r_palm_pose.p - self.r_handle_pose.p), 0.5)
                        + min(np.linalg.norm(self.l_palm_pose.p - self.l_handle_pose.p), 0.5))  # encourage palm be close to handle
        if self.progress < 0:
            reward += 0.5 * self.progress
        
        # elif self.state == GraspState.GRASPING:
        #     reward += 0.2 * (int(self.is_contact_finger))
        #     reward += 0.1 * (int(self.ball_object_contact))
        #     reward -= 0.1 * (int(self.l_is_arm_contact))
        #     reward -= 0.1 * (int(self.r_is_arm_contact))
        #     if self.progress < 0:
        #         reward += 0.5 * self.progress

        # elif self.state == GraspState.GRASPED:
        #     reward += 0.2 * (int(self.is_contact_finger))
        #     reward += 0.1 * (int(self.ball_object_contact))
        #     reward -= 0.1 * (int(self.l_is_arm_contact))
        #     reward -= 0.1 * (int(self.r_is_arm_contact))
        #     reward += 1.0 * self.progress
            
        # if self.early_done:
        #     reward += (self.horizon - self.current_step) * 1.2 * self.progress
        action_penalty = np.sum(np.clip(self.robot.get_qvel(), -1, 1) ** 2) * 0.01
        controller_penalty = (self.r_cartesian_error ** 2) * 1e3
        reward -= 0.01 * (action_penalty + controller_penalty)
        return reward
    
    @DeprecationWarning
    def simple_reset(self):
        self.robot.set_pose(self.robot_init_pose)
        self.instance.set_qpos(self.joint_limits_dict[str(self.index)]['middle'])
        # self.update_cached_state()
        # self.update_imagination(reset_goal=Fal
        # return self.get_observation()

    def reset(self, *, seed: Optional[int] = None, return_info: bool = False, options: Optional[dict] = None):
        # reset status
        # self.robot_l.set_pose(self.robot_init_pose_l)
        self.robot.set_pose(self.robot_init_pose)
        # reset changeable status if randomize instance
        self.reset_internal()   # change instance if randomize instance
        if self.need_flush_when_change_instance and self.change_instance_when_reset:
            self.flush_imagination_config()

        if self.robot_annotation.__contains__(str(self.index)):
            # TODO: change initial position of all task rl env
            self.instance_init_pos = np.array([0,0,0.1])
            # self.instance_init_pos = np.array(self.robot_annotation[str(self.index)]) + self.robot.get_pose().p + np.array([0,0,1])
            # self.instance.set_qpos(self.joint_limits_dict[str(self.index)]['middle']/2)
        else:
            self.instance_init_pos = self.pos
        self.pos = self.instance_init_pos
        pos = self.pos + np.random.random(3) * self.rand_pos  # can add noise here to randomize loaded position
        random_orn = (np.random.rand() * 2 - 1) * self.rand_orn
        orn = transforms3d.euler.euler2quat(0, 0, random_orn)
        self.instance.set_root_pose(sapien.Pose(pos, orn))
        self.update_cached_state()
        self.update_imagination(reset_goal=False)
        return self.get_observation()

    def setup_robot_annotation(self, robot_name: str):
        # here we load robot2laptop
        # NOTE: NOT USED, all laptop is set to the same position, center of the table
        current_dir = Path(__file__).parent
        self.pos_path = current_dir.parent.parent.parent / "assets" / "annotation" / f"laptop_{robot_name}_relative_position.json"
        if not os.path.exists(self.pos_path):
            return dict()
        else:
            with open(self.pos_path, "r") as f:
                pos_dict = json.load(f)
            return pos_dict

    @cached_property
    def obs_dim(self):
        if not self.use_visual_obs:
            return len(self.get_oracle_state())
        else:
            return len(self.get_robot_state())

    def is_done(self):
        #TODO: done when the laptop is dropped or opened
        return (self.current_step >= self.horizon) or self.early_done

    @cached_property
    def horizon(self):
        return 250
