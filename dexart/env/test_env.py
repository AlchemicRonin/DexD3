from dexart.env.sim_env.usb_env import USBEnv
import sapien.core as sapien
import numpy as np
from dexart.env.rl_env.base import BaseRLEnv
from dexart.env.rl_env.laptop_env import LaptopRLEnv


if __name__ == '__main__':
    print("000")
    env = LaptopRLEnv(use_gui=True, robot_name='atlas')

    # robot_dof = env.robot.dof
    # viewer = env.create_viewer()
    env.seed(0)
    env.simple_reset()

    while True:
        for i in range(10000):
            if i % 100 == 0:
                arm_action = np.random.random(14) * 2 - 1
                finger_action = (np.random.random(16) * 2 - 1) * np.pi
                action = np.concatenate((arm_action, finger_action), axis=0)
            env.atlas_sim_step(action)
            env.render()
        env.simple_reset()

    viewer.close()