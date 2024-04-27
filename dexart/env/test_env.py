import sapien.core as sapien
import numpy as np
from dexart.env.rl_env.base import BaseRLEnv
from dexart.env.rl_env.laptop_env import LaptopRLEnv
# from sapien.utils import Viewer


if __name__ == '__main__':
    # print("000")
    env = LaptopRLEnv(use_gui=True, robot_name='atlas')

    # robot_dof = env.robot.dof
    # viewer = env.create_viewer()
    # viewer = Viewer(env.renderer)
    env.seed(0)
    env.reset()
    
    # env.robot.set_qpos([-3.14/4, 0 ,1.57 ,3.14/2 ,0 ,0, 0 ] + [3.14/4, 0 ,1.57 ,-3.14/2 ,0 ,0, 0 ] + [0] * 16)

    while True:
        # for i in range(1000):
        #     if i % 1000 == 0:
        #         arm_action = np.random.random(12) * 2 - 1
        #         finger_action = (np.random.random(16) * 2 - 1) * np.pi
        #         action = np.concatenate((arm_action, finger_action), axis=0)
        #     action = np.zeros(28)
        env.step(np.array([0,0,0,0,0,0,  0,0,0,0,0,0]+[0]*16))
        env.render()
    viewer.close()