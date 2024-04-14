import sapien.core as sapien
from sapien.utils.viewer import Viewer
import numpy as np

def demo(fix_root_link, balance_passive_force):
    engine = sapien.Engine()
    renderer = sapien.SapienRenderer()
    engine.set_renderer(renderer)

    scene_config = sapien.SceneConfig()
    scene = engine.create_scene(scene_config)
    scene.set_timestep(1 / 240.0)
    scene.add_ground(0)
    scene_config.gravity = np.array([0.0, 0.0, -9.81])

    scene.set_ambient_light([0.5, 0.5, 0.5])
    scene.add_directional_light([0, 1, -1], [0.5, 0.5, 0.5])

    viewer = Viewer(renderer)
    viewer.set_scene(scene)
    viewer.set_camera_xyz(x=-2, y=0, z=1)
    viewer.set_camera_rpy(r=0, p=-0.3, y=0)

    # Load URDF
    loader: sapien.URDFLoader = scene.create_urdf_loader()
    loader.fix_root_link = fix_root_link
    # robot: sapien.Articulation = loader.load("atlas_description/urdf/atlas_v4_with_multisense.urdf")
    # filename = "atlas_description/urdf/atlas_v4_with_multisense.urdf"
    filename = "atlas_description/urdf/atlas_allegro.urdf"
    robot_builder = loader.load_file_as_articulation_builder(filename)
    robot = robot_builder.build()
    robot.set_root_pose(sapien.Pose([0, 0, 0], [1, 0, 0, 0]))
    print(f"Robot joints: {len(robot.get_joints())}")
    print(f"Robot active joints: {len(robot.get_active_joints())}")

    # Set initial joint positions
    # arm_init_qpos = [4.71, 2.84, 0, 0.75, 4.62, 4.48, 4.88]
    # gripper_init_qpos = [0, 0, 0, 0, 0, 0]
    # init_qpos = arm_init_qpos + gripper_init_qpos
    # robot.set_qpos(init_qpos)
    active_joints = robot.get_active_joints()
    target_qpos = [0] * len(active_joints)
    if balance_passive_force:
            qf = robot.compute_passive_force(
                gravity=True, 
                coriolis_and_centrifugal=True, 
            )
            robot.set_qf(qf)
    for joint_idx, joint in enumerate(active_joints):
        joint.set_drive_property(stiffness=50, damping=5)
    

    while not viewer.closed:
        for joint_idx, joint in enumerate(active_joints):
            # joint.set_drive_target(target_qpos[joint_idx])
            pass
        
        scene.step()
        scene.update_render()
        viewer.render()
        

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--fix-root-link', action='store_true')
    parser.add_argument('--balance-passive-force', action='store_true')
    args = parser.parse_args()

    demo(fix_root_link=args.fix_root_link,
         balance_passive_force=args.balance_passive_force)


if __name__ == '__main__':
    main()