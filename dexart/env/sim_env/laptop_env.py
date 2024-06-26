import os
import random
from pathlib import Path
import numpy as np
import sapien.core as sapien
import transforms3d
from dexart.env.sim_env.base import BaseSimulationEnv
from dexart.env.task_setting import TASK_CONFIG
import json


class LaptopEnv(BaseSimulationEnv):
    def __init__(
        self, use_gui=True, frame_skip=5, friction=5, iter=0, **renderer_kwargs
    ):
        super().__init__(use_gui=use_gui, frame_skip=frame_skip, **renderer_kwargs)
        self.last_openness = None
        self.instance_collision_links = None
        self.instance_links = None
        self.r_handle = None
        self.l_handle = None
        self.handle2link_relative_pose = None
        self.scale_path = None
        self.iter = iter
        # Construct scene
        scene_config = sapien.SceneConfig()
        self.scene = self.engine.create_scene(config=scene_config)
        self.scene.set_timestep(0.004)
        self.instance = None

        # Dummy camera creation to initial geometry object
        if self.renderer and not self.no_rgb:
            cam = self.scene.add_camera(
                "init_not_used", width=10, height=10, fovy=1, near=0.1, far=1
            )
            self.scene.remove_camera(cam)

        self.friction = friction
        # load table
        # TODO: use default table setting for height and size
        self.table = self.create_table()
        self.create_room()
        # default pos and orn, will be used in reset_env
        self.pos = np.array([0, 0, 0.1])
        self.orn = transforms3d.euler.euler2quat(0, 0, 0)

        index = renderer_kwargs["index"]
        self.task_config_name = "laptop"
        self.instance_list = TASK_CONFIG[self.task_config_name]
        if isinstance(index, list):
            self.instance_list = index
            index = -1
        self.change_instance_when_reset = True if index == -1 else False

        self.handle2link_relative_pose_dict = dict()
        self.setup_instance_annotation()
        # for laptop env
        self.init_open_rad = 0.4

        if not self.change_instance_when_reset:
            self.index = self.instance_list[index]
            self.instance, self.revolute_joint, self.revolute_joint_index = (
                self.load_instance(index=self.index)
            )
            self.r_handle = self.revolute_joint.get_child_link()
            self.l_handle = self.revolute_joint.get_parent_link()
            self.instance_links = self.instance.get_links()
            self.instance_collision_links = [
                link
                for link in self.instance.get_links()
                if len(link.get_collision_shapes()) > 0
            ]
            self.l_handle_id = self.r_handle.get_id()
            self.r_handle_id = self.l_handle.get_id()
            self.instance_ids_without_handle = [
                link.get_id() for link in self.instance_links
            ]
            self.instance_ids_without_handle.remove(self.l_handle_id)
            self.instance_ids_without_handle.remove(self.r_handle_id)
            self.last_openness = self.instance.get_qpos()[self.revolute_joint_index]
            if not self.index in self.handle2link_relative_pose_dict:
                self.handle2link_relative_pose_dict[self.index] = {}
                self.handle2link_relative_pose_dict[self.index].update(
                    self.update_handle_relative_pose("r_handle")
                )
                self.handle2link_relative_pose_dict[self.index].update(
                    self.update_handle_relative_pose("l_handle")
                )

        self.reset_env()

    def setup_instance_annotation(self):
        current_dir = Path(__file__).parent
        self.scale_path = (
            current_dir.parent.parent.parent
            / "assets"
            / "annotation"
            / "laptop_scale.json"
        )
        if os.path.exists(self.scale_path):
            with open(self.scale_path, "r") as f:
                self.scale_dict = json.load(f)
        else:
            self.scale_dict = dict()
        self.joint_dicts = dict()
        for instance_index in TASK_CONFIG["laptop"]:
            joint_json_path = (
                current_dir.parent.parent.parent
                / "assets"
                / "sapien"
                / str(instance_index)
                / "mobility_v2.json"
            )
            with open(joint_json_path, "r") as load_f:
                load_dict = json.load(load_f)
            self.joint_dicts[instance_index] = load_dict
        self.joint_limits_dict_path = (
            current_dir.parent.parent.parent
            / "assets"
            / "annotation"
            / "laptop_joint_annotation.json"
        )
        self.joint_limits_dict = dict()
        if os.path.exists(self.joint_limits_dict_path):
            with open(self.joint_limits_dict_path, "r") as f:
                self.joint_limits_dict = json.load(f)

    def load_instance(self, index):
        loader: sapien.URDFLoader = self.scene.create_urdf_loader()
        loader.load_multiple_collisions_from_file = True
        loader.fix_root_link = True
        current_dir = Path(__file__).parent
        urdf_path = str(
            current_dir.parent.parent.parent
            / "assets"
            / "sapien"
            / str(index)
            / "mobility.urdf"
        )
        loader.scale = (
            self.scale_dict[str(index)] if str(index) in self.scale_dict else 1
        )

        instance: sapien.Articulation = loader.load(urdf_path, config={"density": 500})
        for joint in instance.get_joints():
            # TODO: change friction to emphasize dual-arm manipulation
            joint.set_friction(self.friction)
            # joint.set_drive_property(0, 5)

        load_dict = self.joint_dicts[index]
        joint_size = len(load_dict)
        # in sapien, it will auto add the fixed base joint, so the loaded joint_size need to plus 1
        # assert len(laptop.get_joints()) == joint_size + 1
        dof = 0
        revolute_joint = None
        for i, joint_entry in enumerate(load_dict):
            if joint_entry["joint"] == "free":
                dof += 1
            if joint_entry["joint"] == "hinge":
                revolute_joint_index = dof - 1
                revolute_joint = instance.get_active_joints()[revolute_joint_index]
        assert (
            dof == instance.dof
        ), "dof parse error, index={}, calculate_dof={}, real_dof={}".format(
            index, dof, instance.dof
        )
        assert revolute_joint, "revolue_joint can not be None!"
        return instance, revolute_joint, revolute_joint_index

    def reset_env(self):
        if self.change_instance_when_reset:
            if self.instance is not None:
                self.scene.remove_articulation(self.instance)
                self.instance, self.revolute_joint, self.revolute_joint_index = (
                    None,
                    None,
                    None,
                )
            self.index = self.instance_list[
                random.randint(0, len(self.instance_list) - 1)
            ]
            self.instance, self.revolute_joint, self.revolute_joint_index = (
                self.load_instance(index=self.index)
            )
            self.instance.set_qpos(self.joint_limits_dict[str(self.index)]["middle"])
            self.r_handle = self.revolute_joint.get_child_link()
            self.l_handle = self.revolute_joint.get_parent_link()
            self.instance_links = self.instance.get_links()
            self.instance_collision_links = [
                link
                for link in self.instance.get_links()
                if len(link.get_collision_shapes()) > 0
            ]
            self.l_handle_id = self.r_handle.get_id()  # upper surface of laptop
            self.r_handle_id = self.l_handle.get_id()  # base of laptop
            self.instance_ids_without_handle = [
                link.get_id() for link in self.instance_links
            ]
            self.instance_ids_without_handle.remove(self.l_handle_id)
            self.instance_ids_without_handle.remove(self.r_handle_id)
            self.last_openness = self.instance.get_qpos()[self.revolute_joint_index]
            if not self.index in self.handle2link_relative_pose_dict:
                self.handle2link_relative_pose_dict[self.index] = {}
                self.handle2link_relative_pose_dict[self.index].update(
                    self.update_handle_relative_pose("r_handle")
                )
                self.handle2link_relative_pose_dict[self.index].update(
                    self.update_handle_relative_pose("l_handle")
                )
        pos = self.pos  # can add noise here to randomize loaded position
        orn = self.orn
        self.instance.set_root_pose(sapien.Pose(pos, orn))
        self.instance.set_qpos(
            [self.joint_limits_dict[str(self.index)]["left"] + self.init_open_rad]
        )

    def update_handle_relative_pose(self, handle_name=None):
        vertices_global_pose_list = list()

        assert handle_name in [
            "r_handle",
            "l_handle",
        ], "handle_name should be either 'r_handle' or 'l_handle'"
        handle = getattr(self, handle_name)

        if handle_name == "r_handle":
            for collision_mesh in handle.get_collision_shapes():
                vertices = collision_mesh.geometry.vertices
                for vertex in vertices:
                    vertex_relative_pose = sapien.Pose(
                        vertex * collision_mesh.geometry.scale
                    ).transform(collision_mesh.get_local_pose())
                    vertices_global_pose_list.append(
                        handle.get_pose().transform(vertex_relative_pose)
                    )

            z_max = -1e9
            y_max = -1e9
            y_min = 1e9
            x_min = 1e9
            min_z_index = 0
            sum_pos = np.zeros(3)
            for i, vertex_global_pose in enumerate(vertices_global_pose_list):
                sum_pos += vertex_global_pose.p
                z = vertex_global_pose.p[2]
                y = vertex_global_pose.p[1]
                x = vertex_global_pose.p[0]
                if z > z_max:
                    z_max = z
                    max_z_index = i
                if y < y_min:
                    y_min = y
                if y > y_max:
                    y_max = y
                if x < x_min:
                    x_min = x

            # for x and z, we use the corresponding value of the highest vertex
            # for y, we use the mean of all the vertices
            # x = vertices_global_pose_list[max_z_index].p[0]
            # z = z_max
            # y = mean_pos[1]
            y = (y_max + 2*y_min) / 3
            z = z_max
            # x = vertices_global_pose_list[min_z_index].p[0]
            x = x_min

            handle_global_pose = sapien.Pose(np.array([x, y, z]))
            relative_pose = handle.get_pose().inv().transform(handle_global_pose)
        elif handle_name == "l_handle":
            for collision_mesh in handle.get_collision_shapes():
                vertices = collision_mesh.geometry.vertices
                for vertex in vertices:
                    vertex_relative_pose = sapien.Pose(
                        vertex * collision_mesh.geometry.scale
                    ).transform(collision_mesh.get_local_pose())
                    vertices_global_pose_list.append(
                        handle.get_pose().transform(vertex_relative_pose)
                    )

            z_max = -1e9
            y_max = -1e9
            # x_max = -1e9
            x_min = 1e9
            max_z_index = -1
            sum_pos = np.zeros(3)
            for i, vertex_global_pose in enumerate(vertices_global_pose_list):
                # builder: sapien.ActorBuilder = self.scene.create_actor_builder()
                # builder.add_sphere_visual(radius=0.01, color=[0, 1, 0])
                # point: sapien.Actor = builder.build()
                # point.set_pose(vertex_global_pose)
                sum_pos += vertex_global_pose.p
                x = vertex_global_pose.p[0]
                y = vertex_global_pose.p[1]
                z = vertex_global_pose.p[2]
                if z > z_max:
                    z_max = z
                    max_z_index = i
                if x < x_min:
                    x_min = x
                if y > y_max:
                    y_max = y
                # if x > x_max:
                #     x_max = x

            # for x and z, we use the corresponding value of the highest vertex
            # for y, we use the mean of all the vertices
            # x = vertices_global_pose_list[max_z_index].p[0]
            # z = z_max
            # y = mean_pos[1]
            x = x_min + 0.10
            z = z_max + 0.02
            y = y_max - 0.10
            # y = vertices_global_pose_list[max_z_index].p[1]

            handle_global_pose = sapien.Pose(np.array([x, y, z]))
            relative_pose = handle.get_pose().inv().transform(handle_global_pose)
        return {handle_name: relative_pose}

    def get_handle_global_pose(self):
        r_better_global_pose = self.r_handle.get_pose().transform(
            self.handle2link_relative_pose_dict[self.index]["r_handle"]
        )
        l_better_global_pose = self.l_handle.get_pose().transform(
            self.handle2link_relative_pose_dict[self.index]["l_handle"]
        )

        # builder: sapien.ActorBuilder = self.scene.create_actor_builder()
        # builder.add_sphere_visual(radius=0.01, color=[0, 0, 1])
        # right_point: sapien.Actor = builder.build()
        # right_point.set_pose(r_better_global_pose)

        # builder: sapien.ActorBuilder = self.scene.create_actor_builder()
        # builder.add_sphere_visual(radius=0.01, color=[1, 0, 0])
        # left_point: sapien.Actor = builder.build()
        # left_point.set_pose(l_better_global_pose)

        return r_better_global_pose, l_better_global_pose
