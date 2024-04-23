import numpy as np
from dexart.env.task_setting import BOUND_CONFIG


def process_pc(task_name: str, cloud: np.ndarray, camera_pose: np.ndarray, num_points: int,
               np_random: np.random.RandomState, noise_level=0, grouping_info=None, segmentation=None) -> np.ndarray:
    """
    1. only sample pc to num_points
    noise_level=0, group_info=None, segmentation=None

    2. sample pc and add noise
    noise_level!=0, group_info=None, segmentation=None

    3. sample pc,  add noise and use segmentation
    (output : num_points x (3 + NUM_CLASS))
    noise_level!=0, group_info!=None, segmentation!=None (H,W,1)

    other input params combination may cause undefined behavior
    """

    # first remove points with z < -0.05 (remove artifacts)
    within_bound_r = (cloud[..., 2] < -0.05)

    pc = cloud[..., :3]
    pc = pc @ camera_pose[:3, :3].T + camera_pose[:3, 3]
    bound = BOUND_CONFIG[task_name]

    # remove robot table
    within_bound_x = (pc[..., 0] > bound[0]) & (pc[..., 0] < bound[1])
    within_bound_y = (pc[..., 1] > bound[2]) & (pc[..., 1] < bound[3])
    within_bound_z = (pc[..., 2] > bound[4]) & (pc[..., 2] < bound[5])
    within_bound = np.nonzero(np.logical_and.reduce((within_bound_x, within_bound_y, within_bound_z, within_bound_r)))[
        0]

    num_index = len(within_bound)
    if num_index == 0:
        return np.zeros([num_points, 3])
    if num_index < num_points:
        # repeat first point as padding at the end
        indices = np.concatenate([within_bound, np.ones(num_points - num_index, dtype=np.int32) * within_bound[0]])
        if noise_level != 0:
            multiplicative_noise = 1 + np_random.randn(num_index)[:, None] * 0.01 * noise_level  # (num_index, 1)
            multiplicative_noise = np.concatenate(
                [multiplicative_noise, np.ones([num_points - num_index, 1]) * multiplicative_noise[0]], axis=0)
    else:
        indices = within_bound[np_random.permutation(num_index)[:num_points]]
        if noise_level != 0:
            multiplicative_noise = 1 + np_random.randn(num_points)[:, None] * 0.01 * noise_level  # (n, 1)

    if noise_level != 0:
        pc = pc[indices, :] * multiplicative_noise
        cloud = np.concatenate([pc, cloud[indices, 3:]], axis=1)
    else:
        pc = pc[indices, :]
        cloud = np.concatenate([pc, cloud[indices, 3:]], axis=1)

    if segmentation is not None:
        labels = segmentation[indices, :]   # N x 1
        # handle
        r_handle_mask = np.isin(labels, grouping_info['r_handle'])
        l_handle_mask = np.isin(labels, grouping_info['l_handle'])
        instance_body_mask = np.isin(labels, grouping_info['instance_body'])
        l_arm_mask = np.isin(labels, grouping_info['l_arm'])
        r_arm_mask = np.isin(labels, grouping_info['r_arm'])
        l_palm_mask = np.isin(labels, grouping_info['l_palm'])
        r_palm_mask = np.isin(labels, grouping_info['r_palm'])

        thumb_mask = np.isin(labels, grouping_info['thumb'])
        index_mask = np.isin(labels, grouping_info['index'])
        middle_mask = np.isin(labels, grouping_info['middle'])
        ring_mask = np.isin(labels, grouping_info['ring'])

        hand_mask = np.logical_or.reduce([thumb_mask, index_mask, middle_mask, ring_mask, r_palm_mask])

        # (N, 11) == (N, xyz + 8masks)
        cloud = np.concatenate([pc, 
                                r_handle_mask,
                                l_handle_mask,
                                instance_body_mask,
                                l_arm_mask,
                                r_arm_mask,
                                l_palm_mask,
                                r_palm_mask,
                                hand_mask], axis=1)

    return cloud

def add_gaussian_noise(cloud: np.ndarray, np_random: np.random.RandomState, noise_level=1):
    # cloud is (n, 3)
    num_points = cloud.shape[0]
    multiplicative_noise = 1 + np_random.randn(num_points)[:, None] * 0.01 * noise_level  # (n, 1)
    return cloud * multiplicative_noise