from torch.utils.data import Dataset
from tqdm import tqdm
import numpy as np
from os.path import join as pjoin
import torch


# train set: data/faucet_img/train.npy
class SemSegDataset(Dataset):
    def __init__(self, root_dir='data/laptop', split='train', use_img=True, only_img=False, point_channel=3, half=False):
        self.root_dir = root_dir
        self.half = half
        self.split = split
        self.data = self.load_data()
        self.use_img = use_img
        self.only_img = only_img
        self.point_channel = point_channel
        self.labelweights = np.ones(4)

    def load_data(self):
        if self.split == 'train':
            if self.half:
                data = np.load(pjoin(self.root_dir, 'train_half.npy'), 'r')
            else:
                data = np.load(pjoin(self.root_dir, 'train.npy'), 'r')
        elif self.split == 'val':
            data = np.load(pjoin(self.root_dir, 'val.npy'), 'r')
        elif self.split == 'test':
            data = np.load(pjoin(self.root_dir, 'test.npy'), 'r')
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        num_point, num_channel = sample.shape

        if self.use_img:
            points = sample[:, 0:self.point_channel]
            # argmax will confuse with all 0 and first class
            # add a dummy class to avoid this (range [0,8], 8 for dummy class)
            labels = np.argmax(
                np.concatenate([sample[:, 3:],np.ones((num_point, 1))],axis=1), 
                axis=1)
            
        if self.only_img:
            # only use image as input
            points = sample[512:, 0:self.point_channel]
            labels = np.argmax(
                np.concatenate([sample[512:, 3:],np.ones((160, 1))],axis=1),
                axis=1)
            
        else:
            # only use camera-captured point cloud as input
            points = sample[0:512, 0:self.point_channel]  
            labels = np.argmax(
                np.concatenate([sample[0:512, 3:],np.ones((512, 1))],axis=1),
                axis=1)
        return torch.tensor(points), torch.tensor(labels)


# test
if __name__ == '__main__':
    import numpy as np
    import open3d as o3d
    import matplotlib.pyplot as plt
    from icecream import ic, install

    install()
    dataset = SemSegDataset(split='train')
    for i in tqdm(range(len(dataset))):
        idx = np.random.randint(0, len(dataset))
        pc, label = dataset[idx]
        colors = plt.get_cmap("tab20")(label/8).reshape(-1, 4)

        obs_cloud = o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(pc[..., 0:3]))
        obs_cloud.colors = o3d.utility.Vector3dVector(colors[:, 0:3])
        # draw the axis
        coordinate = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05, origin=[0, 0, 0])

        o3d.visualization.draw_geometries([obs_cloud, coordinate])