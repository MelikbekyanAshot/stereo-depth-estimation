import os
from torch.utils.data import Dataset
from skimage import io
import numpy as np


class StereoDataset(Dataset):
    """Custom pytorch dataset class implementation for Stereo Driving dataset."""
    def __init__(self, dir_left, dir_right, dir_disparity, transform, disp_transform):
        assert len(os.listdir(dir_left)) == len(os.listdir(dir_right)) == len(os.listdir(dir_disparity))
        self.dir_left = dir_left
        self.dir_right = dir_right
        self.dir_disparity = dir_disparity

        self.left = sorted(os.listdir(dir_left))
        self.right = sorted(os.listdir(dir_right))
        self.disparity = sorted(os.listdir(dir_disparity))
        self.transform = transform
        self.disp_transform = disp_transform

    def __len__(self):
        return len(self.left)

    def __getitem__(self, idx):
        # Left
        left_image = io.imread(self.dir_left + self.left[idx])
        left_image = self.transform(left_image)

        # Right
        right_image = io.imread(self.dir_right + self.right[idx])
        right_image = self.transform(right_image)

        # Disparity
        disparity_img = io.imread(self.dir_disparity + self.disparity[idx])
        disparity = np.ascontiguousarray(disparity_img, dtype=np.float32) / 256.0
        disparity = self.disp_transform(disparity)

        sample = {
            'left_image': left_image,
            'right_image': right_image,
            'disparity': disparity,
        }

        return sample
