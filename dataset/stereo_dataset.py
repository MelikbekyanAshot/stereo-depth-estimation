import os
from torch.utils.data import Dataset
from skimage import io


ROOT_LEFT = 'D:/StereoDrivingDataset/left/'
ROOT_RIGHT = 'D:/StereoDrivingDataset/right/'
ROOT_DISPARITY = 'D:/StereoDrivingDataset/disparity/'


class StereoDataset(Dataset):
    """Custom pytorch dataset class implementation for Stereo Driving dataset."""
    def __init__(self, transform):
        assert len(os.listdir(ROOT_LEFT)) == len(os.listdir(ROOT_RIGHT)) == len(os.listdir(ROOT_DISPARITY))
        self.left = sorted(os.listdir(ROOT_LEFT))
        self.right = sorted(os.listdir(ROOT_RIGHT))
        self.disparity = sorted(os.listdir(ROOT_DISPARITY))
        self.transform = transform

    def __len__(self):
        return len(os.listdir(ROOT_LEFT))

    def __getitem__(self, idx):
        # Left
        left_img = io.imread(ROOT_LEFT + self.left[idx])
        left = self.transform(left_img)

        # Right
        right_img = io.imread(ROOT_RIGHT + self.right[idx])
        right = self.transform(right_img)

        # Disparity
        disparity_img = io.imread(ROOT_DISPARITY + self.disparity[idx])
        disparity = disparity_img.astype(np.float32) / 256.
        disparity = self.transform(disparity)

        return left, right, disparity
