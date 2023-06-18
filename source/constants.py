TRAIN_ROOT = 'D:/StereoDrivingDataset/train/'
TRAIN_LEFT = TRAIN_ROOT + 'left/'
TRAIN_RIGHT = TRAIN_ROOT + 'right/'
TRAIN_DISPARITY = TRAIN_ROOT + 'disparity/'

TEST_ROOT = 'D:/StereoDrivingDataset/test/'
TEST_LEFT = TEST_ROOT + 'left_image/'
TEST_RIGHT = TEST_ROOT + 'right_image/'
TEST_DISPARITY = TEST_ROOT + 'disparity_map/'
MIDDLEBURY_ROOT = 'D:/Middlebury/data/'

IMG_WIDTH = 128
IMG_HEIGHT = 64
N_CHANNELS = 3
BATCH_SIZE = 16
MAX_DISP = 192
LEARNING_RATE = 1e-3
K = (3, 3, 7)
P = (1, 1, 3)
S = (1, 1, 1)
