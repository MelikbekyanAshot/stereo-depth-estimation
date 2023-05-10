from constants import TRAIN_LEFT, TRAIN_RIGHT, TRAIN_DISPARITY, TEST_RIGHT, TEST_LEFT, TEST_DISPARITY, \
    IMG_WIDTH, IMG_HEIGHT, BATCH_SIZE
from dataset.stereo_dataset import StereoDataset
from model import DepthMap

from torch.utils.data import DataLoader

import torchvision.transforms as tfs

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

import wandb


if __name__ == '__main__':
    model = DepthMap()

    wandb_logger = WandbLogger(
        project='depth-map-estimation',
        name='Linknet(resnet50), k=(3x3x7), img=64x128')

    transforms = tfs.Compose([
        tfs.ToPILImage(),
        tfs.Resize(IMG_WIDTH),
        tfs.CenterCrop((IMG_HEIGHT, IMG_WIDTH)),
        tfs.ToTensor(),
    ])
    disp_transforms = tfs.Compose([
        tfs.ToPILImage(),
        tfs.Resize((IMG_HEIGHT, IMG_WIDTH)),
        tfs.CenterCrop((IMG_HEIGHT, IMG_WIDTH)),
        tfs.ToTensor(),
    ])

    train_stereo_ds = StereoDataset(dir_left=TRAIN_LEFT, dir_right=TRAIN_RIGHT,
                                    dir_disparity=TRAIN_DISPARITY,
                                    transform=transforms, disp_transform=disp_transforms)
    train_dl = DataLoader(train_stereo_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, )

    test_stereo_ds = StereoDataset(dir_left=TEST_LEFT, dir_right=TEST_RIGHT,
                                   dir_disparity=TEST_DISPARITY,
                                   transform=transforms, disp_transform=disp_transforms)
    test_dl = DataLoader(test_stereo_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, )

    trainer = pl.Trainer(
        accelerator='gpu',
        max_epochs=1,
        #     callbacks=[StochasticWeightAveraging(swa_lrs=1e-2)],
        logger=wandb_logger,
        #     accumulate_grad_batches=7,
        #     gradient_clip_val=0.5, gradient_clip_algorithm="value"
    )

    trainer.fit(model, train_dl, test_dl)
    wandb.finish()