import pytorch_lightning as pl
import torch
from torch import nn
from segmentation_models_pytorch import Linknet
import math

from metrics import rate, epe, rmse, bad

from constants import LEARNING_RATE, MAX_DISP, K, P, S


def basic_block(in_channels: int, out_channels: int, k=None, p=None, s=None):
    return nn.Sequential(
        nn.Conv3d(in_channels, out_channels, kernel_size=k, padding=p, stride=s),
        nn.BatchNorm3d(out_channels),
        nn.LeakyReLU(inplace=True))


class DepthMap(pl.LightningModule):
    def __init__(self, encoder_name='resnet18', encoder_depth=5):
        super().__init__()

        # Feature extraction
        self.feature_extractor = Linknet(
            encoder_name=encoder_name,
            encoder_weights="imagenet",
            in_channels=3,
            encoder_depth=encoder_depth,
        )

        # Stereo matching
        self.disparity_regressor = nn.Sequential(
            nn.Sequential(
                basic_block(2, 4, K, P, S),
                basic_block(4, 4, K, P, S)),
            nn.Sequential(
                basic_block(4, 8, K, P, S),
                basic_block(8, 8, K, P, S)),
            nn.Sequential(
                basic_block(8, 8, K, P, S),
                basic_block(8, 8, K, P, S)),
            nn.Sequential(
                basic_block(8, 8, K, P, S),
                basic_block(8, 4, K, P, S)),
            nn.Sequential(
                basic_block(4, 4, K, P, S),
                nn.Conv3d(4, 1, kernel_size=K, padding=P, stride=S)))

        self.leaky_relu = nn.LeakyReLU()

        self.loss_fn = nn.SmoothL1Loss()

        self.dropout = nn.Dropout(0.2)

        # Weights initialization
        for m in self.disparity_regressor.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

        self.save_hyperparameters()

    def forward(self, left, right):
        # LinkNet
        left = self.feature_extractor(left)
        right = self.feature_extractor(right)

        # Concat
        features = torch.cat([left, right], dim=1).unsqueeze(2)

        # Stereo matching
        x = self.disparity_regressor(features)

        x = x.squeeze(1)

        x = self.leaky_relu(x)

        return x

    def training_step(self, batch, batch_idx):
        left, right, gt_disparity = batch['left_image'], batch['right_image'], batch['disparity']

        # Forward pass
        predicted_map = self(left, right)

        # Compute loss
        mask = gt_disparity < MAX_DISP
        loss = self.loss_fn(predicted_map[mask], gt_disparity[mask])

        self.log_dict({'loss (train)': loss,
                       'RATE1 (train)': rate(predicted_map, gt_disparity, mask, 1.0),
                       'RATE3 (train)': rate(predicted_map, gt_disparity, mask, 3.0),
                       'RATE5 (train)': rate(predicted_map, gt_disparity, mask, 5.0),
                       'EPE (train)': epe(predicted_map, gt_disparity, mask),
                       'BAD 0.5 (train)': bad(predicted_map, gt_disparity, 0.5),
                       'BAD 1.0 (train)': bad(predicted_map, gt_disparity, 1.0),
                       'BAD 2.0 (train)': bad(predicted_map, gt_disparity, 2.0),
                       'BAD 4.0 (train)': bad(predicted_map, gt_disparity, 4.0),
                       'RMSE (train)': rmse(predicted_map, gt_disparity)})
        if batch_idx % 100 == 0:
            self.log_image(key="samples (train)",
                           images=[left[:4], gt_disparity[:4], predicted_map[:4]],
                           caption=["scene", "ground truth", 'predicted'])

        return loss

    def validation_step(self, batch, batch_idx):
        left, right, gt_disparity = batch['left_image'], batch['right_image'], batch['disparity']

        # Forward pass
        predicted_map = self(left, right)

        #
        mask = gt_disparity < MAX_DISP
        loss = self.loss_fn(predicted_map[mask], gt_disparity[mask])

        self.log_dict({'loss (val)': loss,
                       'RATE1 (val)': rate(predicted_map, gt_disparity, mask, 1.0),
                       'RATE3 (val)': rate(predicted_map, gt_disparity, mask, 3.0),
                       'RATE5 (val)': rate(predicted_map, gt_disparity, mask, 5.0),
                       'EPE (val)': epe(predicted_map, gt_disparity, mask),
                       'BAD 0.5 (val)': bad(predicted_map, gt_disparity, 0.5),
                       'BAD 1.0 (val)': bad(predicted_map, gt_disparity, 1.0),
                       'BAD 2.0 (val)': bad(predicted_map, gt_disparity, 2.0),
                       'BAD 4.0 (val)': bad(predicted_map, gt_disparity, 4.0),
                       'RMSE (val)': rmse(predicted_map, gt_disparity)})

        if batch_idx % 100 == 0:
            self.log_image(key="samples (val)",
                           images=[left[0], gt_disparity[0], predicted_map[0]],
                           caption=["scene", "ground truth", 'predicted'])
        return loss

    def test_step(self, batch, batch_idx):
        pass

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=LEARNING_RATE)
        return optimizer
