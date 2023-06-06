import torch
import lightning as pl
from models.pose_losses import CameraPoseLoss
from util import utils
import logging


class BasePoseLightningModule(pl.LightningModule):
    def __init__(self, config):
        super().__init__()

        self._cfg = config
        self._backbone = torch.load(self._cfg.get('backbone_path'))
        self._pose_loss = CameraPoseLoss(self._cfg.get('loss')).cuda()

    def forward(self):
        pass

    def training_step(self, batch, batch_idx):
        input, pose_gt = batch.get('img'), batch.get('pose')
        output = self.forward(input)
        loss = self._pose_loss(output, pose_gt)
        self.log("train_loss", loss, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        input, pose_gt = batch.get('img'), batch.get('pose')
        output = self.forward(input)
        loss = self._pose_loss(output, pose_gt)
        self.log("val_loss", loss, prog_bar=True, sync_dist=True)

        posit_err, orient_err = utils.pose_err(output.double(), pose_gt)
        logging.info("[Epoch-{}] validation camera pose loss: {:.3f}, "
                     "Mean camera pose error: {:.2f}[m], {:.2f}[deg]".format(
            self.current_epoch + 1, loss.item(), posit_err.mean().item(), orient_err.mean().item()))

        return loss

    def test_step(self, batch, batch_idx):
        input, pose_gt = batch.get('img'), batch.get('pose')
        output = self.forward(input)
        loss = self._pose_loss(output, pose_gt)
        self.log("test_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self._cfg.get('lr'))
        return optimizer
