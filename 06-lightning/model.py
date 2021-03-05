from __future__ import annotations

from typing import List, Tuple

import pytorch_lightning as pl
from pytorch_lightning.metrics.functional import accuracy
import torch
import torch.optim as optim
import torch.nn as nn
from torch.nn import functional as F
import matplotlib.pyplot as plt


class MNISTClassifier(nn.Module):

    def __init__(self):
        super(MNISTClassifier, self).__init__()

        # mnist images are (1, 28, 28) (channels, width, height)
        self.layer_1 = nn.Linear(28 * 28, 128)
        self.layer_2 = nn.Linear(128, 256)
        self.layer_3 = nn.Linear(256, 10)

    def forward(self, x):
        batch_size, channels, width, height = x.size()

        # (b, 1, 28, 28) -> (b, 1*28*28)
        x = x.view(batch_size, -1)

        # layer 1
        x = self.layer_1(x)
        x = F.relu(x)

        # layer 2
        x = self.layer_2(x)
        x = F.relu(x)

        # layer 3
        x = self.layer_3(x)

        # probability distribution over labels
        x = F.log_softmax(x, dim=1)

        return x


class LightningMNISTClassifier(pl.LightningModule):

    def __init__(self, backbone: nn.Module = None, lr: float = 1e-3):
        super().__init__()
        if backbone is None:
            backbone = MNISTClassifier()
        self.backbone = backbone
        self.lr = lr
        self.hparams = {
            'learning_rate': self.lr
        }

    def forward(self, x: torch.tensor) -> torch.tensor:
        return self.backbone(x)

    def cross_entropy_loss(
        self, logits: torch.tensor, labels: torch.tensor
    ) -> torch.tensor:
        return F.nll_loss(logits, labels)

    def training_step(
        self,
        batch: torch.Tensor | Tuple[torch.Tensor] | List[torch.Tensor],
        batch_idx: int
    ) -> torch.tensor:
        x, y = batch
        logits = self(x)
        loss = self.cross_entropy_loss(logits, y)
        y_hat = torch.argmax(logits, dim=1)
        acc = accuracy(y_hat, y)
        self.log('train_loss', loss)
        self.log('train_acc', acc, prog_bar=True)
        return loss

    def validation_step(
        self,
        batch: torch.Tensor | Tuple[torch.Tensor] | List[torch.Tensor],
        batch_idx: int
    ):
        x, y = batch
        logits = self(x)
        loss = self.cross_entropy_loss(logits, y)
        y_hat = torch.argmax(logits, dim=1)
        acc = accuracy(y_hat, y)
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)
        if batch_idx == 0:
            return (x, y, y_hat)

    def validation_epoch_end(self, outputs):
        x, y, y_hat = outputs[0]
        x, y, y_hat = x.cpu(), y.cpu(), y_hat.cpu()
        x, y, y_hat = x[:10], y[:10], y_hat[:10]
        rows, cols = (2, 5)

        fig, ax = plt.subplots(rows, cols, figsize=(12, 4))

        i = 0
        j = 0
        for (img, label, predicted) in zip(x, y, y_hat):
            ax[i][j].imshow(img[0, :, :], cmap='gray')
            title = f'Label: {label} | Predicted: {predicted}'
            ax[i][j].set_title(title)
            ax[i][j].grid(False)
            ax[i][j].set_xticklabels([])
            ax[i][j].set_yticklabels([])
            j += 1
            if j >= cols:
                i += 1
                j = 0
        self.logger.experiment.add_figure('example_images', fig, self.global_step)

    def test_step(
        self,
        batch: torch.Tensor | Tuple[torch.Tensor] | List[torch.Tensor],
        batch_idx: int
    ):
        x, y = batch
        logits = self(x)
        loss = self.cross_entropy_loss(logits, y)
        y_hat = torch.argmax(logits, dim=1)
        acc = accuracy(y_hat, y)
        self.log('test_loss', loss)
        self.log('test_acc', acc, prog_bar=True)

    def configure_optimizers(self) -> optim.Optimizer:
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
