from __future__ import annotations

from typing import List, Tuple

import pytorch_lightning as pl
from pytorch_lightning.metrics.functional import accuracy
import torch
import torch.optim as optim
import torch.nn as nn
from torch.nn import functional as F


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

    def forward(self, x: torch.tensor):
        return self.backbone(x)

    def cross_entropy_loss(self, logits: torch.tensor, labels: torch.tensor):
        return F.nll_loss(logits, labels)

    def training_step(
        self,
        batch: torch.Tensor | Tuple[torch.Tensor] | List[torch.Tensor],
        batch_idx: int
    ):
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
        return loss

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
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
