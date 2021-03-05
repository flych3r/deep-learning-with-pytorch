import pytorch_lightning as pl
import torch
import torch.optim as optimizer
from torch import nn
from torch.nn import functional as F


class MNISTClassifier(nn.Module):

    def __init__(self):
        super(MNISTClassifier, self).__init__()

        # mnist images are (1, 28, 28) (channels, width, height)
        self.layer_1 = torch.nn.Linear(28 * 28, 128)
        self.layer_2 = torch.nn.Linear(128, 256)
        self.layer_3 = torch.nn.Linear(256, 10)

    def forward(self, x):
        batch_size, channels, width, height = x.size()

        # (b, 1, 28, 28) -> (b, 1*28*28)
        x = x.view(batch_size, -1)

        # layer 1
        x = self.layer_1(x)
        x = torch.relu(x)

        # layer 2
        x = self.layer_2(x)
        x = torch.relu(x)

        # layer 3
        x = self.layer_3(x)

        # probability distribution over labels
        x = torch.log_softmax(x, dim=1)

        return x


class LightningMNISTClassifier(pl.LightningModule):

    def __init__(self, backbone=None, lr=1e-3):
        super().__init__()
        if backbone is None:
            backbone = MNISTClassifier()
        self.backbone = backbone
        self.lr = lr

    def forward(self, x):
        return self.backbone(x)

    def cross_entropy_loss(self, logits, labels):
        return F.nll_loss(logits, labels)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self.backbone(x)
        loss = self.cross_entropy_loss(logits, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self.backbone(x)
        loss = self.cross_entropy_loss(logits, y)
        self.log('val_loss', loss)

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self.backbone(x)
        loss = self.cross_entropy_loss(logits, y)
        labels_hat = torch.argmax(logits, dim=1)
        test_acc = torch.sum(y == labels_hat).item() / (len(y) * 1.0)
        self.log_dict({'test_loss': loss, 'test_acc': test_acc})

    def configure_optimizers(self):
        optim = optimizer.Adam(self.parameters(), lr=self.lr)
        return optim
