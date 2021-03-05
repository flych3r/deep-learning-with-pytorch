from argparse import ArgumentParser

import pytorch_lightning as pl
import torch
from data import MNISTDataModule
from model import LightningMNISTClassifier

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--max_epochs', default=5, type=int)
    parser.add_argument('--gpus', default=None, type=int)
    args = parser.parse_args()

    model = LightningMNISTClassifier(lr=args.lr)
    mnist_data = MNISTDataModule()

    gpus = args.gpus if torch.cuda.is_available() else None
    trainer = pl.Trainer(max_epochs=args.max_epochs, default_root_dir='runs', gpus=gpus)

    trainer.fit(model, mnist_data)
    trainer.test(model)
