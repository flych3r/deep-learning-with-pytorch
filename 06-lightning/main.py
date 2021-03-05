from argparse import ArgumentParser
from model import LightningMNISTClassifier
from data import MNISTDataModule
import pytorch_lightning as pl
import torch


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--max_epochs', default=5, type=int)
    parser.add_argument('--gpus', default=None, type=int)
    args = parser.parse_args()

    model = LightningMNISTClassifier()
    mnist_data = MNISTDataModule()

    gpus = args.gpus if torch.cuda.is_available() else None
    trainer = pl.Trainer(max_epochs=args.max_epochs, default_root_dir='runs', gpus=gpus)

    trainer.fit(model, mnist_data)
    trainer.test(model)
