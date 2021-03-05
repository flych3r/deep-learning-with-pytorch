import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import MNIST
import os

PATH = './data'
N_WORKERS = os.cpu_count()


def mnist_dataloaders():
    # ----------------
    # TRANSFORMS
    # ----------------
    # prepare transforms standard to MNIST
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # ----------------
    # TRAINING, VAL DATA
    # ----------------
    mnist_train = MNIST(PATH, train=True, download=True, transform=transform)

    # train (55,000 images), val split (5,000 images)
    mnist_train, mnist_val = random_split(mnist_train, [55000, 5000])

    # ----------------
    # TEST DATA
    # ----------------
    mnist_test = MNIST(PATH, train=False, download=True, transform=transform)

    # ----------------
    # DATALOADERS
    # ----------------
    # The dataloaders handle shuffling, batching, etc...
    mnist_train = DataLoader(mnist_train, batch_size=64)
    mnist_val = DataLoader(mnist_val, batch_size=64)
    mnist_test = DataLoader(mnist_test, batch_size=64)
    return mnist_train, mnist_val, mnist_test


class MNISTDataModule(pl.LightningDataModule):

    def __init__(self, data_dir=PATH, batch_size=64, num_workers=N_WORKERS):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        # transforms for images
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        # prepare transforms standard to MNIST
        mnist_train = MNIST(self.data_dir, train=True, download=True, transform=transform)
        mnist_test = MNIST(self.data_dir, train=False, download=True, transform=transform)

        self.mnist_train, self.mnist_val = random_split(mnist_train, [55000, 5000])
        self.mnist_test = mnist_test

    def train_dataloader(self):
        return DataLoader(
            self.mnist_train, batch_size=self.batch_size, num_workers=self.num_workers
        )

    def val_dataloader(self):
        return DataLoader(
            self.mnist_val, batch_size=self.batch_size, num_workers=self.num_workers
        )

    def test_dataloader(self):
        return DataLoader(
            self.mnist_test, batch_size=self.batch_size, num_workers=self.num_workers
        )
