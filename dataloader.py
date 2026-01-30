import pandas as pd
import numpy as np
import torch
from PIL import Image
import io
import matplotlib.pyplot as plt
import torchvision

class DataLoader():
    def __init__(self, cfg):
        self.cfg = cfg
        self.parquet_data = None
        self.labels = None
        self.images = None
        self.labelNames = {
            0: 'blues', 1: 'classical', 2: 'country', 3: 'deathmetal', 4: 'doommetal',
            5: 'drumnbass', 6: 'electronic', 7: 'folk', 8: 'grime', 9: 'heavymetal',
            10: 'hiphop', 11: 'jazz', 12: 'lofi', 13: 'pop', 14: 'psychedelicrock',
            15: 'punk', 16: 'reggae', 17: 'rock', 18: 'soul', 19: 'techno'
        }

        self.train_images = None
        self.train_labels = None
        self.val_images = None
        self.val_labels = None
        self.test_images = None
        self.test_labels = None

    def download_data(self):
        print("Downloading data...")
        self.parquet_data = pd.read_parquet(
            "hf://datasets/eong/20k-Album-Covers-within-20-Genres/data/train-00000-of-00001-f37f5042abc5be8d.parquet"
        )

    def test_download(self):
        print("Testing download...")
        image_data = self.parquet_data.loc[12, 'image']['bytes']
        image = Image.open(io.BytesIO(image_data))
        plt.imshow(image)
        plt.show()

    def process_to_tensors(self, verbose=False):
        print("Converting data to tensors...")
        n_samples = len(self.parquet_data)
        labels = torch.ones((n_samples, 1))
        size = self.cfg['input_size']
        im_tensors = torch.empty((n_samples, 3, size, size))
        to_tensor = torchvision.transforms.Compose([
            torchvision.transforms.Resize((size, size)),
            torchvision.transforms.ToTensor()
        ])

        for i in range(n_samples):
            if i % 1000 == 0:
                print(i)
            labels[i] = self.parquet_data.loc[i, 'label']
            im_bytes = self.parquet_data.loc[i, 'image']['bytes']
            image = Image.open(io.BytesIO(im_bytes)).convert("RGB")
            image_tensor = to_tensor(image)
            im_tensors[i] = image_tensor
            if (i == 0 or i == n_samples - 1) and verbose:
                plt.imshow(image)
                plt.show()

        self.labels = labels
        self.images = im_tensors

    def verify_process(self, im_ind=None):
        print("Verifying processing...")
        if im_ind is None:
            image_tensor = self.images[8908]
        else:
            image_tensor = self.images[im_ind]
        to_pil = torchvision.transforms.ToPILImage()
        image = to_pil(image_tensor)
        plt.imshow(image)
        plt.show()

    def create_train_val_test(self):
        n = self.labels.shape[0]
        torch.manual_seed(self.cfg['train']['seed'])
        shuffle = torch.randperm(n)

        train_split = 1 - self.cfg['train']['val_split'] - self.cfg['train']['test_split']
        val_split = self.cfg['train']['val_split']

        train_end = int(train_split * n)
        val_end = int((train_split + val_split) * n)

        train_idx = shuffle[:train_end]
        val_idx = shuffle[train_end:val_end]
        test_idx = shuffle[val_end:]

        self.train_images = self.images[train_idx]
        self.train_labels = self.labels[train_idx]
        self.val_images = self.images[val_idx]
        self.val_labels = self.labels[val_idx]
        self.test_images = self.images[test_idx]
        self.test_labels = self.labels[test_idx]

    def return_train_data(self):
        return self.train_images, self.train_labels

    def return_val_data(self):
        return self.val_images, self.val_labels

    def return_test_data(self):
        return self.test_images, self.test_labels

    def main(self):
        self.download_data()
        self.test_download()
        self.process_to_tensors(verbose=False)
        self.verify_process()
        self.create_train_val_test()