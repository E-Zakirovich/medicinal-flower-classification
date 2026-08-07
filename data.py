"""
data.py
~~~~~~~~

I created a file called data.py in order to load the dataset to
my convolutional neural network through main.py file. Inside of
this file I will ise augmentation, random splitting, subset ope-
rations.
"""

import torch
from torch.utils.data import Subset, DataLoader, random_split
from torchvision import datasets, transforms
import config


class Dataset:
    def __init__(self):

        """
        here is the place where augmentation happens. following
        code is transformation for train dataset
        """

        self.train_dataset_transform = transforms.Compose([
            # I am changing the size of the image
            transforms.Resize(
                (
                    config.image_size, # image size
                    config.image_size  # image size
                )
            ),

            # half of the dataset images are flipped horizontally,
            transforms.RandomHorizontalFlip(p = config.horizontal_flip),

            # half of the dataset images are flipped vertically,
            transforms.RandomHorizontalFlip(p=config.vertical_flip),

            # all dataset images rotated between -a to a degrees
            transforms.RandomRotation(config.angle),

            # I will make a tensor from images
            transforms.ToTensor(),

            # normalization part
            transforms.Normalize(
                mean = config.mean, # mean
                std = config.std, # config
            )
        ]
        )