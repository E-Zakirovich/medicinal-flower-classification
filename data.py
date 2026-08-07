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

        """
        transformation for validation and test dataset.
        """

        self.validation_and_test_transformation = transforms.Compose([
            # I am changing the size of the image
            transforms.Resize(
                (
                    config.image_size,  # image size
                    config.image_size  # image size
                )
            ),

            # I will make a tensor from images
            transforms.ToTensor(),

            # normalization part
            transforms.Normalize(
                mean=config.mean,  # mean
                std=config.std,  # config
            )
        ]
        )

    """
    I need to load the dataset with following method.
    """


    def __load_dataset(self):

        # Build train, validation and test datasets with their respective transforms
        train_data = datasets.ImageFolder(
            root = config.path, # path of the dataset
            transform = self.train_dataset_transform # connection to transform code
        )
        validation_data = datasets.ImageFolder(
            root = config.path,
            transform = self.validation_and_test_transformation
        )
        test_data = datasets.ImageFolder(
            root = config.path,
            transform = self.validation_and_test_transformation
        )