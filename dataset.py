import torch
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import transforms, datasets
import config


class Data:
    def __init__(self):

        self.transform_dataset = transforms.Compose([

            # make the size of the images same for cnn & nn
            transforms.Resize(
                (config.image_size, config.image_size)
                ),

            # flip & rotate images in order to avoid underfitting & overfitting
            transforms.RandomHorizontalFlip(p = config.flip_amount),
            transforms.RandomVerticalFlip(p = config.flip_amount),
            transforms.RandomRotation(degrees = config.rotation_amount),

            # make a tensor for cnn
            transforms.ToTensor(),

            # normalization
            transforms.Normalize(
                mean = (0.485, 0.456, 0.406),
                std = (0.229, 0.224, 0.225)
                )
            ]
        )

        self.validation_and_test_transformation = transforms.Compose([

            # make the size of the images same for cnn & nn
            transforms.Resize(
                (config.image_size, config.image_size)
            ),

            # make a tensor for cnn
            transforms.ToTensor(),

            # normalization
            transforms.Normalize(
                mean = (0.485, 0.456, 0.406),
                std = (0.229, 0.224, 0.225)
            )
        ])