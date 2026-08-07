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


class Dataset:
    def __init__(self):
        ...