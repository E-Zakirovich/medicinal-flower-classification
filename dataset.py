import torch
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import transforms, datasets
import config


class Data:
    def __init__(self):
        ...