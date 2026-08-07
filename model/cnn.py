"""
cnn.py
~~~~~~

I will write the codes convolutional neural network in this file.
Inside  there is four  convolutional  layers connected  to neural
networks.
"""


import torch
from torch import nn
import config
from config import in_out_channels


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        # first convolutional neural network layer
        self.first_convolutional_layer = nn.Conv2d(
            in_channels = config.in_out_channels[0],
            out_channels = config.in_out_channels[1],
            kernel_size = config.kernel_size,
            stride = config.stride,
            padding = config.padding
        )

        # batch normalization
        self.first_batch_normalization = nn.BatchNorm2d(config.in_out_channels[1])

        # second convolutional neural network layer
        self.second_convolutional_layer = nn.Conv2d(
            in_channels=config.in_out_channels[1],
            out_channels=config.in_out_channels[2],
            kernel_size=config.kernel_size,
            stride=config.stride,
            padding=config.padding
        )

        # batch normalization
        self.second_batch_normalization = nn.BatchNorm2d(config.in_out_channels[2])

        # third convolutional neural network layer
        self.third_convolutional_layer = nn.Conv2d(
            in_channels=config.in_out_channels[2],
            out_channels=config.in_out_channels[3],
            kernel_size=config.kernel_size,
            stride=config.stride,
            padding=config.padding
        )

        # batch normalization
        self.third_batch_normalization = nn.BatchNorm2d(config.in_out_channels[3])

        # fourth convolutional neural network layer
        self.first_convolutional_layer = nn.Conv2d(
            in_channels=config.in_out_channels[3],
            out_channels=config.in_out_channels[4],
            kernel_size=config.kernel_size,
            stride=config.stride,
            padding=config.padding
        )

        # batch normalization
        self.fourth_batch_normalization = nn.BatchNorm2d(config.in_out_channels[4])