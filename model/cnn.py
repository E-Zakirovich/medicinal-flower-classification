import torch
import torch.nn as nn

import config


class CNN(nn.Module):

    def __init__(self):
        super(CNN, self).__init__()

        self.convolution_layer_one = nn.Conv2d(
            in_channels = 3,
            out_channels = 32,
            kernel_size = 3,
            padding = 1,
            stride = 1
        )

        self.batch_normalization_one = nn.BatchNorm2d(32)

        self.convolution_layer_two = nn.Conv2d(
            in_channels = 32,
            out_channels = 64,
            kernel_size = 3,
            padding = 1,
            stride = 1
        )
        self.batch_normalization_two = nn.BatchNorm2d(64)

        self.convolution_layer_three = nn.Conv2d(
            in_channels = 64,
            out_channels = 128,
            kernel_size = 3,
            padding = 1,
            stride = 1
        )
        self.batch_normalization_three = nn.BatchNorm2d(128)

        self.convolution_layer_four = nn.Conv2d(
            in_channels = 128,
            out_channels = 256,
            kernel_size = 3,
            padding = 1,
        )
        self.batch_normalization_four = nn.BatchNorm2d(256)

        self.pool = nn.MaxPool2d(
            kernel_size = 2,
            stride = 2
            )

        self.relu = nn.ReLU()

        self.input_layer_and_hidden_layer = nn.Linear(config.input_layer, config.hidden_layer)
        self.dropout = nn.Dropout(p = config.dropout)
        self.hidden_layer_and_output_layer = nn.Linear(config.hidden_layer, config.output_layer)


    def forward(self, x):
        x = self.pool(self.relu(self.batch_normalization_one(self.convolution_layer_one(x))))
        x = self.pool(self.relu(self.batch_normalization_two(self.convolution_layer_two(x))))
        x = self.pool(self.relu(self.batch_normalization_three(self.convolution_layer_three(x))))
        x = self.pool(self.relu(self.batch_normalization_four(self.convolution_layer_four(x))))

        x = x.view(x.size(0), -1)

        x = self.relu(self.input_layer_and_hidden_layer(x))
        x = self.dropout(x)
        x = self.hidden_layer_and_output_layer(x)
        return x