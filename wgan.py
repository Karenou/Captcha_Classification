import torch.nn as nn
import numpy as np
from nn_block import conv_block, cnn_trans_block, fc_block, ResnetBlock

class Generator(nn.Module):
    def __init__(self, num_channel=3):
        super(Generator, self).__init__()
        # input z: (64, 3, 44, 140)
        self.encoder = nn.Sequential(
            *conv_block(num_channel, 16, kernel_size=3, stride=1, padding=0, normalize="instance", activation="relu"),
            *conv_block(16, 32, kernel_size=3, stride=2, padding=0, normalize="instance", activation="relu"),
            *conv_block(32, 64, kernel_size=3, stride=2, padding=0, normalize="instance", activation="relu")
        )

        self.bottle_neck = nn.Sequential(
                ResnetBlock(64), 
                ResnetBlock(64), 
                ResnetBlock(64), 
                ResnetBlock(64)
        )

        self.decoder = nn.Sequential(
            *cnn_trans_block(64, 32, kernel_size=3, stride=2, padding=0, normalize="instance", activation="relu"),
            *cnn_trans_block(32, 16, kernel_size=3, stride=2, padding=0, normalize="instance", activation="relu"),
            *cnn_trans_block(16, num_channel, kernel_size=6, stride=1, padding=0, activation="tanh")
        )

    def forward(self, z):
        img = self.encoder(z)
        img = self.bottle_neck(img)
        img = self.decoder(img)
        return img


class Discriminator(nn.Module):
    def __init__(self, num_channel=3):
        super(Discriminator, self).__init__()

        self.conv_layer = nn.Sequential(
            *conv_block(num_channel, 8, kernel_size=3, stride=1, padding=1, normalize=None, activation="leakyrelu", alpha=0.2),
            nn.MaxPool2d(2),
            *conv_block(8, 16, kernel_size=3, stride=1, padding=0, normalize="batch", activation="leakyrelu", alpha=0.2),
            nn.MaxPool2d(2),
            *conv_block(16, 32, kernel_size=3, stride=1, padding=0, normalize="batch", activation="leakyrelu", alpha=0.2),
            nn.MaxPool2d(2),
            # to reduce channel from 32 to 1
            *conv_block(32, 1, kernel_size=1, stride=1, padding=0, normalize=None, activation=None)
        )
        
        self.fc_layer = nn.Sequential(
            *fc_block(64, 1, normalize=None, activation="sigmoid", dropout=0.2)
        )

    def forward(self, x):
        x = self.conv_layer(x)
        # flatten
        x = x.view(x.size(0), -1)
        x = self.fc_layer(x)
        return x