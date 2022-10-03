"""initial model taken from: https://github.com/techshot25/Autoencoders/blob/master/simple-autoencoder.ipynb"""
# another init model notebook useful: https://github.com/camilo-cf/Autoencoder-PCA/blob/main/Simple_AutoEncoder.ipynb
import torch
from torch import nn, optim
import numpy as np
from matplotlib import pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
from sklearn.datasets import make_swiss_roll
from sklearn.preprocessing import MinMaxScaler


class Autoencoder(nn.Module):
    """
    in_shape [int] : input shape
    enc_shape [int] : desired encoded shape
    """

    def __init__(self, in_shape, enc_shape):
        super(Autoencoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(in_shape, 500),
            # nn.SELU(True),
            # nn.Dropout(0.2),
            # nn.Linear(500, 128),
            nn.SELU(True),
            nn.Dropout(0.2),
            nn.Linear(500, enc_shape),
        )
        self.decoder = nn.Sequential(
            nn.BatchNorm1d(enc_shape),
            nn.Linear(enc_shape, 500),
            nn.SELU(True),
            nn.Dropout(0.2),
            nn.Linear(500, in_shape)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    
    
class AutoencoderV2(nn.Module):
    """
    in_shape [int] : input shape
    enc_shape [int] : desired encoded shape
    """

    def __init__(self, in_shape, enc_shape):
        super(AutoencoderV2, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(in_shape, 10012),
            # nn.SELU(inplace=True),
            nn.SELU(),
            nn.Dropout(0.2, inplace=True),
            nn.Linear(10012, 5012),
            # nn.SELU(inplace=True),
            nn.SELU(),
            nn.Dropout(0.2, inplace=True),
            nn.Linear(5012, 1012),
            # nn.SELU(inplace=True),
            nn.SELU(),
            nn.Dropout(0.2, inplace=True),
            nn.Linear(1012, 512),
            # nn.SELU(inplace=True),
            nn.SELU(),
            nn.Dropout(0.2, inplace=True),
            nn.Linear(512, enc_shape),
        )
        self.decoder = nn.Sequential(
            # nn.BatchNorm1d(enc_shape),
            nn.Linear(enc_shape, 512),
            # nn.SELU(inplace=True),
            nn.SELU(),
            nn.Dropout(0.2, inplace=True),
            nn.Linear(512, 1012),
            # nn.SELU(inplace=True),
            nn.SELU(),
            nn.Dropout(0.2, inplace=True),
            nn.Linear(1012, 5012),
            # nn.SELU(inplace=True),
            nn.SELU(),
            nn.Dropout(0.2, inplace=True),
            nn.Linear(5012, 10012),
            # nn.SELU(inplace=True),
            nn.SELU(),
            nn.Dropout(0.2, inplace=True),
            nn.Linear(10_012, in_shape),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    
    
class Regressor(nn.Module):
    """
    in_shape [int] : input shape
    out_shape [int] : prediction vector shape
    """
    def __init__(self, in_shape, out_shape=140):

        super(Regressor, self).__init__()

        self.regressor = nn.Sequential(
            nn.Linear(in_shape, 512),
            nn.SELU(True),
            nn.Dropout(0.2),
            # nn.Linear(500, 64),
            # nn.SELU(True),
            # nn.Dropout(0.2),
            nn.Linear(512, out_shape),
        )

    def forward(self, x):
        x = self.regressor(x)
        return x