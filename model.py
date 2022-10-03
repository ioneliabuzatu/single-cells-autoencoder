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

        self.encode = nn.Sequential(
            nn.Linear(in_shape, 500),
            # nn.SELU(True),
            # nn.Dropout(0.2),
            # nn.Linear(500, 128),
            nn.SELU(True),
            nn.Dropout(0.2),
            nn.Linear(500, 64),
            nn.SELU(True),
            nn.Dropout(0.2),
            nn.Linear(64, enc_shape),
        )
        self.decode = nn.Sequential(
            nn.BatchNorm1d(enc_shape),
            nn.Linear(enc_shape, 64),
            nn.SELU(True),
            nn.Dropout(0.2),
            nn.Linear(64, 128),
            nn.SELU(True),
            nn.Dropout(0.2),
            nn.Linear(128, 500),
            nn.SELU(True),
            nn.Dropout(0.2),
            nn.Linear(500, in_shape)
        )

    def forward(self, x):
        x = self.encode(x)
        x = self.decode(x)
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