"""initial model taken from: https://github.com/techshot25/Autoencoders/blob/master/simple-autoencoder.ipynb"""
# another init model notebook useful: https://github.com/camilo-cf/Autoencoder-PCA/blob/main/Simple_AutoEncoder.ipynb
import torch
from torch import nn, optim
import numpy as np
from matplotlib import pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
from sklearn.datasets import make_swiss_roll
from sklearn.preprocessing import MinMaxScaler
from torch.nn import functional as F


class AutoencoderV3(nn.Module):
    """
    in_shape [int] : input shape
    enc_shape [int] : desired encoded shape
    """

    def __init__(self, in_shape, enc_shape):
        super(AutoencoderV3, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(in_shape, 15_000),
            nn.BatchNorm1d(15_000),
            nn.SELU(True),
            nn.Dropout(0.2),

            nn.Linear(35_000, 5000),
            nn.BatchNorm1d(5000),
            nn.SELU(True),
            nn.Dropout(0.2),
            nn.Linear(5000, enc_shape),
        )
        self.decoder = nn.Sequential(
            nn.Linear(enc_shape, 5000),
            nn.BatchNorm1d(5000),
            nn.SELU(True),
            nn.Dropout(0.2),
            nn.Linear(5000, 35_000),
            nn.BatchNorm1d(35_000),
            nn.SELU(True),
            nn.Dropout(0.2),
            nn.Linear(35_000, in_shape)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class Autoencoder(nn.Module):
    """
    in_shape [int] : input shape
    enc_shape [int] : desired encoded shape
    """

    def __init__(self, in_shape, enc_shape):
        super(Autoencoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(in_shape, 1000),
            nn.BatchNorm1d(1000),
            nn.SELU(True),
            nn.Dropout(0.2),
            nn.Linear(1000, 500),
            nn.BatchNorm1d(500),
            nn.SELU(True),
            nn.Dropout(0.2),
            nn.Linear(500, enc_shape),
        )
        self.decoder = nn.Sequential(
            nn.Linear(enc_shape, 1000),
            nn.BatchNorm1d(1000),
            nn.SELU(True),
            nn.Dropout(0.2),
            nn.Linear(1000, 500),
            nn.BatchNorm1d(500),
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

            nn.Linear(in_shape, 15012),
            nn.BatchNorm1d(15012),
            nn.SELU(True),
            nn.Dropout(0.1),

            nn.Linear(15012, 10012),
            nn.BatchNorm1d(10012),
            nn.SELU(True),
            nn.Dropout(0.1),

            nn.Linear(10012, 5012),
            nn.BatchNorm1d(5012),
            nn.SELU(True),
            nn.Dropout(0.1),

            nn.Linear(5012, 1012),
            nn.BatchNorm1d(1012),
            nn.SELU(True),
            nn.Dropout(0.2),

            nn.Linear(1012, 512),
            nn.BatchNorm1d(512),
            nn.SELU(True),
            nn.Dropout(0.1),

            nn.Linear(512, enc_shape),
        )
        self.decoder = nn.Sequential(
            nn.Linear(enc_shape, 512),
            nn.BatchNorm1d(512),
            nn.SELU(True),
            nn.Dropout(0.1),

            nn.Linear(512, 1012),
            nn.BatchNorm1d(1012),
            nn.SELU(True),
            nn.Dropout(0.1),

            nn.Linear(1012, 5012),
            nn.BatchNorm1d(5012),
            nn.SELU(True),
            nn.Dropout(0.2),

            nn.Linear(5012, 10012),
            nn.BatchNorm1d(10_012),
            nn.SELU(True),
            nn.Dropout(0.1),

            nn.Linear(10_012, 15_012),
            nn.BatchNorm1d(15012),
            nn.SELU(True),
            nn.Dropout(0.1),

            nn.Linear(15012, in_shape),
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
            nn.Linear(512, out_shape),
        )

    def forward(self, x):
        x = self.regressor(x)
        return x
    

class SingleFeatureRegressor(nn.Module):
    """
    in_shape [int] : input shape
    out_shape [int] : prediction vector shape
    """
    def __init__(self, in_shape, out_shape=1):

        super(SingleFeatureRegressor, self).__init__()

        self.regressor = nn.Sequential(
            nn.Linear(in_shape, 512),
            nn.SELU(True),
            nn.Dropout(0.2),
            nn.Linear(512, out_shape),
        )

    def forward(self, x):
        x = self.regressor(x)
        return x
    
class VariationalEncoder(nn.Module):
    def __init__(self, latent_dims, device):
        super(VariationalEncoder, self).__init__()
        self.linear1 = nn.Linear(22050, 512)
        self.linear2 = nn.Linear(512, latent_dims)
        self.linear3 = nn.Linear(512, latent_dims)

        self.N = torch.distributions.Normal(0, 1)
        self.N.loc = self.N.loc.to(device)  # hack to get sampling on the GPU
        self.N.scale = self.N.scale.to(device)
        self.kl = 0

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.linear1(x))

        mu = self.linear2(x)
        sigma = torch.exp(self.linear3(x))
        z = mu + sigma*self.N.sample(mu.shape)
        self.kl = (sigma**2 + mu**2 - torch.log(sigma) - 1/2).sum()
        return z
    
    
class Decoder(nn.Module):
    def __init__(self, latent_dims):
        super(Decoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.Linear(latent_dims, 512),
            nn.ReLU(True),
            nn.Linear(512, 22050)
        )
        # z = F.relu(self.linear1(z))
        # z = torch.sigmoid(self.linear2(z)

        # self._decoder = nn.Sequential(
        #     nn.Linear(latent_dims, 5000),
        #     nn.BatchNorm1d(5000),
        #     nn.SELU(True),
        #     nn.Dropout(0.2),
        #     nn.Linear(5000, 22050)
        # )

    def forward(self, z):
        z = torch.sigmoid(self.decoder(z))
        return z

# class Decoder(nn.Module):
#     def __init__(self, latent_dims):
#         super(Decoder, self).__init__()
#         self.linear1 = nn.Linear(latent_dims, 512)
#         self.linear2 = nn.Linear(512, 22050)
#
#     def forward(self, z):
#         z = F.relu(self.linear1(z))
#         z = torch.sigmoid(self.linear2(z))
#         return z.
#


class VariationalAutoencoder(nn.Module):
    def __init__(self, latent_dims, device):
        super(VariationalAutoencoder, self).__init__()
        self.encoder = VariationalEncoder(latent_dims, device)
        self.decoder = Decoder(latent_dims)

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)