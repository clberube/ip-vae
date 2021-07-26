# @Author: cberube
# @Date:   2021-07-26 17:07:50
# @Email:  charles.berube@polymtl.ca
# @Last modified by:   cberube
# @Last modified time: 2021-07-26 17:07:14


import pkg_resources

import torch
import torch.nn as nn


class IPVAE(nn.Module):
    """IP-VAE architecture"""
    def __init__(self):
        """Initializes layers"""
        super(IPVAE, self).__init__()
        # Encoder
        self.fc1 = nn.Linear(20, 16)
        self.fc2 = nn.Linear(16, 8)
        self.fc31 = nn.Linear(8, 2)
        self.fc32 = nn.Linear(8, 2)
        # Decoder
        self.fc4 = nn.Linear(2, 8)
        self.fc5 = nn.Linear(8, 16)
        self.fc6 = nn.Linear(16, 20)

    def encode(self, x):
        """Decodes a latent vector sample.

        Args:
            x (tensor): input

        Returns:
            mu, logvar (tensors), the mean and variance of q(z|x)

        """
        h1 = torch.tanh(self.fc1(x))
        h2 = torch.tanh(self.fc2(h1))
        return self.fc31(h2), self.fc32(h2)

    def reparameterize(self, mu, logvar):
        """Reparametrization trick.

        Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        https://arxiv.org/abs/1312.6114

        Args:
            mu (tensor): the mean of q(z|x)
            logvar (tensor): natural log of the variance of q(z|x)

        Returns:
            z
        """
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        """Decodes a latent vector sample.

        Args:
            z (tensor):

        Returns:
            x' (tensor), reconstructed output

        """
        h4 = torch.tanh(self.fc4(z))
        h5 = torch.tanh(self.fc5(h4))
        return self.fc6(h5)

    def forward(self, x):
        """IP-VAE forward pass"""
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    def load_weights(self):
        wt_path = pkg_resources.resource_filename('ipvae', '/weights.pt')
        model = nn.DataParallel(self)
        model.load_state_dict(torch.load(wt_path))
        return model
