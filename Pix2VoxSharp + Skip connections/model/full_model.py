import torch
import torch.nn as nn
from .encoder import Encoder
from .decoder import Decoder


class Pix2VoxSharp(nn.Module):
    def __init__(self, configs):
        super(Pix2VoxSharp, self).__init__()
        self.encoder = Encoder(configs)
        self.decoder = Decoder(configs)

    def forward(self, x:torch.Tensor):
        batch_size = x.size(0)
        feature_maps = self.encoder(x)
        upprojection1 = feature_maps.contiguous().view((batch_size, 9408, 2, 2, 2))
        volume = self.decoder(upprojection1)
        return volume
        