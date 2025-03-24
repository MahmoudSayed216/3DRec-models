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
        upprojection1 = feature_maps[-1].contiguous().view((batch_size, 4704, 2, 2, 2))
        upprojection2 = feature_maps[-2].contiguous().view((batch_size, 1176, 4, 4, 4))
        upprojection3 = feature_maps[-3].contiguous().view((batch_size, 294, 8, 8, 8))
        # upprojection4 = feature_maps[-4].contiguous().view((batch_size, 147, 16, 16, 16))
        volume = self.decoder(upprojection1, upprojection2, upprojection3)
        return volume
        