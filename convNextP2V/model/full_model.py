import torch
import torch.nn as nn
from .encoder import Encoder
from .decoder import Decoder


class Pix2VoxSharp(nn.Module):
    def __init__(self, configs):
        super(Pix2VoxSharp, self).__init__()
        self.encoder = Encoder(configs)
        self.decoder = Decoder(configs)
        self.batch_size = configs["train"]["batch_size"]
    def forward(self, x:torch.Tensor):
        # batch_size = x.size(0)
        feature_maps = self.encoder(x)
        upprojection = feature_maps.contiguous().view((self.batch_size, 6272, 2, 2, 2))
        volume = self.decoder(upprojection)
        return volume
        