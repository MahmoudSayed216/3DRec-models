import torch.nn as nn
import torch
from .convNextUpsamplingBlock import ConvNextUpBlock

class Decoder(nn.Module):
    def __init__(self, configs):
        super(Decoder, self).__init__()

        bias = configs["model"]["use_bias"]
        upsample_mode = configs['model']['upsample_mode']

        self.upsample1 = ConvNextUpBlock(in_channels=6272, out_channels=512, upsample_mode=upsample_mode, target_side_length=4)
        
        
        self.upsample2 = ConvNextUpBlock(in_channels=512, out_channels=256, upsample_mode=upsample_mode, target_side_length=8)

        self.upsample3 = ConvNextUpBlock(in_channels=256, out_channels=128, upsample_mode=upsample_mode, target_side_length=16)
        
        self.upsample4 = ConvNextUpBlock(in_channels=128, out_channels=32, upsample_mode=upsample_mode, target_side_length=32)
        
        
        self.upsample5 = nn.Sequential(nn.Conv3d(in_channels=32, out_channels=1, kernel_size=1))


    def forward(self, features_maps):
        volume = self.upsample1(features_maps)
        volume = self.upsample2(volume)
        volume = self.upsample3(volume)
        volume = self.upsample4(volume)
        raw_feature = volume
        volume = self.upsample5(volume)
        raw_feature = torch.cat((raw_feature, volume), dim=1)


        return raw_feature, volume

