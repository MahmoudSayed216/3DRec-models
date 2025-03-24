import torch.nn as nn
import torch


class Decoder(nn.Module):
    def __init__(self, configs):
        super(Decoder, self).__init__()

        bias = configs["model"]["use_bias"]

        self.upsample1 = nn.Sequential(nn.ConvTranspose3d(in_channels=4704, out_channels=512, kernel_size=4, stride=2, padding=1, bias=bias),
                                    #    nn.BatchNorm3d(num_features=1024),
                                       nn.LayerNorm(normalized_shape=(512, 4, 4, 4)),
                                       nn.GELU()) 
        
        self.upsample2 = nn.Sequential(nn.ConvTranspose3d(in_channels=1688, out_channels=256, kernel_size=4, stride=2, padding=1, bias=bias),
                                    #    nn.BatchNorm3d(num_features=512),
                                       nn.LayerNorm(normalized_shape=(256, 8, 8, 8)),
                                       nn.GELU())
        
        self.upsample3 = nn.Sequential(nn.ConvTranspose3d(in_channels=550, out_channels=128, kernel_size=4, stride=2, padding=1, bias=bias),
                                    #    nn.BatchNorm3d(num_features=256),
                                       nn.LayerNorm(normalized_shape=(128, 16, 16, 16)),
                                       nn.GELU()
                                       )
        
        self.upsample4 = nn.Sequential(nn.ConvTranspose3d(in_channels=128, out_channels=32, kernel_size=4, stride=2, padding=1, bias=bias),
                                    #    nn.BatchNorm3d(num_features=12),
                                       nn.LayerNorm(normalized_shape=(32, 32, 32, 32)),
                                       nn.GELU()
                                       )
        
        self.upsample5 = nn.Sequential(nn.ConvTranspose3d(in_channels=32, out_channels=1, kernel_size=1, bias=bias))


    def forward(self, up1, up2, up3):
        volume = self.upsample1(up1)
        volume = torch.cat((volume, up2), dim=1)
        volume = self.upsample2(volume)
        volume = torch.cat((volume, up3), dim=1)
        volume = self.upsample3(volume)
        volume = self.upsample4(volume)
        raw_feature = volume
        volume = self.upsample5(volume)
        raw_feature = torch.cat((raw_feature, volume), dim=1)


        return raw_feature, volume

