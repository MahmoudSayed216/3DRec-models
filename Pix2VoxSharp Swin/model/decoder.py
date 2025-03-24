import torch.nn as nn
import torch


class Decoder(nn.Module):
    def __init__(self, configs):
        super(Decoder, self).__init__()

        bias = configs["model"]["use_bias"]

        self.upsample1 = nn.Sequential(nn.ConvTranspose3d(in_channels=6272, out_channels=768, kernel_size=4, stride=2, padding=1, bias=bias),
                                    #    nn.BatchNorm3d(num_features=1024),
                                       nn.LayerNorm(normalized_shape=(768, 4, 4, 4)),
                                       nn.GELU()) 
        
        self.upsample2 = nn.Sequential(nn.ConvTranspose3d(in_channels=768, out_channels=384, kernel_size=4, stride=2, padding=1, bias=bias),
                                    #    nn.BatchNorm3d(num_features=512),
                                       nn.LayerNorm(normalized_shape=(384, 8, 8, 8)),
                                       nn.GELU())
        
        self.upsample3 = nn.Sequential(nn.ConvTranspose3d(in_channels=384, out_channels=192, kernel_size=4, stride=2, padding=1, bias=bias),
                                    #    nn.BatchNorm3d(num_features=256),
                                       nn.LayerNorm(normalized_shape=(192, 16, 16, 16)),
                                       nn.GELU()
                                       )
        
        self.upsample4 = nn.Sequential(nn.ConvTranspose3d(in_channels=192, out_channels=48, kernel_size=4, stride=2, padding=1, bias=bias),
                                    #    nn.BatchNorm3d(num_features=12),
                                       nn.LayerNorm(normalized_shape=(48, 32, 32, 32)),
                                       nn.GELU()
                                       )
        
        self.upsample5 = nn.Sequential(nn.ConvTranspose3d(in_channels=48, out_channels=1, kernel_size=1, bias=bias))


    def forward(self, features_maps):
        
        print(features_maps.shape)
        volume = self.upsample1(features_maps)
        volume = self.upsample2(volume)
        volume = self.upsample3(volume)
        volume = self.upsample4(volume)
        raw_feature = volume
        volume = self.upsample5(volume)
        raw_feature = torch.cat((raw_feature, volume), dim=1)


        return raw_feature, volume

