import torch.nn as nn
import torch


class _3d_depthwise_separable_convolution(nn.Module):
    def __init__(self, in_channels, out_channels, stride, side_length):
        super().__init__()
        self.dwsc_layer = nn.Sequential(
            nn.Conv3d(in_channels, in_channels, 3, stride, padding=1, groups=in_channels),
            nn.LayerNorm(normalized_shape=(in_channels, side_length, side_length, side_length)),
            nn.GELU(),
            nn.Conv3d(in_channels, out_channels, 1),
            nn.LayerNorm(normalized_shape=(out_channels, side_length, side_length, side_length)),
            nn.GELU()
        )
        

    def forward(self, x):
        x = self.dwsc_layer(x)

        return x


class Decoder(nn.Module):
    def __init__(self, configs):
        super(Decoder, self).__init__()

        bias = configs["model"]["use_bias"]

        self.upsample1 = nn.Sequential(
                                        nn.ConvTranspose3d(in_channels=4704, out_channels=768, kernel_size=4, stride=2, padding=1, bias=bias),
                                        nn.LayerNorm(normalized_shape=(768, 4, 4, 4)),
                                        nn.GELU()) 
        
        self.upsample2 = nn.Sequential(
                                        _3d_depthwise_separable_convolution(1944, 500, 1, 4),
                                        nn.ConvTranspose3d(in_channels=500, out_channels=384, kernel_size=4, stride=2, padding=1, bias=bias),
                                        nn.LayerNorm(normalized_shape=(384, 8, 8, 8)),
                                        nn.GELU())
        
        self.upsample3 = nn.Sequential(
                                        _3d_depthwise_separable_convolution(678, 200, 1, 8),
                                        nn.ConvTranspose3d(in_channels=200, out_channels=192, kernel_size=4, stride=2, padding=1, bias=bias),
                                        nn.LayerNorm(normalized_shape=(192, 16, 16, 16)),
                                        nn.GELU()
                                       )
        
        self.upsample4 = nn.Sequential(
                                        _3d_depthwise_separable_convolution(192, 100, 1, 16),
                                        nn.ConvTranspose3d(in_channels=100, out_channels=48, kernel_size=4, stride=2, padding=1, bias=bias),
                                        nn.LayerNorm(normalized_shape=(48, 32, 32, 32)),
                                        nn.GELU()
                                       )
        
        self.upsample5 = nn.Sequential(nn.ConvTranspose3d(in_channels=48, out_channels=1, kernel_size=1, bias=bias))


    def forward(self, us1, us2, us3):
        volume = self.upsample1(us1)
        catted = torch.cat((volume, us2), dim=1)
        volume = self.upsample2(catted)
        catted2 = torch.cat((volume, us3), dim=1)
        volume = self.upsample3(catted2)
        volume = self.upsample4(volume)
        raw_feature = volume
        volume = self.upsample5(volume)
        raw_feature = torch.cat((raw_feature, volume), dim=1)


        return raw_feature, volume

