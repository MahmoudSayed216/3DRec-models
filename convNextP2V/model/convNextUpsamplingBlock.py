import torch
import torch.nn as nn


class ConvNextUpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, upsample_mode, target_side_length):
        super(ConvNextUpBlock, self).__init__()
        self.upsample_mode = upsample_mode

        if self.upsample_mode == 'transpose':
            self.upsample = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)
        
        elif self.upsample_mode == 'pixelshuffle':
            self.pixelshuffle = nn.PixelShuffle(2)
            self.upsample = nn.Conv3d(in_channels, out_channels*4, kernel_size=1)

        self.dw_conv = nn.Conv3d(out_channels, out_channels, kernel_size=7, padding=3, groups=out_channels)

        self.norm = nn.LayerNorm(normalized_shape=(out_channels, target_side_length, target_side_length, target_side_length))

        self.pw_conv1 = nn.Conv3d(out_channels, 4*out_channels, kernel_size=1)
        self.pw_conv2 = nn.Conv3d(4*out_channels, out_channels, kernel_size=1)

        self.gelu = nn.GELU()


    def forward(self, x):
        x = self.upsample(x)
        if self.upsample_mode == 'pixelshuffle':
            x = self.pixelshuffle(x)
        
        residual = x

        x = self.norm(x)
        x = self.dw_conv(x)
        x = self.pw_conv1(x)
        x = self.gelu(x)
        x = self.pw_conv2(x)

        return x + residual