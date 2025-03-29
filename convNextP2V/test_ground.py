# from model.convNextUpsamplingBlock import ConvNextUpBlock
# import torch

# in_channels = 4
# out_channels = 8

# upblock = ConvNextUpBlock(in_channels, out_channels, 'transpose', 4)

# input = torch.rand((10, in_channels, 2, 2, 2))

# output = upblock(input)

# print(output.shape)

from model.full_model import Pix2VoxSharp
import yaml
import torch

with open("/home/mahmoud-sayed/Desktop/Graduation Project/current/Pix2Vox Models/convNextP2V/config.yaml", "r") as f:
        configs = yaml.safe_load(f)


model = Pix2VoxSharp(configs=configs)

input = torch.rand(size=(4, 3, 224, 224))

raw_feature, volume = model(input)

print(volume.shape)