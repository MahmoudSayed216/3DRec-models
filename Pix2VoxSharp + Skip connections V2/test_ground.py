from model import full_model
import torch

# enc = encoder.Encoder({"model":{"pretrained": False}})
model = full_model.Pix2VoxSharp({"model":{"pretrained": False, "use_bias":True}})

output = model(torch.rand((1, 3, 224, 224)))
print(output[0].shape)
print(output[1].shape)
