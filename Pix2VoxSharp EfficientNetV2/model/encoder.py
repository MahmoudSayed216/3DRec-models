import torch
import torch.nn as nn
import timm
from torchvision.models import convnext_base

class Encoder(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.backbone = self._prep_convnext(configs["model"]["pretrained"])

        
    def _prep_convnext(self, pretrained):
        model_name = None if pretrained == "None" else pretrained
        efficient_net = timm.create_model(model_name, pretrained=True, features_only = True)
        #Unfreezing all the layers
        for param in efficient_net.parameters():
            param.requires_grad = False
        #Except for the last stage
        for param in efficient_net.blocks[4].parameters():
            param.requires_grad = True
        return efficient_net


    def forward(self, image: torch.Tensor):
        # images = images.permute(1, 0, 2, 3, 4).contiguous()
        features = self.backbone(image)
        return features[-1]
