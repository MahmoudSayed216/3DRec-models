import torch
import torch.nn as nn
import timm
from torchvision.models import convnext_base

class Encoder(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.backbone = self._prep_swin(configs["model"]["pretrained"])

        
    def _prep_swin(self, pretrained):
        model_name = None if pretrained == "None" else pretrained
        swin_base = timm.create_model(model_name, pretrained=False, features_only = True)
        #Unfreezing all the layers
        for param in swin_base.parameters():
            param.requires_grad = False
        #Except for the last stage
        for param in swin_base.layers_3.parameters():
            param.requires_grad = True
        return swin_base


    def forward(self, image: torch.Tensor):
        # images = images.permute(1, 0, 2, 3, 4).contiguous()
        features = self.backbone(image)
        print(features[-1].shape)
        
        return features[-1].permute(0, 3, 1, 2)
