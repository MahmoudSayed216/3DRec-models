import torch
import torch.nn as nn
from torchvision.models import convnext_small, ConvNeXt_Small_Weights


class ClothesClassifier(nn.Module):
    def _init_(self, num_classes_per_category):
        super(ClothesClassifier, self)._init_()
        
        # base_model = models.resnet152(pretrained=True)
        base_model = convnext_small(weights=ConvNeXt_Small_Weights)
        # self.feature_extractor = nn.Sequential(*list(base_model.children())[:-1])
        self.feature_extractor = nn.Sequential(*list(self.convnext.features.children())[:6])

        self.fc_layer = nn.ModuleList([
            nn.Linear(2048, num_classes) for num_classes in num_classes_per_category
        ])
    
    def forward(self, x):
        x = self.feature_extractor(x)
        x = x.view(x.size(0), -1)
        outputs = [fc(x) for fc in self.fc_layer]
        return outputs
num_classes_per_category = [4]
model = ClothesClassifier(num_classes_per_category).to("cuda")
print(df["season"].value_counts())
print(len(df["season"].unique()))