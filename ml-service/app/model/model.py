import torch.nn as nn
from torchvision import models

class SignLanguageModel(nn.Module):
    def __init__(self, num_classes=26, pretrained=False):
        super().__init__()

        self.model = models.resnet18(
            weights=None if not pretrained else models.ResNet18_Weights.DEFAULT
        )

        self.model.fc = nn.Sequential(
            nn.Dropout(0.5),                 # fc.0
            nn.Linear(512, 512),             # fc.1
            nn.ReLU(),                       # fc.2
            nn.Dropout(0.5),                 # fc.3
            nn.Linear(512, num_classes)      # fc.4
        )

    def forward(self, x):
        return self.model(x)
