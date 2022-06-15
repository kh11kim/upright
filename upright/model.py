import torch
from torch import nn
from torch import Tensor
from torchvision import transforms
import pandas as pd
from PIL import Image
import os
import matplotlib.pyplot as plt
import numpy as np


#identity to replace
class Identity(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        return x

#describe our head
class UprightClassifier(nn.Module):
    def __init__(self, n):
        super(UprightClassifier, self).__init__()
        self.n = n
        self.flatten = nn.Flatten()
        self.classification_stack = nn.Sequential(
            nn.Linear(n*2048, 1000), #might use conv1d here to be able to use batchnorm1d
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(1000, 4),
            nn.Tanh(),
        )

    def forward(self, x):
        x = self.flatten(x)
        quaternion = self.classification_stack(x)
        return quaternion / torch.norm(quaternion, dim=1, keepdim=True)

#describe our model
class Upright(nn.Module):
    def __init__(self):
        super(Upright, self).__init__()
        self.n = 3
        self.backbone = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True)
        self.backbone.fc = Identity()
        self.head = UprightClassifier(self.n)
      
    def forward(self, input):
        features = list()
        for i in range(self.n):
            features.append(self.backbone(input[:,i]))
        
        return self.head(torch.cat(features, 1))

#describe our 6D head
class UprightClassifier6D(nn.Module):
    def __init__(self, n):
        super(UprightClassifier6D, self).__init__()
        self.n = n
        self.flatten = nn.Flatten()
        self.classification_stack = nn.Sequential(
            nn.Linear(n*2048, 1000), #might use conv1d here to be able to use batchnorm1d
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(1000, 6),
            nn.Tanh(),
        )

    def forward(self, x):
        x = self.flatten(x)
        d6 = self.classification_stack(x)
        #gram-schmidt to make it into a rotation matrix
        a1, a2 = d6[..., :3], d6[..., 3:]
        b1 = torch.nn.functional.normalize(a1, dim=-1)
        b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
        b2 = torch.nn.functional.normalize(b2, dim=-1)
        b3 = torch.cross(b1, b2, dim=-1)
        return torch.stack((b1, b2, b3), dim=-2)

#describe our 6D model
class Upright6D(nn.Module):
    def __init__(self):
        super(Upright6D, self).__init__()
        self.n = 3
        self.backbone = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True)
        self.backbone.fc = Identity()
        self.head = UprightClassifier6D(self.n)

    def forward(self, input):
        features = list()
        for i in range(self.n):
            features.append(self.backbone(input[:,i]))

        return self.head(torch.cat(features, 1))