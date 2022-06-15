import torch
from torch import nn
from torch import Tensor
from torchvision import transforms
import pandas as pd
from PIL import Image
import os
import matplotlib.pyplot as plt
import numpy as np

from upright.util import *

class GeodesicLoss(nn.Module):
    def __init__(self, eps: float = 1e-7, reduction: str = "mean") -> None:
        super().__init__()
        self.eps = eps
        self.reduction = reduction

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        input = quaternion_to_matrix(input).double()
        target = quaternion_to_matrix(target)

        R_diffs = input @ target.permute(0, 2, 1)
        traces = R_diffs.diagonal(dim1=-2, dim2=-1).sum(-1)
        dists = torch.acos(torch.clamp((traces - 1) / 2, -1 + self.eps, 1 - self.eps))
        if self.reduction == "none":
            return dists
        elif self.reduction == "mean":
            return dists.mean()
        elif self.reduction == "sum":
            return dists.sum()

class QuaternionGeodesicLoss(nn.Module):
    #like geodesic loss but with a component that penalizes outputs being
    def __init__(self, eps: float = 1e-7, reduction: str = "mean") -> None:
        super().__init__()
        self.eps = eps
        self.reduction = reduction

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        input_quat = input
        input = quaternion_to_matrix(input).double()
        target = quaternion_to_matrix(target)

        R_diffs = input @ target.permute(0, 2, 1)
        traces = R_diffs.diagonal(dim1=-2, dim2=-1).sum(-1)
        dists = torch.acos(torch.clamp((traces - 1) / 2, -1 + self.eps, 1 - self.eps))
        sum_constr = torch.square(torch.sum(torch.square(input_quat), dim=-1) - 1)

        res = dists + sum_constr

        if self.reduction == "none":
            return res
        elif self.reduction == "mean":
            return res.mean()
        elif self.reduction == "sum":
            return res.sum()


class QuaternionLoss(nn.Module):
    def __init__(self, reduction: str = "mean") -> None:
        super().__init__()
        self.reduction = reduction

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        res =  torch.square(torch.sum(torch.square(input), dim=-1) - 1)

        if self.reduction == "none":
            return res
        elif self.reduction == "mean":
            return res.mean()
        elif self.reduction == "sum":
            return res.sum()

def angle_between(a,b):
    R = np.dot(a,b.T)
    theta = (np.trace(R) -1)/2
    return np.arccos(theta) * (180/np.pi)