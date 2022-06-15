import torch
from torch import nn
from torch import Tensor
from torchvision import transforms
import pandas as pd
from PIL import Image
import os
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

def imshow(inp, title=None):
    #imshow for a tensor
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.figure()
    plt.imshow(inp)

# helper functions to compute the loss over the test set for validation purposes
def quaternion_to_matrix(quaternions: torch.Tensor) -> torch.Tensor:
    r, i, j, k = torch.unbind(quaternions, -1)
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))

def quaternion_to_axis_angle(quaternions: torch.Tensor) -> torch.Tensor:
    norms = torch.norm(quaternions[..., 1:], p=2, dim=-1, keepdim=True)
    half_angles = torch.atan2(norms, quaternions[..., :1])
    angles = 2 * half_angles
    eps = 1e-6
    small_angles = angles.abs() < eps
    sin_half_angles_over_angles = torch.empty_like(angles)
    sin_half_angles_over_angles[~small_angles] = (
        torch.sin(half_angles[~small_angles]) / angles[~small_angles]
    )
    # for x small, sin(x/2) is about x/2 - (x/2)^3/6
    # so sin(x/2)/x is about 1/2 - (x*x)/48
    sin_half_angles_over_angles[small_angles] = (
        0.5 - (angles[small_angles] * angles[small_angles]) / 48
    )
    return quaternions[..., 1:] / sin_half_angles_over_angles

def compute_test_loss(model, device, data_loader_test, criterion):
    loss_sum = 0.0
    for i, data in enumerate(data_loader_test, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs) #returns results
        loss = criterion(outputs, labels)
        loss_sum += loss.item()

    loss = loss_sum / len(data_loader_test.dataset)

    return loss

def draw_curve(fig, ax, data, labels):
    colors = ['orangered', 'gold']#'coral', 'gold', 'khaki', 'moccasin', 'goldenrod']

    i = 0
    for line, label in zip(data, labels):
        ax.plot([i for i in range(len(line))], line, color=colors[i], label='loss on '+label)
        i+=1

    if len(data[0]) == 1:
        ax.legend()
    fig.savefig('loss.pdf', dpi=300, format="pdf")


def save_snapshot(model, data_loader_test, data_loader_train, fig, name_prefix=""):
    path = 'models/'+name_prefix
    print("Saving snapshot", path)
    torch.save(model.state_dict(), path+"-model.pt")
    torch.save(data_loader_test, path+"-dl-test.pt")
    torch.save(data_loader_train, path+"-dl-train.pt")
    fig.savefig(path+"-loss.pdf", dpi=300, format="pdf")