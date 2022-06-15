"""
inference.py
Sample code for loading and running one of our trained models
"""


import torch
from torchvision import transforms
from PIL import Image

from upright.model import Upright, Upright6D
from upright.util import *
from upright.loss import angle_between

#pre-process function for the images by cropping and normalizing
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# load all images in `paths`, preprocess and stack them.
# if the network was trained for n (e.g 3) images, `paths` must contain n paths
def get_images_from_paths(paths):
    return torch.stack([preprocess(Image.open(path)) for path in paths])

# take the image objects in PIL format, preprocess and stack them.
# if the network was trained for n (e.g 3) images, `images` must contain n objects
def get_images_from_PIL(images):
    return torch.stack([preprocess(pil_image) for pil_image in images])

#compute the angle between label and predicted quaternion through rotation matrix
def compute_loss_quat(label, output):
    return angle_between(quaternion_to_matrix(label).detach().numpy(), quaternion_to_matrix(output).double().detach().numpy())

#compute the angle between label and predicted rotation matrix
def compute_loss_rot_mat(label, output):
    return angle_between(label.detach().numpy(), output.detach().numpy())

#the model is set up for (mini-batches) which adds one dimension to the input and output
#this wrapper handles this by wrapping input and unwrapping it again
def predict(model, input):
    return model(torch.unsqueeze(input,dim=0))[0]

"""
Set up the model
"""

mode = "QUAT"
model = None
path_model = "data/eval-models/quat/quat-epoch100.pt"

if mode == "QUAT":
    # set up the model to evaluate
    model = Upright()
    model.load_state_dict(torch.load(path_model,map_location=torch.device('cpu')))
    model.eval()

elif mode == "6D":
    # set up the model to evaluate
    model = Upright6D()
    model.load_state_dict(torch.load(path_model,map_location=torch.device('cpu')))
    model.eval()

"""
Load images and run one prediction
"""
images = ["data/eval-models/data-eval/0ba2f102dac84625b156c515c670b3e8_002_master_chef_can_0.jpg",
          "data/eval-models/data-eval/0ba2f102dac84625b156c515c670b3e8_002_master_chef_can_1.jpg",
          "data/eval-models/data-eval/0ba2f102dac84625b156c515c670b3e8_002_master_chef_can_2.jpg"]
images = get_images_from_paths(images) #alternatively can use get_images_from_PIL if they are already in memory
prediction = predict(model,images)
print("Prediction for the images:",prediction)