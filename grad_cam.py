import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.models import ResNet50_Weights
import torch.nn as nn
import torchvision.models as models
from torch.utils.data.dataloader import DataLoader
from torchvision.utils import make_grid, save_image
from train_resnet import GtsrbCNN,Traffic_Dataset
import os
import pickle
import matplotlib.pyplot as plt
import numpy as np

# Grad-CAM
#from pytorch_grad_cam.utils import visualize_cam
import sys
sys.path.append("../pytorch_grad_cam")
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

with open('../data/dataset/GTSRB/train.pkl', 'rb') as f:
            train = pickle.load(f)
            train_x, train_y = train['data'], train['labels']
with open('../data/dataset/GTSRB/test.pkl', 'rb') as f:
    test = pickle.load(f)
    test_x, test_y = test['data'], test['labels']

batch_size = 1
n_class = len(np.unique(train_y))
#model = GtsrbCNN(n_class)
model = models.resnet50()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
loss_func = nn.CrossEntropyLoss()

train_dataset = Traffic_Dataset(train_x,train_y)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
for image,label in train_loader:
       break
#image = image[0].unsqueeze(0)
print(image.shape)
label = [ClassifierOutputTarget(label)]

n_class = len(np.unique(train_y))
#model = GtsrbCNN(n_class)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
loss_func = nn.CrossEntropyLoss()

"""model_path = "../data/model/resnet_gtsrb/model.tar"
if os.path.isfile(model_path):
    checkpoint = torch.load(model_path)
    model.eval()
    model.load_state_dict(checkpoint["model_state_dict"])"""

# Grad-CAM
#target_layer = [model.res_block.fc]
target_layer = [model.fc]
cam = GradCAM(model, target_layer)
grayscale_cam = cam(input_tensor=image,targets=label)
grayscale_cam = grayscale_cam[0, :]
print(grayscale_cam.shape)
visualization = show_cam_on_image(image, grayscale_cam, use_rgb=True)
plt.imshow(visualization)
plt.show()
