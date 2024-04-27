# https://noconocolib.hatenablog.com/entry/2019/01/12/231723を参照
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.models import ResNet50_Weights
import torch.nn as nn
import torchvision.models as models
from torch.utils.data.dataloader import DataLoader
from torchvision.utils import make_grid, save_image
import torch.nn.functional as F
from train_resnet import GtsrbCNN,Traffic_Dataset
import os
import pickle
import matplotlib.pyplot as plt
import numpy as np
import cv2



class GradCam:
    def __init__(self, model):
        self.model = model.eval()
        self.feature = None
        self.gradient = None

    def save_gradient(self, grad):
        self.gradient = grad

    def __call__(self, x):
        image_size = (x.size(-1), x.size(-2))
        feature_maps = []
        
        for i in range(x.size(0)):
            img = x[i].data.cpu().numpy()
            img = img - np.min(img)
            if np.max(img) != 0:
                img = img / np.max(img)

            feature = x[i].unsqueeze(0)
            
            for name, module in self.model.named_children():
                if name == 'classifier':
                    feature = feature.view(feature.size(0), -1)
                feature = module(feature)
                if name == 'features':
                    feature.register_hook(self.save_gradient)
                    self.feature = feature
                    
            classes = F.sigmoid(feature)
            one_hot, _ = classes.max(dim=-1)
            self.model.zero_grad()
            one_hot.backward()

            weight = self.gradient.mean(dim=-1, keepdim=True).mean(dim=-2, keepdim=True)
            
            mask = F.relu((weight * self.feature).sum(dim=1)).squeeze(0)
            mask = cv2.resize(mask.data.cpu().numpy(), image_size)
            mask = mask - np.min(mask)
            
            if np.max(mask) != 0:
                mask = mask / np.max(mask)
                
            feature_map = np.float32(cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET))
            cam = feature_map + np.float32((np.uint8(img.transpose((1, 2, 0)) * 255)))
            cam = cam - np.min(cam)
            
            if np.max(cam) != 0:
                cam = cam / np.max(cam)
                
            feature_maps.append(transforms.ToTensor()(cv2.cvtColor(np.uint8(255 * cam), cv2.COLOR_BGR2RGB)))
            
        feature_maps = torch.stack(feature_maps)
        
        return feature_maps
    
with open('../data/dataset/GTSRB/train.pkl', 'rb') as f:
            train = pickle.load(f)
            train_x, train_y = train['data'], train['labels']
with open('../data/dataset/GTSRB/test.pkl', 'rb') as f:
    test = pickle.load(f)
    test_x, test_y = test['data'], test['labels']

batch_size = 1
n_class = len(np.unique(train_y))
model = GtsrbCNN(n_class)
for name,module in model.res_block.named_children():
     print(name)
# fcをターゲットレイヤーにすること、grad_camクラスについてもっと詳しく読むこと、必要なもののインポートが足りていないのでインポートすること。

train_dataset = Traffic_Dataset(train_x,train_y)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

for image,label in train_loader:
       break

image = image.to(torch.float64)
label = label.to(torch.float64)
print((image.requires_grad))
print((label.requires_grad))
grad_cam = GradCam(model)

feature_image = grad_cam(image).squeeze(dim=0)
feature_image = transforms.ToPILImage()(feature_image)

pred_idx = model(image).max(1)[1]
print("pred: ", label[int(pred_idx)])
plt.title("Grad-CAM feature image")
plt.imshow(feature_image.resize((image.shape[1],image.shape[2],image.shape[3])))