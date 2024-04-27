# Basic Modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# PyTorch Modules
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms, datasets
import torchvision.transforms as transforms
from torch.utils.data.dataset import Subset
import torchvision.models as models
import torch.optim as optim
from torchvision.utils import make_grid, save_image
from train_resnet import GtsrbCNN,Traffic_Dataset

# Grad-CAM
from gradcam.utils import visualize_cam
from gradcam import GradCAM, GradCAMpp
from PIL import Image
import pickle


device = torch.device("cuda:0" if torch.cuda.is_available()  else "cpu")
model = models.resnet50(pretrained=True)
model.fc = nn.Linear(2048,5)
model = torch.nn.DataParallel(model).to(device)
print(model.module.fc)
#exit()

### GTSRB用のコード
with open('../data/dataset/GTSRB/train.pkl', 'rb') as f:
            train = pickle.load(f)
            train_x, train_y = train['data'], train['labels']
with open('../data/dataset/GTSRB/test.pkl', 'rb') as f:
    test = pickle.load(f)
    test_x, test_y = test['data'], test['labels']

#batch_size = 1
n_class = len(np.unique(train_y))
#model = GtsrbCNN(n_class)
### 

model.eval()
#model.load_state_dict(torch.load('trained_model.pt'))

# Grad-CAM
target_layer = model.module#.features
#target_layer = model.res_block.fc
gradcam = GradCAM(model, target_layer)
gradcam_pp = GradCAMpp(model, target_layer)

images = []
# あるラベルの検証用データセットを呼び出してる想定
path = "../data/template/GTSRB/5_pic/0_0.jpg"
img = Image.open(path)
torch_img = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])(img).to(device)

normed_torch_img = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(torch_img)[None]
mask, _ = gradcam(normed_torch_img)
heatmap, result = visualize_cam(mask, torch_img)

mask_pp, _ = gradcam_pp(normed_torch_img)
heatmap_pp, result_pp = visualize_cam(mask_pp, torch_img)

images.extend([torch_img.cpu(), heatmap, heatmap_pp, result, result_pp])
grid_image = make_grid(images, nrow=5)
grid_image = torchvision.transforms.functional.to_pil_image(grid_image)
# 結果の表示
grid_image.save("abcd.jpg")
print("Done!")
#transforms.ToPILImage()(grid_image)