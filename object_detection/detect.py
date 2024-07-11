import sys
import torch
import torchvision
import torchvision.models.detection as detection
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.ops import box_iou
from dataset import *
from tqdm import tqdm
from func import *
from torchvision.models.detection import ssdlite320_mobilenet_v3_large

meta_train_path = "../../data/dataset/GTSRB/Train.csv"
dataset_path = '../../data/dataset/GTSRB/'

batch_size = 2
transform = transforms.Compose([
    transforms.Resize((32,32)),  # Resize image to 32x32
    transforms.ToTensor(),          # Convert image to tensor
    transforms.Normalize(mean=[0.3403, 0.3121, 0.3214], std=[0.1595, 0.1590, 0.1683])  # Normalize with training data mean and std
])
num_class = 5  # Including background
num_epochs = 50

# Device setting (use GPU if available)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# Load the pre-trained SSD model with a MobileNetV3 backbone
model = ssdlite320_mobilenet_v3_large(pretrained=True)
model = model.to(device)

# Modify the model's number of classes
model.head.classification_head.num_classes = num_class

# Parameters for optimizer
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

# Create the custom dataset and data loader
train_dataset = CustomDataset(dataset_path, meta_train_path, transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

# Training loop
for epoch in range(num_epochs):
    print(f'Epoch {epoch+1}/{num_epochs}')
    train_model(model, train_loader, optimizer, device)

    # モデルのチェックポイントを保存
    torch.save(model.state_dict(), f'models/model_epoch_{epoch+1}.pth')