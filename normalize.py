import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import os
import pandas as pd


def compute_mean_std(loader):
    mean = 0.0
    std = 0.0
    total_images_count = 0
    for images, _ in loader:
        image_count_in_batch = images.size(0)
        images = images.view(image_count_in_batch, images.size(1), -1)
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)
        total_images_count += image_count_in_batch
    print(total_images_count)

    mean /= total_images_count
    std /= total_images_count
    return mean, std

class MyDataset(Dataset):
    def __init__(self, dataset_path, meta_data_path, transform):
        super().__init__()
        self.path_data = dataset_path + pd.read_csv(meta_data_path)["Path"]
        self.class_data = pd.read_csv(meta_data_path)["ClassId"]
        self.transforms = transform

    # ここで取り出すデータを指定している
    def __getitem__(self,index):
        path = self.path_data[index]
        img = Image.open(path) #ここで画像の読み込み
        # データの変形 (transforms)
        img = self.transforms(img)
        label = int(self.class_data[index])
        return img, label

    # この method がないと DataLoader を呼び出す際にエラーを吐かれる
    def __len__(self):
        return len(self.path_data)

# データの前処理
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor()
])

# データセットのインスタンス化
train_dataset = MyDataset("../data/dataset/GTSRB/", "../data/dataset/GTSRB/Train.csv", transform=transform)

# データローダーの作成
full_loader = DataLoader(train_dataset, batch_size=64, shuffle=False)

# 平均と標準偏差の計算
mean, std = compute_mean_std(full_loader)
print(f"Calculated mean: {mean}, std: {std}")

# Train
# mean: tensor([0.3403, 0.3121, 0.3214]), std: tensor([0.1595, 0.1590, 0.1683])

# Test
# mean: tensor([0.3374, 0.3096, 0.3208]), std: tensor([0.1615, 0.1604, 0.1709])