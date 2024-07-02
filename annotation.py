# 物体検出モデル学習に使用する関数・クラスを定義

import torch
from torch.utils.data import Dataset
from torchvision.models import ResNet50_Weights
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from tqdm import tqdm
import time
import pandas as pd
import numpy as np
from PIL import Image
import cv2
from torchvision import transforms
from torch.utils.data import DataLoader
import json

class ObjectDetectionModel(nn.Module):
    def __init__(self, num_classes):
        super(ObjectDetectionModel, self).__init__()
        self.backbone = models.resnet50(pretrained=True)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, 256)
        self.classifier = nn.Linear(256, num_classes)
        
        # 円のパラメータ (cx, cy, r)
        self.circle_regressor = nn.Linear(256, 3)
        
        # 三角形のパラメータ ((x1, y1), (x2, y2), (x3, y3))
        #self.triangle_regressor = nn.Linear(256, 6)

    def forward(self, x):
        features = self.backbone(x)
        class_logits = self.classifier(features)
        circle_params = self.circle_regressor(features)
        #triangle_params = self.triangle_regressor(features)
        return class_logits, circle_params#, triangle_params

def classification_loss(predicted_classes, true_classes):
    return F.cross_entropy(predicted_classes, true_classes)

def circle_loss(predicted_circle, true_circle):
    # 予測と真の円の中心座標と半径に対するL2損失
    return F.mse_loss(predicted_circle, true_circle)



# 画像、アノテーションデータなどを返すクラス
class Annotation(Dataset):
    def __init__(self, dataset_path, meta_data_path, json_dir, transform):
        super().__init__()
        df = pd.read_csv(meta_data_path)
        df = df[df["shape"]=="circle"].reset_index(drop=True) # 円の画像だけ
        self.json_dir = json_dir
        self.path_data = dataset_path + df["Path"]
        self.class_data = df["ClassId"]
        self.x1, self.y1 = df.new_x1, df.new_y1
        self.x2, self.y2 = df.new_x2, df.new_y2
        self.shape_type = df["shape"]
        self.transforms = transform

    # ここで取り出すデータを指定している
    def __getitem__(self,index):
        path = self.path_data[index]
        json_path = path.split("/")[-1].split(".")[0] + ".json"
        with open(json_path) as f:
            di = json.load(f)
        img = Image.open(path) #ここで画像の読み込み
        # データの変形 (transforms)
        img = self.transforms(img)
        label = self.class_data[index]
        polygon_info = get_polygon_info(self.x1[index],self.y1[index],self.x2[index],self.y2[index],self.shape_type[index])
        return img, label, polygon_info

    # この method がないと DataLoader を呼び出す際にエラーを吐かれる
    def __len__(self):
        return len(self.path_data)
    
def get_polygon_info(x1, y1, x2, y2, shape_type):
    # 長方形の幅と高さを計算
    width = x2 - x1
    height = y2 - y1
    if (shape_type == "circle"):#|(shape_type == "octagon"):
        # 円の半径は長方形の幅と高さの小さい方の半分
        radius = min(width, height) // 2
        # 円の中心は長方形の中心
        center_x = x1 + width // 2
        center_y = y1 + height // 2
        return shape_type, radius, center_x, center_y
    """elif shape_type == "triangle":
        top_x = x1 + width//2
        top_y = y1
        left_x, left_y = x1, y2
        right_x, right_y = x2, y2
        return top_x, top_y, left_x, left_y, right_x, right_y
    elif shape_type == "r_triangle":
        bottom_x = x1 + width//2
        bottom_y = y2
        left_x, left_y = x1, y1
        right_x, right_y = x2, y1
        return bottom_x, bottom_y, left_x, left_y, right_x, right_y
    elif shape_type == "rectangle":"""

if __name__=="__main__":
    meta_train_path = "../data/dataset/GTSRB/Train.csv"
    dataset_path = '../data/dataset/GTSRB/'
    batch_size = 1
    annotation_dir = "../data/dataset/GTSRB/annotation"
    # 訓練ループのサンプル
    num_classes = 1 #5  # 円、三角形、逆三角形、八角形、菱形の5クラス
    model = ObjectDetectionModel(num_classes)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)


    # データの前処理
    transform = transforms.Compose([
        transforms.Resize((32,32)),  # 画像を32x32にリサイズ
        transforms.ToTensor(),          # 画像をテンソルに変換
        transforms.Normalize(mean=[0.3403, 0.3121, 0.3214], std=[0.1595, 0.1590, 0.1683])  # 正規化(訓練データのmeanとstd)
    ])
    train_dataset = Annotation(dataset_path, meta_train_path,transform)
    data_size = len(train_dataset)
    train_loader = DataLoader(train_dataset,batch_size=batch_size,shuffle=False)

    count = 0
    for image, label, polygon_info in train_loader:
        image = image[0]
        image = image.permute(1, 2, 0).cpu().numpy()
        image = (image * 255).astype(np.uint8)  # 正規化を逆にする
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if polygon_info[0][0] == "circle":
            shape_type, radius, center_x, center_y = polygon_info
            shape_type, radius, center_x, center_y = shape_type, int(radius), int(center_x), int(center_y)
        image_with_circle = cv2.circle(image, (center_x, center_y), radius, (0, 255, 0), 1)
        
        # 画像を保存
        cv2.imwrite(f"{annotated_path}/{count}.png",image_with_circle)
        count += 1