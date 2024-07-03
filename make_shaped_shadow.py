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
import random
import re
import os

N = 0.43
ichou_shadow_dir = "../data/tree/"

def draw_shadow(image, inside_points, pattern):
    mask_img = cv2.imread(ichou_shadow_dir + f"ichou_shadow{str(pattern)}.png")
    if mask_img is None:
        raise FileNotFoundError(f"Image not found: {ichou_shadow_dir}ichou_shadow{str(pattern)}.png")

    # 画像をNumPy配列に変換し、チャネル順序を変換
    image = image.permute(1, 2, 0).numpy()  # (C, H, W) -> (H, W, C)
    image = (image * 255).astype(np.uint8)  # 0-1から0-255の範囲に変換
    
    img = cv2.cvtColor(image, cv2.COLOR_RGB2Lab)

    # 白いピクセルのマスクを作成
    white_mask = np.all(mask_img == [255, 255, 255], axis=-1)
    
    # 
    inside_circle_white = np.zeros((32,32),dtype=bool)
    for x,y in inside_points:
        # inside_pointsに対応するwhite_maskの値を代入
        inside_circle_white[x,y] = white_mask[x,y]
    # Lチャンネルに対して白いピクセルの位置を0.4倍する
    img[inside_circle_white, 0] = img[inside_circle_white, 0] * N

    # 値の範囲を0-255にクリップ
    img = np.clip(img, 0, 255).astype(np.uint8)

    # LabからRGB空間に変換
    img = cv2.cvtColor(img, cv2.COLOR_Lab2RGB)
    return img

def make_ichou_shadow(image, inside_points):
    tree_image_dir = "../data/tree/"
    dir_list = os.listdir(tree_image_dir)
    pattern = r'ichou(\d+)'
    shadow_pattern = []
    for f in dir_list:
        match = re.search(pattern, f)
        if match:
            shadow_pattern.append(int(match.group(1)))
    pattern = random.choice(shadow_pattern)
    draw_image = draw_shadow(image, inside_points, pattern)
    return draw_image

# 画像、アノテーションデータなどを返すクラス
class Annotation(Dataset):
    def __init__(self, dataset_path, meta_data_path, json_dir, transform):
        super().__init__()
        df = pd.read_csv(meta_data_path)
        self.df = df[df["shape"]=="circle"].reset_index(drop=True) # 円の画像だけ
        self.json_dir = json_dir
        self.dataset_path = dataset_path
        self.transforms = transform

    # ここで取り出すデータを指定している
    def __getitem__(self,index):
        img_path = self.dataset_path + self.df["Path"][index]
        json_path = self.json_dir + img_path.split("/")[-1].split(".")[0] + ".json"
        with open(json_path) as f:
            di = json.load(f)
        points = di["shapes"][0]["points"]
        shape_type = di["shapes"][0]["shape_type"]
        img = Image.open(img_path) #ここで画像の読み込み
        # データの変形 (transforms)
        img = self.transforms(img)
        label = self.df["ClassId"][index]
        file_name = di["imagePath"].split("/")[-1]
        return img, label, shape_type, file_name, points

    # この method がないと DataLoader を呼び出す際にエラーを吐かれる
    def __len__(self):
        return len(self.df)
    
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

def get_inside_points(center,radius,img_size=32):
    # 円の内側の座標を格納するリスト
    inside_circle = []

    # 画像内のすべてのピクセルを調べる
    for x in range(img_size):
        for y in range(img_size):
            # ピクセルが円の内部にあるかどうかを判定
            if (x - center[0]) ** 2 + (y - center[1]) ** 2 < radius ** 2:
                inside_circle.append((x, y))
    return inside_circle

if __name__=="__main__":
    meta_train_path = "../data/dataset/GTSRB/Train.csv"
    dataset_path = '../data/dataset/GTSRB/'
    batch_size = 1
    annotation_dir = "../data/dataset/GTSRB/annotation/"
    save_path = "../data/adv_img/shaped_shadow"
    # 訓練ループのサンプル
    num_classes = 1 #5  # 円、三角形、逆三角形、八角形、菱形の5クラス
    #model = ObjectDetectionModel(num_classes)
    #optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)


    # データの前処理
    transform = transforms.Compose([
        transforms.Resize((32,32)),  # 画像を32x32にリサイズ
        transforms.ToTensor(),          # 画像をテンソルに変換
        #transforms.Normalize(mean=[0.3403, 0.3121, 0.3214], std=[0.1595, 0.1590, 0.1683])  # 正規化(訓練データのmeanとstd)
    ])
    ann_dataset = Annotation(dataset_path, meta_train_path, annotation_dir, transform)
    data_size = len(ann_dataset)
    ann_loader = DataLoader(ann_dataset,batch_size=batch_size,shuffle=False)

    count = 0
    for image, label, shape_type, file_name, points in tqdm(ann_loader):
        image = image[0]
        file_name = file_name[0]
        center = torch.Tensor(points[0])
        radius_tensor = torch.Tensor(points[0])-torch.Tensor(points[1])
        radius = torch.norm(radius_tensor)
        inside_points = get_inside_points(center,radius)
        shaped_shadow_img = make_ichou_shadow(image, inside_points)
        image = Image.fromarray(shaped_shadow_img)
        image.save(f"{save_path}/{file_name}")

        """image = image.permute(1, 2, 0).cpu().numpy()
        image = (image * 255).astype(np.uint8)  # 正規化を逆にする
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if polygon_info[0][0] == "circle":
            shape_type, radius, center_x, center_y = polygon_info
            shape_type, radius, center_x, center_y = shape_type, int(radius), int(center_x), int(center_y)
        image_with_circle = cv2.circle(image, (center_x, center_y), radius, (0, 255, 0), 1)
        
        # 画像を保存
        cv2.imwrite(f"{annotated_path}/{count}.png",image_with_circle)
        count += 1"""