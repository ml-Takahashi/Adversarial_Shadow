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
from shaped_shadow_func import *

N = 0.43
ichou_shadow_dir = "../data/tree/"

def make_mask(image, inside_points, leaf_num=10):
    # ここで敵対的な影のマスクを作成する予定
    pass

# 標識部分に影をつけて返すクラス
class Annotation(Dataset):
    def __init__(self, dataset_path, meta_data_path, transform):
        super().__init__()
        self.dataset_path = dataset_path
        self.df = pd.read_csv(meta_data_path)
        self.df = self.df[self.df["shape"].isin(["circle", "triangle"])]
        self.df.reset_index(drop=True, inplace=True)
        self.transform = transform
        self.label_dict = {"circle": 0, "triangle": 1, "r_triangle": 2, "rectangle":3, "octagon":4}

    def __len__(self):
        return len(self.df)

    # ここで取り出すデータを指定している
    def __getitem__(self,idx):
        img_path = self.dataset_path + self.df["Path"][idx]
        file_name = img_path.split("/")[-1]
        image = Image.open(img_path)
        image = self.transform(image)
        label = torch.tensor([self.label_dict[self.df["shape"][idx]]],dtype=torch.int64) #self.df["ClassId"][idx]
        box = torch.tensor([self.df["new_x1"][idx], self.df["new_y1"][idx], self.df["new_x2"][idx], self.df["new_y2"][idx]], dtype=torch.int64).reshape(-1,4)
        shape_type = self.df["shape"][idx]
        box_info = get_box_info(box)
        if shape_type == "circle":
            target = get_circle_info(*box_info)
            points_in_circle = get_points_in_circle(target)
            shaped_shadow_img = make_ichou_shadow(image, points_in_circle)
        elif shape_type == "triangle":
            target = get_triangle_info(*box_info)
            points_in_triangle = get_points_in_triangle(*target)
            shaped_shadow_img = make_ichou_shadow(image, points_in_triangle)
        return shaped_shadow_img, file_name


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
    ann_dataset = Annotation(dataset_path, meta_train_path, transform)
    data_size = len(ann_dataset)
    ann_loader = DataLoader(ann_dataset,batch_size=batch_size,shuffle=False)

    for image, file_name in tqdm(ann_loader):
        image = image[0].numpy()
        #image = image.permute(1, 2, 0).cpu().numpy()
        #image = (image * 255).astype(np.uint8)  # 正規化を逆にする
        file_name = file_name[0]
        image = Image.fromarray(image)
        #image.save(f"{save_path}/{file_name}")
        image.save(f"{save_path}/{file_name}")
