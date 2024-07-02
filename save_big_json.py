# 物体検出に使用するjsonファイルを1つ作成し多数の辞書をまとめて保存
# 今は円状の標識のみ対象

import torch
from torch.utils.data import Dataset
from torchvision.models import ResNet50_Weights
import torch.nn as nn
import torchvision.models as models
from tqdm import tqdm
import time
import pandas as pd
import numpy as np
from PIL import Image
import cv2
from torchvision import transforms
from torch.utils.data import DataLoader
import json



# 訓練用アノテーションクラス（shape_typeがわかっている前提のため）
class Annotation(Dataset):
    def __init__(self, dataset_path, meta_data_path,transform):
        super().__init__()
        df = pd.read_csv(meta_data_path)
        df = df[df["shape"]=="circle"].reset_index(drop=True) # 円の画像だけ
        self.path_data = dataset_path + df["Path"]
        self.class_data = df["ClassId"]
        self.x1, self.y1 = df.new_x1, df.new_y1
        self.x2, self.y2 = df.new_x2, df.new_y2
        self.shape_type = df["shape"]
        self.transforms = transform

    # ここで取り出すデータを指定している
    def __getitem__(self,index):
        path = self.path_data[index]
        img = Image.open(path) #ここで画像の読み込み
        # データの変形 (transforms)
        img = self.transforms(img)
        label = self.class_data[index]
        polygon_info = get_polygon_info(self.x1[index],self.y1[index],self.x2[index],self.y2[index],self.shape_type[index])
        return img, label,self.path_data[index], polygon_info

    # この method がないと DataLoader を呼び出す際にエラーを吐かれる
    def __len__(self):
        return len(self.path_data)
def get_filename(f_path):
    return f_path.split("/")[-1].split(".")[0]
    
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
    annotated_path = "../data/annotation"
    dict_list = []

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
    for images, labels, file_path,polygon_info in tqdm(train_loader):
        label = int(labels[0])
        file_name = get_filename(file_path[0])
        json_name = file_name+".json"

        if polygon_info[0][0] == "circle":
            shape_type, radius, center_x, center_y = polygon_info
            shape_type, radius, center_x, center_y = shape_type[0], int(radius), int(center_x), int(center_y)
        
        if shape_type == "circle":
        #center以外の点をどこを取るか決める処理
            if (center_x + radius) < 32:
                p  = [(center_x + radius), center_y]
            elif (center_x - radius) > 0:
                p = [(center_x - radius), center_y]
            elif (center_y + radius) < 32:
                p  = [(center_y + radius), center_x]
            elif (center_y - radius) > 0:
                p = [(center_y - radius), center_x]
        
            points = [[center_x,center_y], p]
        
        di = {
            "shapes":[{
            "label": label,
            "points": points,
            "shape_type": "circle"
            }],
            "imagePath": file_path[0]
            }
        dict_list.append(di)
        
    save_path = f"../data/dataset/GTSRB/circle_annotation.json"
    with open(save_path, 'wt') as f:
        json.dump(dict_list, f)
