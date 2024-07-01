import pickle
import cv2 as cv
import random
import numpy as np
import os
import sys
from functions import *
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd
import re

# 影の開始ポイント
shadow_point = ["top", "right", "bottom", "left"]
if len(sys.argv) > 1:
    N = float(sys.argv[1])  # Nをfloat型に変換
else:
    print("影の比率を入力してください")
    exit()

ichou_shadow_dir = "../data/tree/"

def draw_shadow(image, pattern):
    mask_img = cv.imread(ichou_shadow_dir + f"ichou_shadow{str(pattern)}.png")
    if mask_img is None:
        raise FileNotFoundError(f"Image not found: {ichou_shadow_dir}ichou_shadow{str(pattern)}.png")

    # 画像をNumPy配列に変換し、チャネル順序を変換
    image = image.permute(1, 2, 0).numpy()  # (C, H, W) -> (H, W, C)
    image = (image * 255).astype(np.uint8)  # 0-1から0-255の範囲に変換
    
    img = cv.cvtColor(image, cv.COLOR_RGB2Lab)

    # 白いピクセルのマスクを作成
    white_mask = np.all(mask_img == [255, 255, 255], axis=-1)

    # Lチャンネルに対して白いピクセルの位置を0.4倍する
    img[white_mask, 0] = img[white_mask, 0] * N

    # 値の範囲を0-255にクリップ
    img = np.clip(img, 0, 255).astype(np.uint8)

    # LabからRGB空間に変換
    img = cv.cvtColor(img, cv.COLOR_Lab2RGB)

    return img

def make_ichou_shadow(image):
    tree_image_dir = "../data/tree/"
    dir_list = os.listdir(tree_image_dir)
    pattern = r'ichou(\d+)'
    shadow_pattern = []
    for f in dir_list:
        match = re.search(pattern, f)
        if match:
            shadow_pattern.append(int(match.group(1)))
    pattern = random.choice(shadow_pattern)
    draw_image = draw_shadow(image, pattern)
    return draw_image

if __name__ == "__main__":
    dataset_path = '../data/dataset/GTSRB/'
    meta_test_path = "../data/dataset/GTSRB/Test.csv"
    meta_shadow_test_path = f"../data/adv_img/tree/Test/Test_{int(N*100)}.csv" # 作成したメタデータを保存するパス
    adv_shadow_path = f"../data/adv_img/tree/Test/{int(N*100)}/"
    transform = transforms.Compose([
        transforms.Resize((32, 32)),  # 画像を32x32にリサイズ
        transforms.ToTensor(),        # 画像をテンソルに変換
        # transforms.Normalize(mean=[0.3403, 0.3121, 0.3214], std=[0.1595, 0.1590, 0.1683])  # 正規化(訓練データのmeanとstd)
    ])
    batch_size = 1
    test_dataset = MyDataset(dataset_path=dataset_path, meta_data_path=meta_test_path, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    df = pd.DataFrame(columns=['ClassId', 'Path'])

    
    index = 0
    for images, labels in tqdm(test_loader):
        p = make_ichou_shadow(images[0])
        df.loc[index] = [int(labels[0]),f"{int(N*100)}/{index}.png"]
        cv.imwrite(adv_shadow_path+f"{index}.png", cv.cvtColor(p, cv.COLOR_RGB2BGR))
        index += 1

    df.to_csv(meta_shadow_test_path,index=False)

