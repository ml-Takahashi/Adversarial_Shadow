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

def is_point_in_triangle(pt, a, b, c):
    """
    Check if a point is inside a given triangle using the barycentric technique.
    
    :param pt: Tuple (x, y) - The point to check.
    :param a: Tuple (x, y) - The first vertex of the triangle.
    :param b: Tuple (x, y) - The second vertex of the triangle.
    :param c: Tuple (x, y) - The third vertex of the triangle.
    :return: True if the point is inside the triangle, False otherwise.
    """
    def sign(p1, p2, p3):
        return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])

    d1 = sign(pt, a, b)
    d2 = sign(pt, b, c)
    d3 = sign(pt, c, a)

    has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
    has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0)

    return not (has_neg and has_pos)

def get_points_in_triangle(a, b, c, image_width, image_height):
    """
    Get all the points inside the given triangle.
    
    :param a: Tuple (x, y) - The first vertex of the triangle.
    :param b: Tuple (x, y) - The second vertex of the triangle.
    :param c: Tuple (x, y) - The third vertex of the triangle.
    :param image_width: Width of the image.
    :param image_height: Height of the image.
    :return: List of tuples - Points inside the triangle.
    """
    points_in_triangle = []

    for x in range(image_width):
        for y in range(image_height):
            if is_point_in_triangle((x, y), a, b, c):
                points_in_triangle.append((x, y))

    return points_in_triangle

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

def get_box_info(box):
    box = box[0]
    x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
    width = x2 - x1
    height = y2 - y1
    return  x1, y1, x2, y2, width, height

def get_circle_info(x1, y1, x2, y2, width, height):
    target = {"shape_type": "circle"}
    # 円の半径は長方形の幅と高さの小さい方の半分
    radius = min(width, height) // 2
    # 円の中心は長方形の中心
    center_x = x1 + width // 2
    center_y = y1 + height // 2
    target["radius"] = radius
    target["center"] = [center_x, center_y]
    return target

"""def get_triangle_info(x1, y1, x2, y2, width, height):
    top_x = x1 + width//2
    top_y = y1
    left_x, left_y = x1, y2
    right_x, right_y = x2, y2
    return [top_x,top_y], [left_x,left_y], [right_x,right_y], width, height"""
def get_triangle_info(x1, y1, x2, y2, width, height):
    top_y = x1 + width//2
    top_x = y1
    left_y, left_x = x1, y2
    right_y, right_x = x2, y2
    return [top_x,top_y], [left_x,left_y], [right_x,right_y], width, height

    """elif shape_type == "r_triangle":
        bottom_x = x1 + width//2
        bottom_y = y2
        left_x, left_y = x1, y1
        right_x, right_y = x2, y1
        return bottom_x, bottom_y, left_x, left_y, right_x, right_y
    elif shape_type == "rectangle":"""

def get_points_in_circle(target,img_size=32):
    center = target["center"]
    radius = target["radius"]
    # 円の内側の座標を格納するリスト
    points_in_circle = []
    # 画像内のすべてのピクセルを調べる
    for x in range(img_size):
        for y in range(img_size):
            # ピクセルが円の内部にあるかどうかを判定
            if (x - center[0]) ** 2 + (y - center[1]) ** 2 < radius ** 2:
                points_in_circle.append((x, y))
    return points_in_circle