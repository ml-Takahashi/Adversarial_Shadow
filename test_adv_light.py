import cv2
import numpy as np
import sys


def make_light_image(image, coordinate, radius, brightness_increase_ratio):
    # 画像をBGRからLab色空間に変換
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
    # マスクを作成
    mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
    
    # ガウシアンブラーを適用してグラデーションを作成
    mask = cv2.GaussianBlur(mask, (0, 0), radius / 2)
    cv2.circle(mask, coordinate, radius, (255), -1)  # 白い円を描く
    
    # 円の内側の座標を取得
    points_in_circle = np.column_stack(np.where(mask == 255))
    
    # Lチャンネルを分離
    L, a, b = cv2.split(image)
    mask = mask/255
    
    # Lチャンネルに対して指定した比率で明るさを増加
    for (y, x) in points_in_circle:
        L[y, x] = min(255, L[y, x] + L[y, x] * mask[y, x]*(brightness_increase_ratio - 1))

    # 修正後のLチャンネルと元のa、bチャンネルを結合
    image = cv2.merge((L, a, b))
    # 画像をLabからBGR色空間に戻す

    image_with_glare = cv2.cvtColor(image, cv2.COLOR_Lab2BGR)
    return image_with_glare


output_dir = "../data/light_example"
if len(sys.argv) != 2:
    print("Usage: python script.py <number>")
    sys.exit(1)

input_str = sys.argv[1]

try:
    brightness_increase_ratio = float(input_str)
    print(f"The value is: {brightness_increase_ratio}")
except ValueError:
    print("Error: Please provide a valid integer.")
    sys.exit(1)

# 画像を読み込む
image = cv2.imread('../data/sample_sign.png')

# 画像を32x32にリサイズ
image = cv2.resize(image, (32, 32))

# 白飛びをシミュレートする中心座標と半径
center_x, center_y = 10, 10  # 白飛びの中心位置
radius = 7  # 白飛びの半径


image = make_light_image(image, (center_x,center_y), radius, brightness_increase_ratio)
cv2.imwrite(f'{output_dir}/output_image_{brightness_increase_ratio}.jpg', image)
exit()


# マスクを作成
mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
cv2.circle(mask, (center_x, center_y), radius, (255), -1)  # 白い円を描く

# 円の内側の座標を取得
coordinates = np.column_stack(np.where(mask == 255))

# Lチャンネルを分離
L, a, b = cv2.split(image)

# ガウシアンブラーを適用してグラデーションを作成
mask = cv2.GaussianBlur(mask, (0, 0), radius / 2)
cv2.imwrite(f"{output_dir}/output_blur.png", mask)

mask = mask/255
# Lチャンネルに対して指定した比率で明るさを増加
for (y, x) in coordinates:
    L[y, x] = min(255, L[y, x] + L[y, x] * mask[y, x]*(brightness_increase_ratio - 1))

# 修正後のLチャンネルと元のa、bチャンネルを結合
image = cv2.merge((L, a, b))

# 画像をLabからBGR色空間に戻す
image_with_glare = cv2.cvtColor(image, cv2.COLOR_Lab2BGR)

# 画像を保存する
cv2.imwrite(f'{output_dir}/output_image_{brightness_increase_ratio}.jpg', image_with_glare)
