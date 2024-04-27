import cv2
import numpy as np
import random

def draw_and_fill(input_image_path, output_image_path):
    # 画像を読み込む
    image = cv2.imread(input_image_path)

    # 画像の高さと幅を取得
    height, width = image.shape[:2]

    # 2点をランダムで選ぶ（違う辺上から選ぶ）
    edge = random.choice(['top', 'bottom', 'left', 'right'])

    if edge == 'top':
        pt1 = (random.randint(0, width - 1), 0)
        pt2 = (random.randint(0, width - 1), 0)
    elif edge == 'bottom':
        pt1 = (random.randint(0, width - 1), height - 1)
        pt2 = (random.randint(0, width - 1), height - 1)
    elif edge == 'left':
        pt1 = (0, random.randint(0, height - 1))
        pt2 = (0, random.randint(0, height - 1))
    elif edge == 'right':
        pt1 = (width - 1, random.randint(0, height - 1))
        pt2 = (width - 1, random.randint(0, height - 1))

    # 2点を結ぶ直線を引く
    cv2.line(image, pt1, pt2, (255, 255, 255), 2)

    # 直線より下の領域を真っ黒に塗りつぶす
    mask = np.zeros_like(image)
    fill_points = np.array([pt1, pt2, (width // 2, height // 2)])
    cv2.fillPoly(mask, [fill_points], (0, 0, 0))
    image = cv2.bitwise_and(image, mask)

    # 結果を保存
    cv2.imwrite(output_image_path, image)

# テスト用
draw_and_fill('input_image.jpg', 'output_image.jpg')