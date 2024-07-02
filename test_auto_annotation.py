import cv2
import numpy as np

def draw_inner_circle(rect_top_left, rect_bottom_right, image):
    """
    長方形の内側に接する円を画像上に描画する関数。

    Args:
        rect_top_left (tuple): 長方形の左上の座標 (x1, y1)。
        rect_bottom_right (tuple): 長方形の右下の座標 (x2, y2)。
        image (np.ndarray): 画像。

    Returns:
        np.ndarray: 円を描画した画像。
    """
    # 長方形の座標
    x1, y1 = rect_top_left
    x2, y2 = rect_bottom_right

    # 長方形の幅と高さを計算
    width = x2 - x1
    height = y2 - y1

    # 円の半径は長方形の幅と高さの小さい方の半分
    radius = min(width, height) // 2

    # 円の中心は長方形の中心
    center_x = x1 + width // 2
    center_y = y1 + height // 2

    # 円を描画
    image_with_circle = cv2.circle(image, (center_x, center_y), radius, (0, 255, 0), 1)

    return image_with_circle

# テスト用の画像を作成
image = cv2.imread("../data/dataset/GTSRB/Train/0/00000_00002_00006.png")

# 長方形の座標
rect_top_left = (5, 5)
rect_bottom_right = (25, 25)

# 円を描画
image_with_circle = draw_inner_circle(rect_top_left, rect_bottom_right, image)

# 画像を表示
cv2.imwrite("p.png",image_with_circle)
