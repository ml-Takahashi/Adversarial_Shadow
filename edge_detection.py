import cv2
import numpy as np

# 入力画像の読み込み
input_image = cv2.imread('../data/dataset/GTSRB/Train/0/00000_00000_00000.png')

# グレースケールに変換
gray_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)

# Cannyエッジ検出
edges = cv2.Canny(gray_image, 50, 200)

# 輪郭の検出
contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# 輪郭を描画
cv2.drawContours(input_image, contours, -1, (0, 255, 0), 2)

# 結果を表示
cv2.imshow('Contours', input_image)
cv2.waitKey(0)
cv2.destroyAllWindows()