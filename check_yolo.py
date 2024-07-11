import torch

# YOLOv5sモデルのロード
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# カスタムデータセット用にモデルを設定
model.nc = 5  # クラス数を設定
print(model)