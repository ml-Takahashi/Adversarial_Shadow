### ここにshadow_attack.pyをコピーして、テンプレートマッチングで画像を分類するためのコードを書く.
### そのクラスの1枚目の画像をテンプレート画像として使用し、一致度が99%以上なら保存もカウントもせず次に行く.

import os
import cv2
import matplotlib.pyplot as plt
import numpy as np

zoom_rate = 0.5
dataset = "LISA"
file_type = "bmp"
template_dir = f"template/{dataset}/{int(zoom_rate*100)}"
count = 0
predict_list = []
label_list = []
pct_list = [0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2,1.3,1.4,1.5]



try:
    os.mkdir(template_dir)
except FileExistsError:
    print("FileExists!")
    """for name in os.listdir(save_dir):
        os.remove(os.path.join(save_dir, name))"""
class_num = len(os.listdir(template_dir))

now_label = -1


for f in os.listdir(template_dir):
    template = cv2.imread(f"{template_dir}/{f}")
    if "_" not in f:
        label = int(f.split(".")[0])
        for i,p in enumerate(pct_list):
            temp_shape = (int(template.shape[0]*p),int(template.shape[1]*p))
            new_template = cv2.resize(template,temp_shape)
            cv2.imwrite(f"{template_dir}/{label}_{i}.jpg",new_template)

