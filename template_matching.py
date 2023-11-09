### ここにshadow_attack.pyをコピーして、テンプレートマッチングで画像を分類するためのコードを書く.
### そのクラスの1枚目の画像をテンプレート画像として使用し、一致度が99%以上なら保存もカウントもせず次に行く.

import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

zoom_rate = 0.6
dataset = "GTSRB"
file_type = "bmp"
template_dir = f"template/{dataset}/{int(zoom_rate*100)}"
count = 0
predict_list = []
label_list = []


try:
    os.mkdir(template_dir)
except FileExistsError:
    print("FileExists!")
    """for name in os.listdir(save_dir):
        os.remove(os.path.join(save_dir, name))"""
class_num = len(os.listdir(template_dir))

dir_path = f"adv_img/{dataset}/43_no_attack"
file_list = os.listdir(dir_path)
file_num = len(file_list)
now_label = -1

for f_name in tqdm(file_list):
    if file_type in f_name:
        img_path = os.path.join(dir_path,f_name)
        label = int(f_name.split("_")[1])
        image = cv2.imread(img_path)
        img_size = image.shape[0]

        # 下2行はテンプレート保存用のコード
        template = image
        range = int(img_size*zoom_rate/2)*2
        center = int(img_size/2)
        start = center - int(range/2)
        end = center + int(range/2)
        template = image[start:end,start:end]
        cv2.imwrite(f"{template_dir}/{label}.jpg",template)
        continue

        max_similarity = 0
        predict = -1
        for f in os.listdir(template_dir):
            if "_" in f:
                template = cv2.imread(f"{template_dir}/{f}")
                #template = cv2.resize(template,(16,16))
                similarity = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)       
                #print(similarity.shape)  
                #print(template.shape)
                #exit()  
                similarity = max(similarity[0])
                if similarity >= 0.99:
                    file_num -= 1
                    break
                if similarity > max_similarity:
                    max_similarity = similarity
                    #print(similarity)
                    predict = int(f.split("_")[0])
        predict_list.append(predict)
        label_list.append(label)
        if label==predict:
            count += 1
        #print(label,predict)
""""
plt.hist(predict_list)
plt.show()
plt.hist(label_list)
plt.show()"""
print(count/file_num)
print(file_num)
        
"""range = int(img_size*zoom_rate/2)*2
center = int(img_size/2)
start = center - int(range/2)
end = center + int(range/2)
template = image[start:end,start:end]"""
#cv2.imwrite(f"{save_dir}/{f_name}",template)
# テンプレートマッチングを行う
#res = cv2.matchTemplate(image, template, cv2.TM_CCORR_NORMED)
#print(res)