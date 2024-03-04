### ここにshadow_attack.pyをコピーして、テンプレートマッチングで画像を分類するためのコードを書く.
### そのクラスの1枚目の画像をテンプレート画像として使用し、一致度が99%以上なら保存もカウントもせず次に行く.

import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import sys

try:
    template_num = sys.argv[1]
except IndexError:
    print("Enter template_num")
    exit()
zoom_rate = 0.8
dataset = "GTSRB"
file_type = "bmp"
template_dir = f"../data/template/{dataset}/{template_num}_pic"
save_hconcat_dir = f"../data/check_position/{dataset}/{int(zoom_rate*100)}"
dir_path = f"../data/adv_img/{dataset}/43_no_attack"
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

file_list = os.listdir(dir_path)
file_num = len(file_list)
now_label = -1

for idx,f_name in tqdm(enumerate(file_list)):
    if file_type in f_name:
        img_path = os.path.join(dir_path,f_name)
        label = int(f_name.split("_")[1])
        image = cv2.imread(img_path)
        img_size = image.shape[0]

        # 下はテンプレート保存用のコード
        """template = image
        range = int(img_size*zoom_rate/2)*2
        center = int(img_size/2)
        start = center - int(range/2)
        end = center + int(range/2)
        template = image[start:end,start:end]
        cv2.imwrite(f"{template_dir}/{label}.jpg",template)"""
        

        max_similarity = 0
        predict = -1
        for f in os.listdir(template_dir):
            if "_" in f:
                template = cv2.imread(f"{template_dir}/{f}")
                #template = cv2.resize(template,(16,16))
                similarity_list = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)       
                #print(similarity.shape)  
                #print(template.shape)
                #exit()  
                
                
                temp_similarity = max(similarity_list[0])
                if temp_similarity >= 0.999:
                    file_num -= 1
                    break
                if temp_similarity > max_similarity:
                    max_similarity = temp_similarity
                    predict = int(f.split("_")[0])
                    best_template = template
        
        matching_point = similarity_list.argmax()
        row = matching_point//(image.shape[0]-1)
        column = matching_point%(image.shape[1]-1)
        save_img = cv2.rectangle(image,(column,row),(column+best_template.shape[0],row+best_template.shape[1]),(0,0,255))
        save_img = cv2.circle(save_img,center=(column, row),radius=1,color=(255, 0, 0),thickness=2,)
        save_img = cv2.hconcat([save_img, cv2.resize(best_template,(32,32))])
        cv2.imwrite(f"{save_hconcat_dir}/{idx}_{label}_{predict}.jpg",save_img)
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