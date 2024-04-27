# テンプレート画像を保存するためのファイル

import cv2
import pickle
import os
import random
import sys
import shutil

with open('../data/dataset/GTSRB/train.pkl', 'rb') as f:
    train = pickle.load(f)
    train_x, train_y = train['data'], train['labels']
with open('../data/dataset/GTSRB/test.pkl', 'rb') as f:
    test = pickle.load(f)
    test_x, test_y = test['data'], test['labels']

count = dict()
for label in train_y:
    if label in count.keys():
        count[label] += 1
    else:
        count[label] = 1

template_num = int(sys.argv[1])
data_dir = "../data/dataset/GTSRB/training"
save_dir = f"../data/template/GTSRB/{template_num}_pic"

if os.path.exists(save_dir):
    shutil.rmtree(save_dir)

os.makedirs(save_dir)


for label in range(42):
    path_list = os.listdir(f"{data_dir}/{label}")
    path_list.sort()
    random_list = random.sample(path_list,template_num)
    for i,f_name in enumerate(random_list):
        img = cv2.imread(f"{data_dir}/{label}/{f_name}")
        cv2.imwrite(f"{save_dir}/{label}_{i}.jpg",img)

        

    # trainingデータ保存
    """if not os.path.exists(f"../data/dataset/GTSRB/training/{label}/"):
        os.makedirs(f"../data/dataset/GTSRB/training/{label}/")
        count = 0
    cv2.imwrite(f"../data/dataset/GTSRB/training/{label}/{label}_{count}.jpg",image)
    count += 1"""
