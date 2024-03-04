# テンプレート画像を保存するためのファイル

import cv2
import pickle
import os

"../data/template/GTSRB/ten_pic"
with open('../data/dataset/GTSRB/train.pkl', 'rb') as f:
    train = pickle.load(f)
    train_x, train_y = train['data'], train['labels']
with open('../data/dataset/GTSRB/test.pkl', 'rb') as f:
    test = pickle.load(f)
    test_x, test_y = test['data'], test['labels']

for image,label in zip(train_x,train_y):
    # trainingデータ保存
    if not os.path.exists(f"../data/dataset/GTSRB/training/{label}/"):
        os.makedirs(f"../data/dataset/GTSRB/training/{label}/")
        count = 0
    cv2.imwrite(f"../data/dataset/GTSRB/training/{label}/{label}_{count}.jpg",image)
    count += 1
