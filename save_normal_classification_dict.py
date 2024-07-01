# テストデータの分類結果をグラフに起こすためのコード

# トレーニングデータ可視化のためのpythonファイル

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from functions import MyDataset
import os
from torchvision import transforms
from torch.utils.data import DataLoader
from functions import GtsrbCNN,MyDataset
from tqdm import tqdm
import pickle

transform = transforms.Compose([
    transforms.Resize((32,32)),  # 画像を32x32にリサイズ
    transforms.ToTensor(),          # 画像をテンソルに変換
    transforms.Normalize(mean=[0.3403, 0.3121, 0.3214], std=[0.1595, 0.1590, 0.1683])  # 正規化(訓練データのmeanとstd)
])

# データセットのパスを指定
dataset_path = '../data/dataset/GTSRB/'
meta_train_path = "../data/dataset/GTSRB/Train.csv"
meta_test_path = "../data/dataset/GTSRB/Test.csv"
#テスト用のデータ用意してからtrain_test_loopを作る!
save_path = "../data/model/normalized_model_gtsrb.pth" #pthで合ってる？
count_dict_path = "../data/dict/count_test_images.pkl"
normal_acc_path = "../data/dict/count_normal_acc.pkl"
n_class = 43
batch_size = 1
epoch_num = 100
start_epoch = 0

model = GtsrbCNN(n_class)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
loss_func = nn.CrossEntropyLoss()
if os.path.isfile(save_path):
    checkpoint = torch.load(save_path)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    start_epoch = checkpoint["epoch"]
else:
    print(f"The specified path does not exist")
    exit()


test_dataset = MyDataset(dataset_path, meta_test_path, transform)
test_loader = DataLoader(test_dataset,batch_size=batch_size,shuffle=False)



if not os.path.isfile(count_dict_path):
    test_count = dict()
    for images,labels in tqdm(test_loader):
        label = int(labels)
        if label in test_count.keys():
            test_count[label] += 1
        else:
            test_count[label] = 1
    test_count = dict(sorted(test_count.items(), key=lambda x:x[0]))
    with open(count_dict_path,"wb") as f:
        pickle.dump(test_count, f)
else:
    with open(count_dict_path, 'rb') as f:
        test_count = pickle.load(f)

if not os.path.isfile(normal_acc_path):
    acc_count = {key: 0 for key in range(43)}
    model.eval()
    with torch.no_grad():
        for x,y in tqdm(test_loader):
            label = int(y)
            test_pred = model(x)
            if (torch.argmax(test_pred, dim=1) == y):
                acc_count[label] += 1
    with open(normal_acc_path,"wb") as f:
        pickle.dump(acc_count, f)
else:
    with open(normal_acc_path, 'rb') as f:
        acc_count = pickle.load(f)