# トレーニングデータ可視化のためのpythonファイル

import pickle
import matplotlib.pyplot as plt
from functions import MyDataset
import os
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
    

transform = transforms.Compose([
    transforms.Resize((32,32)),  # 画像を32x32にリサイズ
    transforms.ToTensor(),          # 画像をテンソルに変換
    #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 正規化
])

# データセットのパスを指定
dataset_path = '../data/dataset/GTSRB/'
meta_train_path = "../data/dataset/GTSRB/Train.csv"
meta_test_path = "../data/dataset/GTSRB/Test.csv"
#テスト用のデータ用意してからtrain_test_loopを作る!
save_path = "data/model/normal_model_gtsrb.pth" #pthで合ってる？
n_class = len(os.listdir(dataset_path+"Train"))
batch_size = 1
epoch_num = 100
start_epoch = 0

train_dataset = MyDataset(dataset_path, meta_train_path,transform)
train_dataloader = DataLoader(train_dataset,batch_size=batch_size,shuffle=False)

test_dataset = MyDataset(dataset_path, meta_test_path, transform)
test_dataloader = DataLoader(test_dataset,batch_size=batch_size,shuffle=False)


train_count = dict()
for images,labels in tqdm(train_dataloader):
    label = int(labels)
    if label in train_count.keys():
        train_count[label] += 1
    else:
        train_count[label] = 1
train_count = dict(sorted(train_count.items(), key=lambda x:x[0]))

fig1, ax1 = plt.subplots()
ax1.bar(range(len(train_count)),train_count.values())
plt.xticks(range(len(train_count)), train_count.keys())
plt.tight_layout()
plt.xticks(fontsize=7)
fig1.savefig('../data/training_class.png')
plt.close(fig1)


test_count = dict()
for images,labels in tqdm(test_dataloader):
    label = int(labels)
    if label in test_count.keys():
        test_count[label] += 1
    else:
        test_count[label] = 1
test_count = dict(sorted(test_count.items(), key=lambda x:x[0]))

fig2, ax2 = plt.subplots()
ax2.bar(range(len(test_count)),test_count.values())
plt.xticks(range(len(test_count)), test_count.keys())
plt.tight_layout()
plt.xticks(fontsize=7)
fig2.savefig('../data/test_class.png')
plt.close(fig2)