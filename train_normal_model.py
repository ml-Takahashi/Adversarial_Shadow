import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader
import os
from functions import GtsrbCNN,train_test_loop,MyDataset
from torch.utils.data import random_split


# データの前処理
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
save_path = "../data/model/normalized_model_gtsrb.pth" #pthで合ってる？ # ../data/model/model_gtsrb.pth
n_class = 43 #len(os.listdir(dataset_path+"Train"))
batch_size = 64
epoch_num = 100
start_epoch = 0
split_num = 1

train_dataset = MyDataset(dataset_path, meta_train_path,transform)
data_size = len(train_dataset)
train_dataset, _ = random_split(train_dataset,[data_size//split_num,data_size - data_size//split_num])
train_loader = DataLoader(train_dataset,batch_size=batch_size,shuffle=True)


test_dataset = MyDataset(dataset_path, meta_test_path,transform)
test_size = len(test_dataset)
test_dataset, _ = random_split(test_dataset,[test_size//split_num,test_size - test_size//split_num])
test_loader = DataLoader(test_dataset,batch_size=batch_size,shuffle=True)

model = GtsrbCNN(n_class)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
loss_func = nn.CrossEntropyLoss()
if os.path.isfile(save_path):
    checkpoint = torch.load(save_path)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    start_epoch = checkpoint["epoch"]

train_test_loop(train_loader,test_loader,model,loss_func,optimizer,save_path,start_epoch,epoch_num)

