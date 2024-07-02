# 通常の学習に必要なクラス・関数のみ宣言

import torch
from torch.utils.data import Dataset
from torchvision.models import ResNet50_Weights
import torch.nn as nn
import torchvision.models as models
from tqdm import tqdm
import time
import pandas as pd
from PIL import Image


class MyDataset(Dataset):
    def __init__(self, dataset_path, meta_data_path, transform):
        super().__init__()
        self.path_data = dataset_path + pd.read_csv(meta_data_path)["Path"]
        self.class_data = pd.read_csv(meta_data_path)["ClassId"]
        self.transforms = transform

    # ここで取り出すデータを指定している
    def __getitem__(self,index):
        path = self.path_data[index]
        img = Image.open(path) #ここで画像の読み込み
        # データの変形 (transforms)
        img = self.transforms(img)
        label = int(self.class_data[index])
        return img, label

    # この method がないと DataLoader を呼び出す際にエラーを吐かれる
    def __len__(self):
        return len(self.path_data)

class GtsrbCNN(nn.Module):

    def __init__(self,n_class):
        super().__init__()
        self.res_block = models.resnet50(weights=ResNet50_Weights.DEFAULT)
        self.res_block.conv1 = nn.Conv2d(3, 64,kernel_size=(3,3))
        self.res_block.fc = nn.Linear(2048, n_class, bias=True)

    def forward(self, x):
        x = self.res_block(x)
        return x
    
def train_test_loop(train_loader, test_loader, model, loss_func, optimizer, save_path, start_epoch, epoch_num):

    max_loss = 10000

    for epoch in range(start_epoch,start_epoch+epoch_num):
        train_acc, train_loss, test_acc, test_loss = 0, 0, 0, 0
        epoch_start_time = time.time()
        model.train()
        train_num = 0
        for x,y in tqdm(train_loader):
            train_pred = model(x)
            batch_loss = loss_func(train_pred,y)
            batch_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            train_acc += (torch.argmax(train_pred, dim=1) == y).sum()
            train_loss += batch_loss.item() * len(y)
            train_num += len(y)

        model.eval()
        test_num = 0
        with torch.no_grad():
            for x,y in tqdm(test_loader):
                test_pred = model(x)
                batch_loss = loss_func(test_pred,y)
                test_acc += (torch.argmax(test_pred, dim=1) == y).sum()
                test_loss += batch_loss.item() * len(y)
                test_num += len(y)
                

        if (test_loss/test_num) < max_loss:
            max_loss = test_loss/test_num
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),},
                save_path)

        print(f'[{epoch+1}/{epoch_num}] {round(time.time() - epoch_start_time, 2)}', end=' ')
        print(f'Train Acc: {round(float(train_acc / train_num), 4)}', end=' ')
        print(f'Loss: {round(float(train_loss / train_num), 4)}', end=' | ')
        print(f'Test Acc: {round(float(test_acc / test_num), 4)}', end=' ')
        print(f'Loss: {round(float(test_loss / test_num), 4)}')