

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.models import ResNet50_Weights
import torch.nn as nn
import torchvision.models as models
from torch.utils.data.dataloader import DataLoader
from torchvision import transforms

import os
import numpy as np
from PIL import Image
import pickle
from tqdm import tqdm
import time


class My_Dataset(Dataset):

    def __init__(self,file_list,transform=transforms.ToTensor()):
        self.file_list = file_list
        self.transform = transform
    
    def __len__(self):
        return len(self.file_list)

    def __getitem__(self,idx):
        img_path = self.file_list[idx]
        img = Image.open(img_path)
        img_transformed = self.transform(img)

        label = int(img_path.split("/")[4])
        return img_transformed,label

class GtsrbCNN(nn.Module):

    def __init__(self,n_class):

        super().__init__()
        self.res_block = models.resnet50(weights=ResNet50_Weights.DEFAULT)
        self.res_block.conv1 = nn.Conv2d(3, 64,kernel_size=(3,3))
        self.res_block.fc = nn.Linear(2048, n_class, bias=True)

    def forward(self, x):

        x = self.res_block(x)
        return x

def make_file_list(dir_path):
    dir_list = os.listdir(dir_path)
    file_list = []
    for d in dir_list:
        label_path = os.path.join(dir_path,d)
        f_list = os.listdir(label_path)
        file_list.extend([os.path.join(label_path,f) for f in f_list])
    return file_list

def train_test_loop(train_file_list, test_file_list, model, start_epoch, epoch_num, batch_size, loss_func, optimizer, save_path):

    max_loss = 10000

    for epoch in range(start_epoch,start_epoch+epoch_num):
        train_acc, train_loss, test_acc, test_loss = 0, 0, 0, 0
        epoch_start_time = time.time()
        train_dataset = My_Dataset(train_file_list)
        test_dataset = My_Dataset(test_file_list)
        train_num = train_dataset.__len__()
        test_num = test_dataset.__len__()

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)
        model.train()
        for x,y in tqdm(train_loader):
            train_pred = model(x)
            batch_loss = loss_func(train_pred,y)
            batch_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            train_acc += (torch.argmax(train_pred, dim=1) == y).sum()
            train_loss += batch_loss.item() * len(y)

        model.eval()
        with torch.no_grad():
            for x,y in tqdm(test_loader):
                test_pred = model(x)
                batch_loss = loss_func(test_pred,y)
                test_acc += (torch.argmax(test_pred, dim=1) == y).sum()
                test_loss += batch_loss.item() * len(y)

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

    
if __name__ == "__main__":
    save_path = "../data/model/random_shadow_trained_model/model.tar"
    train_dir = "../data/shadow_data/training"
    test_dir = "../data/shadow_data/test"
    train_file_list = make_file_list(train_dir)
    test_file_list = make_file_list(test_dir)

    n_class = len(os.listdir(train_dir))
    batch_size = 64
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

    train_test_loop(train_file_list=train_file_list, test_file_list=test_file_list, model=model, start_epoch=start_epoch,epoch_num=epoch_num,  batch_size=batch_size, loss_func=loss_func, optimizer=optimizer, save_path=save_path)
    
