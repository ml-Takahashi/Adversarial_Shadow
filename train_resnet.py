import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.models import ResNet50_Weights
import torch.nn as nn
import torchvision.models as models
from torch.utils.data.dataloader import DataLoader

import os
import numpy as np
import pickle
from tqdm import tqdm
import time


class Traffic_Dataset(Dataset):

    def __init__(self,x,y):
        self.x = x 
        self.y = torch.LongTensor(y)
    
    def __len__(self):
        return len(self.x)

    def __getitem__(self,idx):
        _x = transforms.ToTensor()(self.x[idx])
        _y = self.y[idx]
        return _x,_y

class GtsrbCNN(nn.Module):

    def __init__(self,n_class):

        super().__init__()
        self.res_block = models.resnet50(weights=ResNet50_Weights.DEFAULT)
        self.res_block.conv1 = nn.Conv2d(3, 64,kernel_size=(3,3))
        self.res_block.fc = nn.Linear(2048, n_class, bias=True)

    def forward(self, x):

        x = self.res_block(x)
        return x

def training(train_data, train_labels, test_data, test_labels, 
             model, start_epoch, epoch_num, batch_size, loss_func, optimizer, save_path):

    max_loss = 10000

    for epoch in range(start_epoch,start_epoch+epoch_num):
        train_acc, train_loss, test_acc, test_loss = 0, 0, 0, 0
        epoch_start_time = time.time()
        train_dataset = Traffic_Dataset(train_data,train_labels)
        test_dataset = Traffic_Dataset(test_data,test_labels)
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

    with open('../data/dataset/GTSRB/train.pkl', 'rb') as f:
            train = pickle.load(f)
            train_x, train_y = train['data'], train['labels']
    with open('../data/dataset/GTSRB/test.pkl', 'rb') as f:
        test = pickle.load(f)
        test_x, test_y = test['data'], test['labels']

    n_class = len(np.unique(train_y))
    batch_size = 64
    epoch_num = 50
    start_epoch = 0

    model = GtsrbCNN(n_class)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    loss_func = nn.CrossEntropyLoss()

    save_path = "../data/model/resnet_gtsrb/model.tar"
    if os.path.isfile(save_path):
        checkpoint = torch.load(save_path)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"]

    training(train_data=train_x, train_labels=train_y, test_data=test_x, test_labels=test_y, 
             model=model, start_epoch=start_epoch,epoch_num=epoch_num,  batch_size=batch_size, loss_func=loss_func, 
             optimizer=optimizer, save_path=save_path)
