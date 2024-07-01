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
import sys
import collections
import pickle
import matplotlib.pyplot as plt
import pandas as pd
from functions import MyDataset,GtsrbCNN

def test_loop(test_loader, model, loss_func):

    model.eval()
    test_acc = 0
    test_loss = 0
    test_num = 0
    with torch.no_grad():
        for x,y in tqdm(test_loader):
            test_pred = model(x)
            batch_loss = loss_func(test_pred,y)
            test_acc += (torch.argmax(test_pred, dim=1) == y).sum()
            test_loss += batch_loss.item() * len(y)
            test_num += len(y)
        
                
        print(f'Test Acc: {round(float(test_acc / test_num), 4)}', end=' ')
        print(f'Loss: {round(float(test_loss / test_num), 4)}')

    
if __name__ == "__main__":



    args = sys.argv
    if len(args) == 1:
        print("Enter shadow rate.")
        exit()
    N = int(float(args[1])*100)

    # データセットのパスを指定
    dataset_path = '../data/adv_img/tree/Test/'
    meta_test_path = f"../data/adv_img/tree/Test/Test_{N}.csv"
    save_path = "../data/model/normalized_model_gtsrb.pth"
    n_class = 43 #len(os.listdir(dataset_path+"Train"))
    batch_size = 64
    transform = transforms.Compose([
    transforms.Resize((32,32)),  # 画像を32x32にリサイズ
    transforms.ToTensor(),          # 画像をテンソルに変換
    transforms.Normalize(mean=[0.3403, 0.3121, 0.3214], std=[0.1595, 0.1590, 0.1683])  # 正規化(訓練データのmeanとstd)
])

    model = GtsrbCNN(n_class)
    #optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    loss_func = nn.CrossEntropyLoss()

    if os.path.isfile(save_path):
        checkpoint = torch.load(save_path)
        model.load_state_dict(checkpoint["model_state_dict"])
        #optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    test_dataset = MyDataset(dataset_path, meta_test_path,transform)
    test_loader = DataLoader(test_dataset,batch_size=batch_size,shuffle=True)
    
    test_loop(test_loader, model, loss_func)
    