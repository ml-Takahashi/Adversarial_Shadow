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
from functions import MyDataset


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

def count_file_num(dir_path,label_num):
    dir_list = os.listdir(dir_path)
    for i,d in enumerate(dir_list):
        dir_list[i] = dir_path + d
    dir_list = sorted(dir_list)

    label_dict = {}
    
    for n in range(label_num):
        label_dict[int(n)] = 0

    for label in dir_list:
        label_num = len(os.listdir(label))
        label = label.split("/")[-1]
        label_dict[int(label)] = label_num
    
    return label_dict

def make_bar_image(dict_path,save_path,title=None):
    if os.path.isfile(dict_path):
        with open(dict_path, "rb") as tf:
            dic = pickle.load(tf)
            fig,ax = plt.subplots()
            x = np.arange(0,43)
            ax.set_xticks(x)
            plt.tick_params(labelsize=8)
            ax.bar(dic.keys(),dic.values(),width=0.8)
            plt.xticks(rotation=90)
            if title is not None:
                plt.title(title)
            plt.savefig(save_path)

def test_loop(test_file_list, model, batch_size, n_class):

    test_acc, test_loss = 0, 0
    epoch_start_time = time.time()
    test_dataset = My_Dataset(test_file_list)
    test_num = test_dataset.__len__()
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    model.eval()
    false_list = []
    with torch.no_grad():
        for x,y in tqdm(test_loader):
            test_pred = model(x)
            batch_loss = loss_func(test_pred,y)
            # 誤ったクラスをリストにまとめる
            bool_list = torch.argmax(test_pred, dim=1) != y
            batch_false = list(y[bool_list])#list(torch.argmax(test_pred, dim=1)[bool_list])
            batch_false = [int(n) for n in batch_false]
            false_list.extend(batch_false)

            test_acc += (torch.argmax(test_pred, dim=1) == y).sum()
            test_loss += batch_loss.item() * len(y)
            
        # 誤分類された数をカウントした辞書
        my_dict = dict(sorted(collections.Counter(false_list).items()))
       
        # キーが存在しないクラス(誤分類されなかったクラス)のキーを設定
        for n in range(n_class):
            if n not in my_dict.keys():
                my_dict[n] = 0
        
        # 昇順でソート
        my_dict = sorted(my_dict.items())
        my_dict = dict((x,y) for x,y in my_dict)

        # クラスごとの誤分類率を保持した辞書を作成
        percent_dict = {}
        for key in my_dict.keys():
            if label_dict[key] != 0:
                percent_dict[key] = my_dict[key]/label_dict[key]
        
        # 誤分類された数をカウントした辞書を記録
        with open("count.pkl", "wb") as tf:
            pickle.dump(my_dict, tf)

        # クラスごとの誤分類率を保持した辞書を保存
        with open("percent.pkl", "wb") as tf:
            pickle.dump(percent_dict, tf)
        make_bar_image("percent.pkl","percent_default_attack_bar.png",title="Misclassification percentage by default attack")
                
        print(my_dict)
        print(percent_dict)
        print(f'{round(time.time() - epoch_start_time, 2)}s', end=' ')
        print(f'Test Acc: {round(float(test_acc / test_num), 4)}', end=' ')
        print(f'Loss: {round(float(test_loss / test_num), 4)}')

    
if __name__ == "__main__":

    # 各クラスごとの誤分類数を棒グラフで保存
    make_bar_image("count.pkl","count_default_attack_bar.png",title="Misclassification count by default attack")
    

    args = sys.argv
    if len(args) > 1:
        test_dir = args[1]
    else:
        print("Enter file name.")
        exit()
    save_path = "../data/model/random_shadow_trained_model/model.tar"
    test_dir = f"../data/shadow_data/training/{test_dir}/"
    test_file_list = make_file_list(test_dir)

    n_class = 43
    batch_size = 64

    label_dict = count_file_num(test_dir,n_class)

    model = GtsrbCNN(n_class)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    loss_func = nn.CrossEntropyLoss()

    if os.path.isfile(save_path):
        checkpoint = torch.load(save_path)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    
    #test_loop(test_file_list=test_file_list, model=model, batch_size=batch_size,n_class=n_class)
    
