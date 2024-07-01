import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader
import os
from functions import GtsrbCNN,MyDataset
from tqdm import tqdm


# データの前処理
transform = transforms.Compose([
    transforms.Resize((32,32)),  # 画像を32x32にリサイズ
    transforms.ToTensor(),          # 画像をテンソルに変換
    transforms.Normalize(mean=[0.3403, 0.3121, 0.3214], std=[0.1595, 0.1590, 0.1683])  # 正規化(訓練データのmeanとstd)
])

# データセットのパスを指定
dataset_path = '../data/dataset/GTSRB/'
meta_test_path = "../data/dataset/GTSRB/Test.csv"
save_path = "../data/model/normalized_model_gtsrb.pth" #pthで合ってる？
n_class = 43
batch_size = 1
epoch_num = 100
start_epoch = 0

test_dataset = MyDataset(dataset_path, meta_test_path,transform)
test_loader = DataLoader(test_dataset,batch_size=batch_size,shuffle=False)

model = GtsrbCNN(n_class)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
loss_func = nn.CrossEntropyLoss()
if os.path.isfile(save_path):
    checkpoint = torch.load(save_path)
    model.load_state_dict(checkpoint["model_state_dict"])


model.eval()
with torch.no_grad():
    for x,y in tqdm(test_loader):
        label = int(y)
        print(label)
        test_pred = model(x)
        print(torch.argmax(test_pred, dim=1))