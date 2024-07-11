from func import *
from torchvision.models.detection import ssdlite320_mobilenet_v3_large
from dataset import CustomDataset
from torchvision import transforms
from torch.utils.data import DataLoader
import pandas as pd


dataset_path = '../../data/dataset/GTSRB/'
meta_test_path = "../../data/dataset/GTSRB/Test.csv"
batch_size = 2

# データの前処理
transform = transforms.Compose([
    transforms.Resize((32,32)),  # 画像を32x32にリサイズ
    transforms.ToTensor(),          # 画像をテンソルに変換
    transforms.Normalize(mean=[0.3403, 0.3121, 0.3214], std=[0.1595, 0.1590, 0.1683])  # 正規化(訓練データのmeanとstd)
])

model = ssdlite320_mobilenet_v3_large(pretrained=True)
model.head.classification_head.num_classes = 5#num_class
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

test_dataset = CustomDataset(dataset_path, meta_test_path,transform)
test_loader = DataLoader(test_dataset,batch_size=batch_size,shuffle=True,collate_fn=collate_fn)

min_loss = 1000000
best_epoch = -1
loss_df = pd.DataFrame(columns=["epoch","loss"])
for i in range(20):
    i = 19
    save_path = f"model_epoch_{i+1}.pth"
    model.load_state_dict(torch.load(save_path))
    loss = test_model(model, test_loader)
    if loss < min_loss:
        min_loss = loss
        print("Best Model!")
    loss_df.loc[i] = [i, loss]

loss_df.to_csv("model_loss.csv",index=False)
    
    #optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    #start_epoch = checkpoint["epoch"]