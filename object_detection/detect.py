import torch
import torchvision
import torchvision.models.detection as detection
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.ops import box_iou
from dataset import CustomDataset
from tqdm import tqdm
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from func import *


if __name__ == "__main__":
    meta_train_path = "../../data/dataset/GTSRB/Train.csv"
    dataset_path = '../../data/dataset/GTSRB/'
    batch_size = 32
    transform = transforms.Compose([
        transforms.Resize((32,32)),  # 画像を32x32にリサイズ
        transforms.ToTensor(),          # 画像をテンソルに変換
        transforms.Normalize(mean=[0.3403, 0.3121, 0.3214], std=[0.1595, 0.1590, 0.1683])  # 正規化(訓練データのmeanとstd)
    ])
    num_class = 5
    num_epochs = 50

    # モデルのロード（pre-trainedのFast R-CNNモデル）
    model = detection.fasterrcnn_resnet50_fpn(pretrained=True)
    # モデルの出力層を変更
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_class)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

    train_dataset = CustomDataset(dataset_path, meta_train_path, transform)
    train_loader = DataLoader(train_dataset,batch_size=batch_size,shuffle=False,collate_fn=collate_fn)

    # image, {'boxes': box, 'labels': label, "image_id": idx, "area": area, "iscrowd": 0}
    # トレーニングループ
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        train_model(model, train_loader, optimizer)

        # モデルのチェックポイントを保存
        torch.save(model.state_dict(), f'model_epoch_{epoch+1}.pth')