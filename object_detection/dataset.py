from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
from func import *
import torch

class CustomDataset(Dataset):
    def __init__(self, dataset_path, meta_data_path, transform):
        self.dataset_path = dataset_path
        self.df = pd.read_csv(meta_data_path)
        self.transform = transform
        self.label_dict = {"circle": 0, "triangle": 1, "r_triangle": 2, "rectangle":3, "octagon":4}
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        img_path = self.dataset_path + self.df["Path"][idx]
        image = Image.open(img_path)
        image = self.transform(image)
        label = torch.tensor([self.label_dict[self.df["shape"][idx]]],dtype=torch.int64) #self.df["ClassId"][idx]
        box = torch.tensor([self.df["new_x1"][idx], self.df["new_y1"][idx], self.df["new_x2"][idx], self.df["new_y2"][idx]], dtype=torch.float32).reshape(-1,4)
        area = get_area(box)
        return image, {'boxes': box, 'labels': label, "image_id": torch.tensor([idx], dtype=torch.int64), "area": area, "iscrowd": torch.tensor([0], dtype=torch.int64)}
