from tqdm import tqdm
import torch

def train_model(model, dataloader, optimizer):
    model.train()
    for images, targets in tqdm(dataloader):
        images = [image for image in images]
        targets = [{k: v for k, v in t.items()} for t in targets]
        
        optimizer.zero_grad()
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        
        losses.backward()
        optimizer.step()

def get_area(box):
    height = box[0,3] - box[0,1]
    width = box[0,2] - box[0,0]
    return torch.tensor([height*width], dtype=torch.float32)


def collate_fn(batch):
    images, targets = zip(*batch)
    return images, list(targets)