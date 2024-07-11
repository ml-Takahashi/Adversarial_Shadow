from tqdm import tqdm
import torch

def train_model(model, dataloader, optimizer, device):
    model.train()
    for images, targets in tqdm(dataloader):
        images = [image.to(device) for image in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        optimizer.zero_grad()
        loss_dict = model(images, targets)
        print(loss_dict)
        exit()
        losses = sum(loss for loss in loss_dict.values())
        
        losses.backward()
        optimizer.step()

def test_model(model, dataloader, device="cpu"):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for images, targets in tqdm(dataloader):
            images = [image.to(device) for image in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            print(loss_dict)
            losses = sum(loss for loss in loss_dict.values())
            
            total_loss += losses.item()

    average_loss = total_loss / len(dataloader)
    print(f'Average test loss: {average_loss}')
    return average_loss

def get_area(box):
    height = box[0,3] - box[0,1]
    width = box[0,2] - box[0,0]
    return torch.tensor([height*width], dtype=torch.float32)


def collate_fn(batch):
    images, targets = zip(*batch)
    return images, list(targets)