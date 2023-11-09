# -*- coding: utf-8 -*-

import argparse
import json
import os
import pickle
from collections import Counter

import cv2
import numpy as np
import torch
import torchvision.datasets as dsets
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm

import gtsrb
import lisa
from gtsrb import GtsrbCNN
from lisa import LisaCNN
from pso import PSO
from utils import (brightness, draw_shadow, judge_mask_type, load_mask,
                   pre_process_image, shadow_edge_blur)


class MyDataset(Dataset):
    def __init__(self, dir_name, transform=transforms.Compose([transforms.ToTensor()])):
        self.file_list = [os.path.join(dir_name,f) for f in os.listdir(dir_name) if (os.path.isfile(os.path.join(dir_name, f))&("bmp" in f))]
        self.file_list.sort()
        self.transform = transform#transforms.Compose([torch.ToTensor()])

    # len()を使用すると呼ばれる
    def __len__(self):
        return len(self.file_list)

    # 要素を参照すると呼ばれる関数    
    def __getitem__(self, idx):
        img_path = self.file_list[idx]
        img = Image.open(img_path) #PIL形式で画像を読み込み
        img_transformed = self.transform(img).view(1,3,32,32) #画像の前処理を実施
        label = int(img_path.split("_")[4])
        return img_transformed,label,img_path

with open('params.json', 'rb') as f:
    params = json.load(f)
    class_n_gtsrb = params['GTSRB']['class_n']
    class_n_lisa = params['LISA']['class_n']
    device = params['device']
    position_list, mask_list = load_mask()

parser = argparse.ArgumentParser(description="Adversarial attack by shadow")
parser.add_argument("--shadow_level", type=float, default=0.43,
                    help="shadow coefficient k")
parser.add_argument("--attack_db", type=str, default="GTSRB",
                    help="the target dataset should be specified for a digital attack")
parser.add_argument("--attack_type", type=str, default="digital",
                    help="digital attack or physical attack")
parser.add_argument("--image_path", type=str, default="./xxx",
                    help="a file path to the target image should be specified for a physical attack")
parser.add_argument("--mask_path", type=str, default="./xxx",
                    help="a file path to the mask should be specified for a physical attack")
parser.add_argument("--image_label", type=int, default=0,
                    help="a ground truth should be specified for a physical attack")
parser.add_argument("--polygon", type=int, default=3,
                    help="shadow shape: n-sided polygon")
parser.add_argument("--n_try", type=int, default=5,
                    help="n-random-start strategy: retry n times")
parser.add_argument("--target_model", type=str, default="normal",
                    help="attack normal model or robust model")

args = parser.parse_args()
shadow_level = args.shadow_level-0.05
target_model = args.target_model
attack_db = args.attack_db
attack_type = args.attack_type
image_path = args.image_path
mask_path = args.mask_path
image_label = args.image_label
polygon = args.polygon
n_try = args.n_try


assert attack_db in ['LISA', 'GTSRB']
if attack_db == "LISA":
    model = LisaCNN(n_class=class_n_lisa).to(device)
    model.load_state_dict(
        torch.load(f'./model/{"adv_" if target_model == "robust" else ""}model_lisa.pth',
                   map_location=torch.device(device)))
    pre_process = transforms.Compose([transforms.ToTensor()])
else:
    model = GtsrbCNN(n_class=class_n_gtsrb).to(device)
    model.load_state_dict(
        torch.load(f'./model/{"adv_" if target_model == "robust" else ""}model_gtsrb.pth',
                   map_location=torch.device(device)))
    pre_process = transforms.Compose([
        pre_process_image, transforms.ToTensor()])
model.eval()


assert attack_type in ['digital', 'physical']
if attack_type == 'digital':
    particle_size = 10
    iter_num = 100
    x_min, x_max = -16, 48
    max_speed = 1.5
else:
    particle_size = 10
    iter_num = 200
    x_min, x_max = -112, 336
    max_speed = 10.
    n_try = 1


def attack(attack_image, label, coords, targeted_attack=False, physical_attack=False, **parameters):
    r"""
    Physical-world adversarial attack by shadow.

    Args:
        attack_image: The image to be attacked.
        label: The ground-truth label of attack_image.
        coords: The coordinates of the points where mask == 1.
        targeted_attack: Targeted / Non-targeted attack.
        physical_attack: Physical / digital attack.

    Returns:
        adv_img: The generated adversarial image.
        succeed: Whether the attack is successful.
        num_query: Number of queries.
    """
    num_query = 0
    succeed = False
    global_best_solution = float('inf')
    global_best_position = None

    for attempt in range(n_try):

        if succeed:
            break

        print(f"try {attempt + 1}:", end=" ")

        pso = PSO(polygon*2, particle_size, iter_num, x_min, x_max, max_speed,
                  shadow_level, attack_image, coords, model, targeted_attack,
                  physical_attack, label, pre_process, **parameters)
        best_solution, best_pos, succeed, query = pso.update_digital() \
            if not physical_attack else pso.update_physical()

        if targeted_attack:
            best_solution = 1 - best_solution
        print(f"Best solution: {best_solution} {'succeed' if succeed else 'failed'}")
        if best_solution < global_best_solution:
            global_best_solution = best_solution
            global_best_position = best_pos
        num_query += query

    adv_image, shadow_area = draw_shadow(
        global_best_position, attack_image, coords, shadow_level)
    adv_image = shadow_edge_blur(adv_image, shadow_area, 3)

    return adv_image, succeed, num_query


def attack_digital():

    img_dir = f'adv_img/{attack_db}/{int(shadow_level*100)}_all_shadow'
    #img_dir = "adv_img/GTSRB/43_random_attack"

    #files_file = [f for f in os.listdir(img_dir+"/0") if (os.path.isfile(os.path.join(img_dir, f))&("bmp" in f))]
    #print(len(files_file))
    #index_list = [f.split("_")[1] for f in files_file]
    batch_size = 1
    transform = transforms.Compose([transforms.ToTensor()])
    #datasets = ImageFolder(img_dir,transform=transform)
    #class2label = [str(i) for i in range(len(os.listdir(img_dir)))]
    #normal_data = image_folder_custom_label(root=img_dir, transform=transform, idx2label=class2label)
    #normal_loader = torch.utils.data.DataLoader(normal_data, batch_size=10, shuffle=False)
    datasets = MyDataset(img_dir)
    #data_loader = iter(normal_loader)
    #data_loader = DataLoader(datasets,batch_size=batch_size,shuffle=False)
    count = 0
    for images,label,img_path in tqdm(datasets):
        outputs = model(images)
        _,pre = torch.max(outputs.data,1)
        if pre==label:
            count += 1
            new_name = img_path.split(".bmp")[0]+"True.bmp"
            os.rename(img_path,new_name)
        else:
            new_name = img_path.split(".bmp")[0]+"False.bmp"
            os.rename(img_path,new_name)

            
    print(count/len(datasets))

    """mask_type = judge_mask_type(attack_db, labels[index])
    if brightness(images[index], mask_list[mask_type]) >= 120:
        adv_img, success, num_query = attack(
            images[index], labels[index], position_list[mask_type])
        cv2.imwrite(f"{save_dir}/{index}_{labels[index]}_{num_query}_{success}.bmp", adv_img)"""

    #print("Attack finished! Success rate: ", end='')


def attack_physical():

    global position_list

    mask_image = cv2.resize(
        cv2.imread(mask_path, cv2.IMREAD_UNCHANGED), (224, 224))
    target_image = cv2.resize(
        cv2.imread(image_path), (224, 224))
    pos_list = np.where(mask_image.sum(axis=2) > 0)

    # EOT is included in the first stage
    adv_img, _, _ = attack(target_image, image_label, pos_list,
                           physical_attack=True, transform_num=10)
    
    cv2.imwrite('./tmp/temp.bmp', adv_img)
    if attack_db == 'LISA':
        predict, failed = lisa.test_single_image(
            './tmp/temp.bmp', image_label, target_model == "robust")
    else:
        predict, failed = gtsrb.test_single_image(
            './tmp/temp.bmp', image_label, target_model == "robust")
    if failed:
        print('Attack failed! Try to run again.')

    # Predict stabilization
    adv_img, _, _ = attack(target_image, image_label, pos_list, targeted_attack=False,
                           physical_attack=True, target=predict, transform_num=10)

    cv2.imwrite('./tmp/adv_img.png', adv_img)
    if attack_db == 'LISA':
        predict, failed = lisa.test_single_image(
            './tmp/adv_img.png', image_label, target_model == "robust")
    else:
        predict, failed = gtsrb.test_single_image(
            './tmp/adv_img.png', image_label, target_model == "robust")
    if failed:
        print('Attack failed! Try to run again.')
    else:
        print('Attack succeed! Try to implement it in the real world.')

    cv2.imshow("Adversarial image", adv_img)
    cv2.waitKey(0)


if __name__ == '__main__':
    print(model)
    exit()
    attack_digital() if attack_type == 'digital' else attack_physical()
    

