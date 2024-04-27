import cv2
import os

dir_path = "../data/adv_img/GTSRB/"
save_path = "../data/shadow_data/training/"
dir_list = ['43_random_attack/', '43_all_attack/', '43_default_attack/']
for i,d in enumerate(dir_list):
    img_dir = dir_path + d
    save_dir = save_path + d
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
    for f in os.listdir(img_dir):
        file_path = img_dir + f
        if "bmp" in file_path:
            label = int(f.split("_")[1])
            save_label_path = save_dir + str(label)  + "/"
            if os.path.isdir(save_label_path):
                img = cv2.imread(file_path)
                cv2.imwrite(save_label_path + f"{len(os.listdir(save_label_path))}.jpg",img)
            else:
                os.mkdir(save_label_path)
                print(save_label_path)
                img = cv2.imread(file_path)
                cv2.imwrite(save_label_path + f"{len(os.listdir(save_label_path))}.jpg",img)