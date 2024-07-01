from PIL import Image
import numpy as np
import os
import re

def create_mask_image(image_path,shadow_num):
    # 画像を読み込み、RGB形式に変換
    image = Image.open(image_path).convert('RGB')
    image = image.resize((32,32))
    
    # 画像をNumPy配列に変換
    image_array = np.array(image)
    
    # 白以外のピクセルを見つける（白はRGBが(255, 255, 255)）
    mask = (image_array[:, :, 0] != 255) | (image_array[:, :, 1] != 255) | (image_array[:, :, 2] != 255)
    
    # 新しい画像用の配列を作成（初期値は黒（0, 0, 0））
    new_image_array = np.zeros_like(image_array)
    
    # マスクに基づいて特定のピクセルを白（255, 255, 255）に設定
    new_image_array[mask] = [255, 255, 255]
    
    # NumPy配列からPIL画像に変換
    new_image = Image.fromarray(new_image_array)
    
    # 新しい画像を表示
    #new_image.show()
    #shadow_num = int(image_path.split(".")[2][-1])
    
    # 新しい画像を保存（必要に応じて）
    new_image.save(f'../data/tree/ichou_shadow{shadow_num}.png')

# 使用例
tree_image_dir = "../data/tree/"
dir_list = os.listdir(tree_image_dir)
pattern = r'ichou(\d+)'
for f in dir_list:
    match = re.search(pattern, f)
    if match:
        shadow_num = int(match.group(1))
        create_mask_image(tree_image_dir+f,shadow_num)
