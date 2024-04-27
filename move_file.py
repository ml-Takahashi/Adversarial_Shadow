import os
import shutil

attack_db = "GTSRB"
shadow_level = 0.43
img_dir = f'adv_img/{attack_db}/{int(shadow_level*100)}_default_attack'

files_file = [f for f in os.listdir(img_dir) if (os.path.isfile(os.path.join(img_dir, f))&("bmp" in f))]

# ラベルごとのディレクトリに分ける場合
for f in files_file:
    label = f.split("_")[1]
    if not os.path.exists(img_dir+"/"+str(f.split("_")[1])):
        # ディレクトリが存在しない場合、ディレクトリを作成する
        os.makedirs(img_dir+"/"+str(f.split("_")[1]))
    #shutil.move(f,f"{img_dir}/{f.split("_")[1]}/{f}" )
    shutil.move(img_dir+"/"+f,img_dir+"/"+str(f.split("_")[1]))

# 一つのディレクトリにまとめる場合
for i in os.listdir(img_dir):
    if os.path.isdir(os.path.join(img_dir,str(i))):
        for f in os.listdir(os.path.join(img_dir,str(i))):
            dir_name = os.path.join(img_dir,str(i))
            if os.path.isfile(os.path.join(dir_name,f)):
                shutil.move(os.path.join(dir_name,f),os.path.join(img_dir))
        os.rmdir(os.path.join(img_dir,str(i)))
    else:
        os.remove(os.path.join(img_dir,str(i)))
print(os.path.join(img_dir,f))
