import pickle
import cv2 as cv
import random
import numpy as np
import os
import sys

# 影の開始ポイント
shadow_point = ["top","right","bottom","left"]
if len(sys.argv) > 1:
    N = sys.argv[1]
else:
    print("影の比率を入力してください")
    exit()


def get_random_point(image):
    random_side = random.sample(shadow_point,2)
    # bottomとtopの組み合わせになった場合、leftかrightからひとつ選び直す
    if ("bottom" in random_side)and("top" in random_side):
        random_side[1] = random.sample(["left","right"],1)[0]
    height,width = image.shape[0],image.shape[1]
    random_point = []
    # random_pointは端の5ピクセルを選ばないようにする。影の意味がなくなるため。
    if "top" in random_side:
        position = random.randint(0,width-1)
        random_point.append([position,0])
    if "bottom" in random_side:
        position = random.randint(0,width-1)
        random_point.append([position,height-1])
    if "left" in random_side:
        position = random.randint(0,height-1)
        random_point.append([0,position])
    if "right" in random_side:
        position = random.randint(0,height-1)
        random_point.append([width-1,position])
    if random_point[0][0] > random_point[1][0]:
        random_point.reverse()
    return random_point,random_side

def choose_point(image,random_point,random_side):
    draw_corner = random_point
    height,width = image.shape[0],image.shape[1]
    if ("left" in random_side)and("bottom" in random_side):
        draw_corner.append([0,height-1])
    elif ("left" in random_side)and("right" in random_side):
        draw_corner.append([width-1,height-1])
        draw_corner.append([0,height-1])
    elif ("left" in random_side)and("top" in random_side):
        draw_corner.append([width-1,0])
        draw_corner.append([width-1,height-1])
        draw_corner.append([0,height-1])
    elif ("right" in random_side)and("bottom" in random_side):
        draw_corner.append([width-1,height-1])
    elif ("right" in random_side)and("top" in random_side):
        draw_corner.append([width-1,height-1])
        draw_corner.append([0,height-1])
        draw_corner.append([0,0])
    return draw_corner

def draw_shadow(image,draw_point):
    draw_point = [[[x,y] for x,y in draw_point]]
    s_img = cv.cvtColor(image, cv.COLOR_BGR2Lab)
    line_mask = np.zeros_like(image)
    cv.fillPoly(line_mask, np.array(draw_point), (255, 255, 255)) # draw_pointで囲われている点を白で塗りつぶす
    mask = np.column_stack((np.all(line_mask == [255, 255, 255], axis=-1))) # リストになっている白の点を縦に並べる(座標をまとめたndarray)
    s_img[mask,0] = s_img[mask,0] * N # 該当する点に影の比率をかける
    s_img = cv.cvtColor(s_img,cv.COLOR_Lab2BGR)
    return s_img


def make_shadow_image(image):
    random_side = ["bottom","top"]
    random_point,random_side = get_random_point(image)
    draw_point = choose_point(image,random_point,random_side)
    draw_image = draw_shadow(image,draw_point)
    return draw_image
    
if __name__=="__main__":
    with open('../data/dataset/GTSRB/train.pkl', 'rb') as f:
        train = pickle.load(f)
        train_x, train_y = train['data'], train['labels']
    with open('../data/dataset/GTSRB/test.pkl', 'rb') as f:
        test = pickle.load(f)
        test_x, test_y = test['data'], test['labels']

    save_train_dir = "../data/shadow_data/training"
    save_test_dir = "../data/shadow_data/test"
    count = 0
    for x,y in zip(train_x,train_y):
        label_dir = f"{save_train_dir}/{y}"
        if not os.path.exists(label_dir):
            os.makedirs(label_dir)
            count = 0
        shadow_image = make_shadow_image(x)
        cv.imwrite(f"{label_dir}/{count}.png",shadow_image)
        count += 1
        
    for x,y in zip(test_x,test_y):
        label_dir = f"{save_test_dir}/{y}"
        if not os.path.exists(label_dir):
            os.makedirs(label_dir)
            count = 0
        shadow_image = make_shadow_image(x)
        cv.imwrite(f"{label_dir}/{count}.png",shadow_image)
        count += 1