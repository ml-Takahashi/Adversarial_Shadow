import cv2
import numpy as np

im = cv2.imread('/home/kanato/research/adv_shadow/data/adv_img/GTSRB/43_default_attack/2_0_35_TrueFalseFalse.bmp')
print(im.shape)
# (225, 400, 3)

print(im.dtype)
# uint8

im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
print(im_gray.shape)
# (225, 400)

print(im_gray.dtype)
# uint8

cv2.imwrite('opencv_gray_cvtcolr.jpg', im_gray)