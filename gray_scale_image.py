import cv2
import numpy as np

img = cv2.imread('receipt.jpeg')

#img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

"""kernel = np.ones((5,5),np.float32)/25
dst = cv2.filter2D(img,-1,kernel)"""
img = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,21,128)

#th, im_gray_th_otsu = cv2.threshold(img, 128, 255, cv2.THRESH_OTSU)
#dst = img - dst

cv2.imwrite('opencv_th_otsu.jpg', img)