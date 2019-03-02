# -*- coding: utf-8 -*-
import cv2
import numpy as np
import re
import math
import matplotlib.pyplot as plt
path='../newdatares/1117_0706000506201)(10,34 9,64 132,37 130,67 )[10,34 9,64 132,37 130,67 ].jpg';
img = cv2.imread(path)
p1=re.compile(r'[[](.*?)[]]',re.S)
number=(re.findall(p1,path)[0]).split()# 原图中书本的四个角点
numarr=[]# 变换后分别在左上、右上、左下、右下四个点
for item in number:
    numarr.extend(item.split(','))# 生成透视变换矩阵
numarr = [int(val) for val in numarr]
print numarr

h = int(round(math.sqrt((numarr[0]-numarr[2])*(numarr[0]-numarr[2]) + (numarr[1]-numarr[3])*(numarr[1]-numarr[3]))))
h = max(h,int(round(math.sqrt((numarr[4]-numarr[6])*(numarr[4]-numarr[6]) + (numarr[5]-numarr[7])*(numarr[5]-numarr[7])))))

w = int(round(math.sqrt((numarr[0]-numarr[4])*(numarr[0]-numarr[4]) + (numarr[1]-numarr[5])*(numarr[1]-numarr[5]))))
w = max(w,int(round(math.sqrt((numarr[2]-numarr[6])*(numarr[2]-numarr[6]) + (numarr[3]-numarr[7])*(numarr[3]-numarr[7])))))

print h,w
pts1 = np.float32([[numarr[0],numarr[1]], [numarr[4],numarr[5]], [numarr[2],numarr[3]], [numarr[6],numarr[7]]])
pts2 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
M = cv2.getPerspectiveTransform(pts1, pts2)
# 进行透视变换
dst = cv2.warpPerspective(img, M, (w, h))
plt.subplot(121), plt.imshow(img[:, :, ::-1]), plt.title('input')
plt.subplot(122), plt.imshow(dst[:, :, ::-1]), plt.title('output')
# img[:, :, ::-1]是将BGR转化为RGB
plt.show()