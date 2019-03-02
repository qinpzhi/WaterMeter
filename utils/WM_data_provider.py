# -*- coding: utf-8 -*-
'''
@Time    : 18-10-17 下午2:40
@Author  : qinpengzhi
@File    : data_provider.py
@Software: PyCharm
@Contact : qinpzhi@163.com
'''
import glob
import os
import re
import numpy as np
import tensorflow as tf
import math
from utils.WM_config import WM_data_cfg
from PIL import Image
import cv2
class WM_DataProvider(object):
    def __init__(self,config):
        self.cfg=config
        self.file_idx=-1
        self.images_name=self._find_imageNames(config)
        perm = np.arange(len(self.images_name))
        np.random.shuffle(perm)
        self.images_name = self.images_name[perm]

    def _find_imageNames(self,config):
        images_name=np.array(glob.glob(os.path.join(config.train_path,"*.jpg")))
        return images_name

    def next_batch(self,batch_size=10):
        input_images=[]
        output_number=[]
        for item in range(0, batch_size):
            self._cycle_files()
            imagedata=self._getImagedata(self.images_name[self.file_idx])
            imageoutput=self._getMapping(self.images_name[self.file_idx])
            input_images.append(imagedata)
            output_number.append(imageoutput)
        input_images=np.array(input_images)
        output_number=np.array(output_number)
        # print input_images.shape,output_number.shape
        return input_images,output_number

    def _getImagedata(self,path):
        # img = np.array(Image.open(path).convert('L').resize((160, 88)))
        # img = np.reshape(img, (88, 160, 1))
        img = cv2.imread(path, 0)
        img=np.array(img)
        img=img[0:88,0:160]
        img=img[:, :, np.newaxis]
        # print img.shape
        return img

    def _getMapping(self,path):
        p1=re.compile(r'[[](.*?)[]]',re.S)
        number=(re.findall(p1,path)[0]).split()
        numarr=[]
        for item in number:
            numarr.extend(item.split(','))
        numarr = [int(val) for val in numarr]
        return numarr

    def _cycle_files(self):
        self.file_idx=self.file_idx+1
        if self.file_idx>=len(self.images_name):
            ##打乱顺序
            perm=np.arange(len(self.images_name))
            np.random.shuffle(perm)
            self.images_name=self.images_name[perm]
            self.file_idx = 0
# #
# a=DataProvider(data_cfg)
# a.next_batch()