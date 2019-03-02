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
import math
from PIL import Image
class D_DataProvider(object):
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
            # print self.images_name[self.file_idx]
            # print imageoutput
            input_images.extend(imagedata)
            output_number.extend(imageoutput)
        # print input_images
        # print np.shape(input_images)
        # print np.shape(output_number)
        return input_images,output_number

    def _getImagedata(self,path):
        imagedata=[]
        aa=np.array(Image.open(self.images_name[self.file_idx]).convert('L').resize((120,36)))
        aa=np.reshape(aa,(36,120,1))
        # print np.shape(aa)
        arr1,arr2,arr3,arr4,arr5=np.split(aa,[24,48,72,96],axis=1)
        # Image.fromarray(arr1.astype('uint8')).convert('L').show()
        # Image.fromarray(arr2.astype('uint8')).convert('L').show()
        # Image.fromarray(arr3.astype('uint8')).convert('L').show()
        # Image.fromarray(arr4.astype('uint8')).convert('L').show()
        # arr5=np.squeeze(arr5,-1)
        # Image.fromarray(arr5.astype('uint8')).convert('L').show()
        imagedata.append(arr5)
        imagedata.append(arr4)
        imagedata.append(arr3)
        imagedata.append(arr2)
        imagedata.append(arr1)
        return imagedata

    def _getMapping(self,path):
        p1=re.compile(r'[(](.*?)[)]',re.S)
        number=(float)(re.findall(p1,path)[0])
        numarr=np.zeros((5,10),dtype=float)
        # numarr=[]
        for i in range(0,5):
            if i==0:
                decimal=number%1
                a1=number%10
                a11=int(math.ceil(a1))%10
                a12 = int(math.floor(a1))
                numarr[i][a11]=decimal
                numarr[i][a12]=1-decimal
            else:
                a1=number%10
                a12 = int(math.floor(a1))
                numarr[i][a12]=1
            number=number/10;
        # print re.findall(p1,path)[0]
        # print numarr
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
# print a.images_name
# a.next_batch()