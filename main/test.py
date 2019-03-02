# -*- coding: utf-8 -*-
'''
@Time    : 18-10-22 下午7:47
@Author  : qinpengzhi
@File    : test.py
@Software: PyCharm
@Contact : qinpzhi@163.com
向外提供接口文件
'''
import tensorflow as tf
import sys
reload(sys)
sys.setdefaultencoding('utf8')
from main.wm_model import WM_Model
from utils.WM_config import WM_model_cfg
from main.d_model import D_Model
from utils.D_config import D_model_cfg
import glob
import requests as req
from PIL import Image
import cv2
import math
import os
import matplotlib.pyplot as plt
from io import BytesIO
import numpy as np
import flask
from flask import jsonify
from flask import request


#注释掉的函数，可以释放注释，运行此文件夹将灰度图中的包含数字区域切割出来再保存到一个文件夹下。
# if __name__=='__main__':
#     sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
#     test_model=WM_Model(WM_model_cfg)
#     test_model.load(sess)
#     ##predict all the picture
#     im_path = '/home/qpz/projects/WaterMeter_new/111/*.jpg'
#     save_path='/home/qpz/projects/WaterMeter_new/222/'
#     # im_path = '/home/qpz/data/水表图片/灰度图片/1129_0706040507(54.53).jpg'
#     im_files = glob.glob(im_path)
#     cnt=150
#     for im_file in im_files:
#         print ('start' +im_file)
#         filename=im_file.split('/')[-1]
#         imagedata = []
#         img = cv2.imread(im_file, 0)
#         img = np.array(img)
#         img = img[0:88, 0:160]
#         img = img[:, :, np.newaxis]
#         img = np.array(img)
#         imagedata.append(img)
#
#         pred = sess.run(test_model.output, feed_dict={test_model.data: imagedata,
#                                                     test_model.keep_prob: 1.0})
#         print pred
#         numarr=pred[0]
#         h = int(round(math.sqrt(
#             (numarr[0] - numarr[2]) * (numarr[0] - numarr[2]) + (numarr[1] - numarr[3]) * (numarr[1] - numarr[3]))))
#         h = max(h, int(round(math.sqrt(
#             (numarr[4] - numarr[6]) * (numarr[4] - numarr[6]) + (numarr[5] - numarr[7]) * (numarr[5] - numarr[7])))))
#
#         w = int(round(math.sqrt(
#             (numarr[0] - numarr[4]) * (numarr[0] - numarr[4]) + (numarr[1] - numarr[5]) * (numarr[1] - numarr[5]))))
#         w = max(w, int(round(math.sqrt(
#             (numarr[2] - numarr[6]) * (numarr[2] - numarr[6]) + (numarr[3] - numarr[7]) * (numarr[3] - numarr[7])))))
#
#         print h, w
#         pts1 = np.float32(
#             [[numarr[0], numarr[1]], [numarr[4], numarr[5]], [numarr[2], numarr[3]], [numarr[6], numarr[7]]])
#         pts2 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
#         M = cv2.getPerspectiveTransform(pts1, pts2)
#         # 进行透视变换
#         imsrc=cv2.imread(im_file)
#         dst = cv2.warpPerspective(imsrc, M, (w, h))
#         print save_path,str(filename)
#         plt.imsave(save_path+str(filename), dst)
#         cnt=cnt+1
#         plt.subplot(121), plt.imshow(imsrc[:, :, ::-1]), plt.title('input')
#         plt.subplot(122), plt.imshow(dst[:, :, ::-1]), plt.title('output')
#         img[:, :, ::-1]#是将BGR转化为RGB
#         plt.show()

def get_NumberArea(im_file):
    tf.reset_default_graph()
    sess1 = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
    test_model = WM_Model(WM_model_cfg)
    test_model.load(sess1)
    try:
        imagedata = []
        response = req.get(im_file)
        imsrc = np.array(Image.open(BytesIO(response.content)))
        img = np.array(Image.open(BytesIO(response.content)).convert('L'))
        # img = cv2.imread(im_file, 0)
        # img = np.array(img)
        img = img[0:88, 0:160]
        img = img[:, :, np.newaxis]
        img = np.array(img)
        imagedata.append(img)

        pred = sess1.run(test_model.output, feed_dict={test_model.data: imagedata,
                                                      test_model.keep_prob: 1.0})

        numarr = pred[0]
        h = int(round(math.sqrt(
            (numarr[0] - numarr[2]) * (numarr[0] - numarr[2]) + (numarr[1] - numarr[3]) * (numarr[1] - numarr[3]))))
        h = max(h, int(round(math.sqrt(
            (numarr[4] - numarr[6]) * (numarr[4] - numarr[6]) + (numarr[5] - numarr[7]) * (numarr[5] - numarr[7])))))

        w = int(round(math.sqrt(
            (numarr[0] - numarr[4]) * (numarr[0] - numarr[4]) + (numarr[1] - numarr[5]) * (numarr[1] - numarr[5]))))
        w = max(w, int(round(math.sqrt(
            (numarr[2] - numarr[6]) * (numarr[2] - numarr[6]) + (numarr[3] - numarr[7]) * (numarr[3] - numarr[7])))))

        pts1 = np.float32(
            [[numarr[0], numarr[1]], [numarr[4], numarr[5]], [numarr[2], numarr[3]], [numarr[6], numarr[7]]])
        pts2 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
        M = cv2.getPerspectiveTransform(pts1, pts2)
        # 进行透视变换
        dst = cv2.warpPerspective(imsrc, M, (w, h))
    except:
        # test_model.unload(sess)
        sess1.close()
    # test_model.unload(sess)
    sess1.close()
    return dst



def getPredict_Number(dst):
    tf.reset_default_graph()
    sess2 = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
    test_model=D_Model(D_model_cfg)
    test_model.load(sess2)
    try:
        ##predict the picture
        # response = req.get(im_file)
        imagedata = []
        img=Image.fromarray(dst)
        aa = np.array(img.convert('L').resize((120, 36)))
        aa = np.reshape(aa, (36, 120, 1))
        # print np.shape(aa)
        arr1, arr2, arr3, arr4, arr5 = np.split(aa, [24, 48, 72, 96], axis=1)
        imagedata.append(arr5)
        imagedata.append(arr4)
        imagedata.append(arr3)
        imagedata.append(arr2)
        imagedata.append(arr1)

        pred = sess2.run(test_model.output, feed_dict={test_model.data: imagedata,
                                                    test_model.keep_prob: 1.0})

        pred = tf.squeeze(pred)
        pred=pred.eval(session=sess2)
        paixu=np.argsort(pred)

        predictval=paixu[4][9]
        predictval=predictval*10+paixu[3][9]
        predictval=predictval*10+paixu[2][9]
        # print predictval
        if paixu[0][9]==0 and paixu[0][8]==9 and paixu[1][9]!=0:
            predictval=predictval*10+min(paixu[1][9],paixu[1][8])
            predictval=predictval*10+9+pred[0][0]/(pred[0][9]+pred[0][0])
        elif abs(paixu[0][9]-paixu[0][8])==1:
            predictval = predictval * 10 + paixu[1][9]
            predictval = predictval * 10 +min(paixu[0][9],paixu[0][8])+pred[0][max(paixu[0][9],paixu[0][8])]/(pred[0][paixu[0][9]]+pred[0][paixu[0][8]])
        else:
            predictval = predictval * 10 + paixu[1][9]
            predictval = predictval * 10 + paixu[0][9]
    except:
        # test_model.unload(sess)
        sess2.close()
    # test_model.unload(sess)
    sess2.close()
    return predictval

def getPredict_Number1(im_file):
    tf.reset_default_graph()
    sess3 = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
    test_model=D_Model(D_model_cfg)
    test_model.load(sess3)
    try:
        ##predict the picture
        response = req.get(im_file)
        imagedata = []
        aa = np.array(Image.open(BytesIO(response.content)).convert('L').resize((120, 36)))
        aa = np.reshape(aa, (36, 120, 1))
        arr1, arr2, arr3, arr4, arr5 = np.split(aa, [24, 48, 72, 96], axis=1)
        imagedata.append(arr5)
        imagedata.append(arr4)
        imagedata.append(arr3)
        imagedata.append(arr2)
        imagedata.append(arr1)

        pred = sess3.run(test_model.output, feed_dict={test_model.data: imagedata,
                                                    test_model.keep_prob: 1.0})

        pred = tf.squeeze(pred)
        pred=pred.eval(session=sess3)
        paixu=np.argsort(pred)

        # print pred
        # print paixu
        predictval=paixu[4][9]
        predictval=predictval*10+paixu[3][9]
        predictval=predictval*10+paixu[2][9]
        # print predictval
        if paixu[0][9]==0 and paixu[0][8]==9 and paixu[1][9]!=0:
            predictval=predictval*10+min(paixu[1][9],paixu[1][8])
            predictval=predictval*10+9+pred[0][0]/(pred[0][9]+pred[0][0])
        elif abs(paixu[0][9]-paixu[0][8])==1:
            predictval = predictval * 10 + paixu[1][9]
            predictval = predictval * 10 +min(paixu[0][9],paixu[0][8])+pred[0][max(paixu[0][9],paixu[0][8])]/(pred[0][paixu[0][9]]+pred[0][paixu[0][8]])
        else:
            predictval = predictval * 10 + paixu[1][9]
            predictval = predictval * 10 + paixu[0][9]
    except:
        # test_model.unload(sess)
        sess3.close()
    # test_model.unload(sess)
    sess3.close()
    return predictval

#
# '''
# flask： web框架，可以通过flask提供的装饰器@server.route()将普通函数转换为服务
# 登录接口，需要传url、username、passwd
# '''
# # 创建一个服务，把当前这个python文件当做一个服务
server = flask.Flask(__name__)


# server.config['JSON_AS_ASCII'] = False
#
# @server.route()可以将普通函数转变为服务 登录接口的路径、请求方式
@server.route('/predict', methods=['get'])
def login():
    im_file = request.values.get('im_file')
    type=request.values.get('type')
    ##如果是二值化的图
    if type=="1":
        if im_file:
            img=get_NumberArea(im_file)
            predictval=getPredict_Number(img)
            res = {'code': 200, 'predictval': predictval}
            return jsonify(res)
        else:
            res = {'code': 999, 'message': '必填参数未填写'}
            return jsonify(res)
    elif type=="0":
        if im_file:
            predictval=getPredict_Number1(im_file)
            res = {'code': 200, 'predictval': predictval}
            return jsonify(res)
        else:
            res = {'code': 999, 'message': '必填参数未填写'}
            return jsonify(res)

if __name__ == '__main__':
    server.run(debug=True, port=8080, host='0.0.0.0')

