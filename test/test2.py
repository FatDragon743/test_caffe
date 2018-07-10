# -*- coding:utf-8 -*-  
'''
Created on 2018年7月9日

@author: Administrator
'''
# -*- coding:utf-8 -*-  
'''
Created on 2018年7月9日

@author: Administrator
'''
import sys
import numpy as np
import os
from tools.ConvertMean import get_mean
caffe_root = 'E:/caffe27/'  #该文件要从路径{caffe_root}/examples下运行，否则要调整这一行。
sys.path.insert(0, caffe_root + 'python')
import caffe
path_root = '../data/tmp/'
model_def = path_root + 'train.prototxt'
model_weights = path_root + 'mymodel.caffemodel'

net = caffe.Net(model_def,      # 定义模型结构
                model_weights,  # 包含了模型的训练权值
                caffe.TEST)     # 使用测试模式(不执行dropout

print net.blobs['Data1'].data.shape
# 对输入数据进行变换
# transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
# transformer.set_transpose('data', (2,0,1))  #将图像的通道数设置为outermost的维数
# transformer.set_mean('data', mu)            #对于每个通道，都减去BGR的均值像素值
# transformer.set_raw_scale('data', 255)      #将像素值从[0,255]变换到[0,1]之间
# transformer.set_channel_swap('data', (2,1,0))  #交换通道，从RGB变换到BGR
# 
# 
# net.blobs['data'].reshape(4,        # batch 大小
#                           3,         # 3-channel (BGR) images
#                           227, 227)  # 图像大小为:227x227

image = caffe.io.load_image('../data/last/1001094.jpg')
transformed_image = transformer.preprocess('data', image)
# plt.imshow(image)
# plt.show()

# 将图像数据拷贝到为net分配的内存中
net.blobs['data'].data[...] = transformed_image
