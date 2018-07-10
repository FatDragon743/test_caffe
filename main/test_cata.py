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

# 加载ImageNet图像均值 (随着Caffe一起发布的)
mu = np.load(path_root + 'train_mean.npy')
mu = mu.mean(1).mean(1)  #对所有像素值取平均以此获取BGR的均值像素值
print ('mean-subtracted values:', zip('BGR', mu))

# 对输入数据进行变换
transformer = caffe.io.Transformer({'data': net.blobs['Data1'].data.shape})
transformer.set_transpose('data', (2,0,1))  #将图像的通道数设置为outermost的维数
transformer.set_mean('data', mu)            #对于每个通道，都减去BGR的均值像素值
transformer.set_raw_scale('data', 255)      #将像素值从[0,255]变换到[0,1]之间
transformer.set_channel_swap('data', (2,1,0))  #交换通道，从RGB变换到BGR


net.blobs['Data1'].reshape(4,        # batch 大小
                          3,         # 3-channel (BGR) images
                          227, 227)  # 图像大小为:227x227

image = caffe.io.load_image('../data/last/1002178.jpg')
transformed_image = transformer.preprocess('data', image)
# plt.imshow(image)
# plt.show()

# 将图像数据拷贝到为net分配的内存中
net.blobs['Data1'].data[...] = transformed_image
#执行测试
caffe.set_device(0)   # 使用第一块显卡
caffe.set_mode_gpu()  # 设为gpu模式
out = net.forward()
labels_file = path_root + 'cata.txt'
labels = np.loadtxt(labels_file, str, delimiter='\t')   #读取类别名称文件
feat= net.blobs['InnerProduct3'].data[0]#取出最后一层（Softmax）属于某个类别的概率值，并打印

# print prob
order=feat.argsort()[0]  #将概率值排序，取出最大值所在的序号 
print 'the class is:',labels[order]   #将该序号转换成对应的类别名称，并打印
# 
# 取出前五个较大值所在的序号
top_inds = feat.argsort()[::-1][:5]
print 'probabilities and labels:' 
print str(zip(feat[top_inds], labels[top_inds]))
import matplotlib.pyplot as plt
plt.figure(figsize=(15, 3))
plt.plot(feat.flat)
plt.show()
# # 取出前五个较大值所在的序号
# top_inds = prob.argsort()[::-1][:5]
# print 'probabilities and labels:' +str(zip(prob[top_inds], labels[top_inds]))