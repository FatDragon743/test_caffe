# -*- coding:utf-8 -*-  
'''
Created on 2018年7月9日

@author: Administrator
'''
import sys
import numpy as np # 加载numpy
# import matplotlib.pyplot as plt # 加载matplotlib

# rcParams是一个包含各种参数的字典结构，含有多个key-value，可修改其中部分值
# plt.rcParams['figure.figsize'] = (10, 10) # 图像显示大小，单位是英寸 
# plt.rcParams['image.interpolation'] = 'nearest'  # 最近邻差值,像素为正方形
# plt.rcParams['image.cmap'] = 'gray'  # 使用灰度输出而不是彩色输出



import os
caffe_root = 'E:/caffe27/'  #该文件要从路径{caffe_root}/examples下运行，否则要调整这一行。
sys.path.insert(0, caffe_root + 'python')
import caffe
os.chdir(caffe_root)


import os
if os.path.isfile(caffe_root + "models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel"):
    print ("CaffeNet found")
else:
    print ('Downloading pre-trained CaffeNet model...')
    
# caffe.set_mode_cpu()
model_def = caffe_root + 'models/bvlc_reference_caffenet/deploy.prototxt'
model_weights = caffe_root + 'models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel'

net = caffe.Net(model_def,      # 定义模型结构
                model_weights,  # 包含了模型的训练权值
                caffe.TEST)     # 使用测试模式(不执行dropout

# 加载ImageNet图像均值 (随着Caffe一起发布的)
mu = np.load(caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy')
mu = mu.mean(1).mean(1)  #对所有像素值取平均以此获取BGR的均值像素值
print ('mean-subtracted values:', zip('BGR', mu))

# 对输入数据进行变换
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})

transformer.set_transpose('data', (2,0,1))  #将图像的通道数设置为outermost的维数
transformer.set_mean('data', mu)            #对于每个通道，都减去BGR的均值像素值
transformer.set_raw_scale('data', 255)      #将像素值从[0,255]变换到[0,1]之间
transformer.set_channel_swap('data', (2,1,0))  #交换通道，从RGB变换到BGR


net.blobs['data'].reshape(50,        # batch 大小
                          3,         # 3-channel (BGR) images
                          227, 227)  # 图像大小为:227x227

image = caffe.io.load_image(caffe_root + 'examples/images/cat.jpg')
transformed_image = transformer.preprocess('data', image)
# plt.imshow(image)
# plt.show()

# 将图像数据拷贝到为net分配的内存中
net.blobs['data'].data[...] = transformed_image


from caffe import layers as L, params as P, to_proto
print to_proto(net)
exit()

### 执行分类
# output = net.forward()#
caffe.set_device(0)   # 使用第一块显卡
caffe.set_mode_gpu()  # 设为gpu模式
output = net.forward()         # 前向传播  
output_prob = output['prob'][0]  #batch中第一张图像的概率值   
print ('predicted class is:', output_prob.argmax())

# 加载ImageNet标签
labels_file = caffe_root + 'data/ilsvrc12/synset_words.txt'
if not os.path.exists(labels_file):
    print ("no synset_words")
    
labels = np.loadtxt(labels_file, str, delimiter='\t')
print ('output label:', labels[output_prob.argmax()])

# sort top five predictions from softmax output
top_inds = output_prob.argsort()[::-1][:5]  # reverse sort and take five largest items

print ('probabilities and labels:')
print (zip(output_prob[top_inds], labels[top_inds]))

# _sort=output_prob.argsort()
# # 查看CPU的分类时间，然后再与GPU进行比较
# #gpu模式下跑一次
# caffe.set_device(0)   # 使用第一块显卡
# caffe.set_mode_gpu()  # 设为gpu模式
# net.forward()         # 前向传播

