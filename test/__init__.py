
# -*- coding:utf-8 -*-  
'''
Created on 2018年7月9日

@author: Administrator
'''
import sys
caffe_root = 'E:/caffe27/'  #该文件要从路径{caffe_root}/examples下运行，否则要调整这一行。
sys.path.insert(0, caffe_root + 'python')
import caffe
import matplotlib.pyplot as plt  
import caffe   
from numpy import *
caffe.set_device(0)  
caffe.set_mode_gpu()   
# 使用SGDSolver，即随机梯度下降算法  

  
# 等价于solver文件中的max_iter，即最大解算次数  
niter = 10000 

# 每隔100次收集一次loss数据  
display= 100  
  
# 每次测试进行100次解算 
test_iter = 100

# 每500次训练进行一次测试
test_interval =500
  
#初始化 
print ceil(niter * 1.0 / display)
train_loss = zeros(100)   
test_loss = zeros(ceil(niter * 1.0 / test_interval)*1.0)  
test_acc = zeros(ceil(niter * 1.0 / test_interval)*1.0)  
