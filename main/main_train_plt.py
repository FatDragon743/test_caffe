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

from numpy import *
caffe.set_device(0)  
caffe.set_mode_gpu()   
# 使用SGDSolver，即随机梯度下降算法  
solver = caffe.SGDSolver('../data/tmp/solver.prototxt')  
  
# 等价于solver文件中的max_iter，即最大解算次数  
niter = 3000

# 每隔100次收集一次loss数据  
display= 10  
  
# 每次测试进行100次解算 
test_iter = 1000

# 每500次训练进行一次测试
test_interval =100
  
#初始化 
train_loss = zeros(int(ceil(niter * 1.0 / display)))   
test_loss = zeros(int(ceil(niter * 1.0 / test_interval))) 
test_acc = zeros(int(ceil(niter * 1.0 / test_interval))) 
  
# 辅助变量  
_train_loss = 0; _test_loss = 0; _accuracy = 0  
# 进行解算  
for it in range(niter):  
    # 进行一次解算  
    solver.step(1)  
    # 统计train loss  
    _train_loss += solver.net.blobs['loss'].data  
    if it % display == 0:  
        # 计算平均train loss  
        train_loss[it // display] = _train_loss / display  
        _train_loss = 0  
  
    if it % test_interval == 0:  
        for test_it in range(test_iter):  
            # 进行一次测试  
            solver.test_nets[0].forward()  
            # 计算test loss  
            _test_loss += solver.test_nets[0].blobs['loss'].data  
            # 计算test accuracy  
            _accuracy += solver.test_nets[0].blobs['acc'].data  
        # 计算平均test loss  
        test_loss[it / test_interval] = _test_loss / test_iter  
        # 计算平均test accuracy  
        test_acc[it / test_interval] = _accuracy / test_iter  
        _test_loss = 0  
        _accuracy = 0  
solver.net.save('../data/tmp/mymodel.caffemodel')
# 绘制train loss、test loss和accuracy曲线  
print '\nplot the train loss and test accuracy\n'  
_, ax1 = plt.subplots()  
ax2 = ax1.twinx()  
  
# train loss -> 绿色  
ax1.plot(display * arange(len(train_loss)), train_loss, 'g')  
# test loss -> 黄色  
ax1.plot(test_interval * arange(len(test_loss)), test_loss, 'y')  
# test accuracy -> 红色  
ax2.plot(test_interval * arange(len(test_acc)), test_acc, 'r')  
  
ax1.set_xlabel('iteration')  
ax1.set_ylabel('loss')  
ax2.set_ylabel('accuracy')  
plt.show()