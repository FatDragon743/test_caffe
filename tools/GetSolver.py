# -*- coding:utf-8 -*-  
'''
Created on 2018年7月9日

@author: Administrator
'''
import sys
caffe_root = 'E:/caffe27/'  #该文件要从路径{caffe_root}/examples下运行，否则要调整这一行。
sys.path.insert(0, caffe_root + 'python')
import caffe
from caffe.proto import caffe_pb2

s = caffe_pb2.SolverParameter()

path='../data/tmp/'
solver_file=path+'solver.prototxt'     #solver文件保存位置

s.train_net = path+'train.prototxt'     # 训练配置文件
s.test_net.append(path+'test.prototxt')  # 测试配置文件
s.test_interval = 782                   # 测试间隔
s.test_iter.append(313)                 # 测试迭代次数
s.max_iter = 1000                      # 最大迭代次数

s.base_lr = 0.0001                       # 基础学习率
s.momentum = 0.9                        # momentum系数
s.weight_decay = 0.0002                   # 权值衰减系数
s.lr_policy = 'step'                    # 学习率衰减方法
s.stepsize=26067                        # 此值仅对step方法有效
s.gamma = 0.1                           # 学习率衰减指数
s.display = 782                         # 屏幕日志显示间隔
s.snapshot = 7820
s.snapshot_prefix = 'shapshot'
s.type = 'SGD'                      # 优化算法
s.solver_mode = caffe_pb2.SolverParameter.GPU

with open(solver_file, 'w') as f:
    f.write(str(s))