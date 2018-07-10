# -*- coding:utf-8 -*-  
'''
Created on 2018年7月9日

@author: Administrator
'''
import sys
caffe_root = 'E:/caffe27/'  #该文件要从路径{caffe_root}/examples下运行，否则要调整这一行。
sys.path.insert(0, caffe_root + 'python')
import caffe
# 训练设置
# 使用GPU
gpu_id = 0
caffe.set_device(gpu_id) # 若不设置,默认为0
caffe.set_mode_gpu()
# # 使用CPU
# caffe.set_mode_cpu()

# 加载Solver，有两种常用方法
# 1. 无论模型中Slover类型是什么统一设置为SGD
# solver = caffe.SGDSolver('/home/xxx/data/solver.prototxt') 
# 2. 根据solver的prototxt中solver_type读取，默认为SGD
solver = caffe.get_solver('../data/tmp/solver.prototxt')

# 训练模型
# 1.1 前向传播
# solver.net.forward()  # train net
# solver.test_nets[0].forward()  # test net (there can be more than one)
# # 1.2 反向传播,计算梯度
# solver.net.backward()
# # 2. 进行一次前向传播一次反向传播并根据梯度更新参数
# solver.step(1)
# 3. 根据solver文件中设置进行完整model训练
solver.solve()