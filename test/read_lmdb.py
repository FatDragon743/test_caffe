# -*- coding:utf-8 -*-  
'''
Created on 2018年7月13日

@author: Administrator
'''

import sys
import os
import numpy as np
caffe_root = 'E:/caffe27/'  #该文件要从路径{caffe_root}/examples下运行，否则要调整这一行。
sys.path.insert(0, caffe_root + 'python')
import caffe
import numpy as np
import lmdb
import argparse
from matplotlib import pyplot
 
if __name__ == '__main__':
    lmdbpath = '../data/tmp/train_lmdb'
 
    env = lmdb.open(lmdbpath, readonly=True)
    with env.begin() as txn:
        cursor = txn.cursor()
        for key, value in cursor:
            #print(key,len(value))#value是string类型
            print 'key: ',key
            datum = caffe.proto.caffe_pb2.Datum()#datum类型
            datum.ParseFromString(value)#转成datum
            flat_x = np.fromstring(datum.data, dtype=np.uint8)#转成numpy类型
            x = flat_x.reshape(datum.channels, datum.height, datum.width)#reshape大小
            y = datum.label#图片的label
            fig = pyplot.figure()#把两张图片显示出来
            ax = fig.add_subplot(121)
            ax.imshow(x[0], cmap='gray')
            ax = fig.add_subplot(122)
            ax.imshow(x[1], cmap='gray')
            pyplot.show()